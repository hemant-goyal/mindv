"""
panel_builder.py — mindv Phase 1

Automated, computationally verified AMR panel generator.

Pipeline:
  1. Parse resistance mutations from CARD / AMRFinderPlus / WHO catalogue
  2. Fetch gene CDS coordinates from NCBI GFF3 (or user-supplied GFF)
  3. Map each protein-level mutation to all single-SNP genomic variants that produce it
  4. VERIFY each entry: extract reference codon from the actual FASTA, translate, confirm
  5. Emit a verified panel JSON in mindv format with full provenance metadata

The verification step is the key differentiator vs. every existing AMR database:
no other tool automatically cross-checks claimed AA annotations against the reference genome.

Usage (via CLI):
  mindv panel-build --source card   --card-json card.json
                    --organism "Mycobacterium tuberculosis"
                    --ref     NC_000962.3.fasta --gff NC_000962.3.gff3
                    --genes   rpoB,katG,gyrA,gyrB,embB,rpsL,rrs,inhA,pncA,eis
                    --output  configs/tb_h37rv.json

Author: Hemant Goyal
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import sys
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import pysam

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Genetic code (standard, codon → single-letter AA)
# ---------------------------------------------------------------------------
GENETIC_CODE: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Public alias used by tests and external code
CODON_TABLE = GENETIC_CODE

AA_3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Stop": "*",
}

COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)


def reverse_complement(seq: str) -> str:
    return complement(seq)[::-1]


def translate(codon: str) -> str:
    return GENETIC_CODE.get(codon.upper(), "?")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GeneAnnotation:
    """CDS boundaries for one gene on a given reference."""
    gene_name: str
    contig: str
    gene_start: int          # 1-based inclusive, genomic + strand
    gene_end: int            # 1-based inclusive, genomic + strand
    strand: str              # "+" or "-"
    locus_tag: str = ""
    product: str = ""
    # For minus-strand genes, gene_end may need calibration (see rpoB M. leprae).
    # panel_builder attempts auto-calibration via anchor-codon search.
    calibrated_end: Optional[int] = None

    @property
    def effective_end(self) -> int:
        return self.calibrated_end if self.calibrated_end is not None else self.gene_end


@dataclass
class ResistanceMutation:
    """Protein-level resistance mutation from a source database."""
    gene: str
    ref_aa: str              # single-letter, e.g. "S"
    codon_number: int        # protein position (1-based)
    alt_aa: str              # single-letter, e.g. "L"
    drug: str
    drug_class: str = ""
    evidence: str = ""       # e.g. "WHO grade 1", "CARD curated"
    source: str = ""         # "card" | "amrfinderplus" | "who_catalogue"
    hgvs_p: str = ""         # original annotation string, e.g. "p.Ser450Leu"


@dataclass
class VerificationResult:
    status: str              # "VERIFIED" | "FAILED" | "WARNING" | "PROMOTER/rRNA"
    extracted_codon: str     # 3-mer extracted from FASTA (or "N/A")
    extracted_aa: str        # translated from extracted codon (or "N/A")
    expected_ref_aa: str     # ref AA claimed by database
    message: str = ""

    @property
    def is_ok(self) -> bool:
        """True for VERIFIED and WARNING; False for FAILED."""
        return self.status in ("VERIFIED", "WARNING", "PROMOTER/rRNA")


@dataclass
class PanelEntry:
    """One row in the output panel JSON — a verified genomic SNP entry."""
    gene: str
    genomic_pos: int         # 1-based genomic position on + strand
    ref_nuc: str             # + strand reference nucleotide
    alt_nuc: str             # + strand alternate nucleotide
    aa_change: str           # e.g. "p.S450L"
    codon_change: str        # e.g. "TCG>TTG"
    drug: str
    drug_class: str = ""
    evidence: str = ""
    source: str = ""
    verification: VerificationResult = field(default_factory=lambda: VerificationResult("UNVERIFIED","","","",""))

    @property
    def panel_key(self) -> str:
        return f"{self.genomic_pos}_{self.alt_nuc}"


# ---------------------------------------------------------------------------
# Core codon arithmetic
# ---------------------------------------------------------------------------

def codon_genomic_positions(codon_number: int, gene: GeneAnnotation) -> Tuple[int, int, int]:
    """
    Return the three 1-based genomic positions of codon N (1-based),
    ordered 5'→3' on the mRNA strand.

    + strand:  positions ascend:  (gene_start + (N-1)*3,  +1,  +2)
    - strand:  positions descend: (effective_end - (N-1)*3,  -1,  -2)

    These are the positions you look up in the FASTA (+strand).
    For a minus-strand gene the first position returned is the HIGHEST
    genomic coordinate (the 5' end of the mRNA codon).
    """
    if gene.strand == "+":
        base = gene.gene_start + (codon_number - 1) * 3
        return (base, base + 1, base + 2)
    else:
        base = gene.effective_end - (codon_number - 1) * 3
        return (base, base - 1, base - 2)


def extract_ref_codon(fasta, gene: GeneAnnotation, codon_number: int) -> str:
    """
    Extract the reference codon (5'→3' on mRNA) from the FASTA.

    Parameters
    ----------
    fasta : pysam.FastaFile or str
        An open pysam.FastaFile, or a path string (the file will be opened
        and closed internally).

    Returns a 3-character uppercase string.
    Raises ValueError if the region is out of range.
    """
    _close_after = False
    if isinstance(fasta, str):
        if not os.path.exists(fasta + ".fai"):
            pysam.faidx(fasta)
        fasta = pysam.FastaFile(fasta)
        _close_after = True

    try:
        positions = codon_genomic_positions(codon_number, gene)
        contig = gene.contig

        if gene.strand == "+":
            lo = positions[0] - 1
            hi = positions[2]
            seq = fasta.fetch(contig, lo, hi).upper()
            if len(seq) != 3:
                raise ValueError(f"Could not fetch 3 bases at {contig}:{lo+1}-{hi}")
            return seq
        else:
            lo = positions[2] - 1   # lowest genomic pos, 0-based
            hi = positions[0]       # highest genomic pos (half-open)
            plus_seq = fasta.fetch(contig, lo, hi).upper()
            if len(plus_seq) != 3:
                raise ValueError(f"Could not fetch 3 bases at {contig}:{lo+1}-{hi}")
            return reverse_complement(plus_seq)
    finally:
        if _close_after:
            fasta.close()


def enumerate_snps(ref_codon: str, alt_aa: str) -> List[Tuple[int, str, str]]:
    """
    Find all single-nucleotide substitutions in ref_codon that produce alt_aa.
    Returns list of (codon_position_0based, ref_nuc_on_codon_strand, alt_nuc_on_codon_strand).
    codon_position_0based: 0, 1, or 2 counting from the 5' end of the mRNA codon.
    """
    results = []
    for pos in range(3):
        for alt_nuc in "ACGT":
            if alt_nuc == ref_codon[pos]:
                continue
            mut_codon = ref_codon[:pos] + alt_nuc + ref_codon[pos + 1:]
            if translate(mut_codon) == alt_aa:
                results.append((pos, ref_codon[pos], alt_nuc))
    return results


def codon_pos_to_genomic(codon_number: int, codon_pos_0based: int,
                          gene: GeneAnnotation) -> int:
    """
    Convert a codon position (0,1,2) to a 1-based genomic position.
    """
    positions = codon_genomic_positions(codon_number, gene)
    return positions[codon_pos_0based]


def to_plus_strand_nuc(nuc_on_codon_strand: str, strand: str) -> str:
    """
    Convert a nucleotide expressed on the mRNA strand to its + strand equivalent.
    For + strand genes: identity.
    For - strand genes: complement (NOT reverse complement — single base).
    """
    if strand == "+":
        return nuc_on_codon_strand
    return complement(nuc_on_codon_strand)


# ---------------------------------------------------------------------------
# Verification engine
# ---------------------------------------------------------------------------

def verify_entry(fasta: pysam.FastaFile, gene: GeneAnnotation,
                 codon_number: int, expected_ref_aa: str,
                 expected_alt_aa: str, codon_pos_0based: int) -> VerificationResult:
    """
    Extract the reference codon at codon_number, translate it, and confirm
    it matches expected_ref_aa. Also confirm that substituting at codon_pos_0based
    with the alt nucleotide produces expected_alt_aa.
    """
    try:
        ref_codon = extract_ref_codon(fasta, gene, codon_number)
    except Exception as exc:
        return VerificationResult(
            status="FAILED", extracted_codon="???", extracted_aa="?",
            expected_ref_aa=expected_ref_aa,
            message=f"FASTA extraction error: {exc}"
        )

    extracted_aa = translate(ref_codon)

    if extracted_aa != expected_ref_aa:
        return VerificationResult(
            status="FAILED",
            extracted_codon=ref_codon,
            extracted_aa=extracted_aa,
            expected_ref_aa=expected_ref_aa,
            message=(
                f"Codon {codon_number}: expected ref AA '{expected_ref_aa}', "
                f"got '{extracted_aa}' from codon '{ref_codon}'. "
                f"Check gene coordinates or codon numbering convention."
            ),
        )

    return VerificationResult(
        status="VERIFIED",
        extracted_codon=ref_codon,
        extracted_aa=extracted_aa,
        expected_ref_aa=expected_ref_aa,
        message="",
    )


# ---------------------------------------------------------------------------
# GFF3 parser
# ---------------------------------------------------------------------------

def parse_gff3(gff_path: str, target_genes: Optional[List[str]] = None) -> Dict[str, GeneAnnotation]:
    """
    Parse an NCBI GFF3 file and return a dict of gene_name → GeneAnnotation.

    Handles gzipped (.gz) files transparently.
    Prioritises CDS features; falls back to gene features.

    target_genes: if provided, only parse annotations for these gene names.
    """
    annotations: Dict[str, GeneAnnotation] = {}

    opener = gzip.open if gff_path.endswith(".gz") else open

    with opener(gff_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            contig, source, ftype, start, end, score, strand, phase, attrs = parts
            if ftype not in ("CDS", "gene"):
                continue

            # Parse attributes field (key=value;key=value;...)
            attr_dict: Dict[str, str] = {}
            for attr in attrs.split(";"):
                attr = attr.strip()
                if "=" in attr:
                    k, v = attr.split("=", 1)
                    attr_dict[k.strip()] = v.strip()

            gene_name = (
                attr_dict.get("gene", "")
                or attr_dict.get("Name", "")
                or attr_dict.get("locus_tag", "")
            )
            locus_tag = attr_dict.get("locus_tag", "")
            product = attr_dict.get("product", "")

            if not gene_name:
                continue
            if target_genes and gene_name not in target_genes:
                # Also check locus_tag
                if locus_tag not in target_genes:
                    continue

            start_1b = int(start)   # GFF3 is 1-based
            end_1b = int(end)

            key = gene_name or locus_tag
            # CDS overrides gene feature (more precise)
            if key not in annotations or ftype == "CDS":
                annotations[key] = GeneAnnotation(
                    gene_name=gene_name,
                    contig=contig,
                    gene_start=start_1b,
                    gene_end=end_1b,
                    strand=strand,
                    locus_tag=locus_tag,
                    product=product,
                )
                log.debug("  Loaded %s: %s %s:%d-%d (%s)", ftype, key, contig, start_1b, end_1b, strand)

    return annotations


# ---------------------------------------------------------------------------
# Source database parsers
# ---------------------------------------------------------------------------

class CardParser:
    """
    Parse CARD database JSON (card.json from https://card.mcmaster.ca/download).

    Extracts 'protein variant model' entries for a given organism, yielding
    ResistanceMutation objects.

    CARD SNP format example:
      snp: {"protein_change": "S531L", "position": 531, "change": "L"}
      where 531 is the amino acid position (1-based).
    """

    def __init__(self, card_json_path: str, organism: str):
        self.path = card_json_path
        # Normalise organism name for fuzzy matching
        self.organism_lower = organism.lower()
        self._data: Optional[dict] = None

    def _load(self):
        if self._data is None:
            log.info("Loading CARD JSON from %s ...", self.path)
            with open(self.path, "rt") as fh:
                self._data = json.load(fh)

    def parse(self) -> List[ResistanceMutation]:
        self._load()
        mutations: List[ResistanceMutation] = []

        for aro_id, entry in self._data.items():
            if not isinstance(entry, dict):
                continue
            model_type = entry.get("model_type", "")
            if model_type != "protein variant model":
                continue

            # Check organism
            taxon_data = entry.get("ARO_category", {})
            organisms = [v.get("category_aro_name", "").lower()
                         for v in taxon_data.values()
                         if v.get("category_aro_class_name", "") == "Species"]
            if not any(self.organism_lower in o for o in organisms):
                continue

            # Drug info
            drug_classes = [v.get("category_aro_name", "")
                            for v in taxon_data.values()
                            if v.get("category_aro_class_name", "") == "Drug Class"]
            drug = drug_classes[0] if drug_classes else "unknown"
            drug_class = "; ".join(drug_classes)

            gene_name = entry.get("model_name", "").split(" ")[0]

            # SNP entries
            snp_table = entry.get("model_param", {}).get("snp", {}).get("param_value", {})
            for snp_id, snp in snp_table.items():
                raw = snp.get("value", "")          # e.g. "S531L"
                m = re.match(r"^([A-Z\*])(\d+)([A-Z\*])$", raw)
                if not m:
                    log.debug("  CARD: unrecognised SNP format '%s' — skipping", raw)
                    continue
                ref_aa, codon_num, alt_aa = m.group(1), int(m.group(2)), m.group(3)
                mutations.append(ResistanceMutation(
                    gene=gene_name,
                    ref_aa=ref_aa,
                    codon_number=codon_num,
                    alt_aa=alt_aa,
                    drug=drug,
                    drug_class=drug_class,
                    evidence="CARD curated",
                    source="card",
                    hgvs_p=f"p.{ref_aa}{codon_num}{alt_aa}",
                ))

        log.info("CARD parser: %d mutations loaded for '%s'", len(mutations), self.organism_lower)
        return mutations


class AmrFinderParser:
    """
    Parse NCBI AMRFinderPlus mutation TSV.

    The file 'AMR_CDS.fa' + 'AMRProt' + tabular mutation_all.tsv are
    downloaded from https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/AMRFinderPlus/database/latest/

    Column order (relevant columns):
      0  gene_symbol   e.g. rpoB
      3  sequence_name e.g. RpoB with mutation conferring resistance to rifampin
      7  drug_class
      8  drug_subclass (specific drug)
      9  method        e.g. POINTX (point mutation based)
     10  target_length (aa)
     11  ref_name      reference protein accession
     16  element_type  AMR, STRESS, etc.
     17  element_subtype POINT
     18  class         drug class
     19  subclass
     20+ mutation details

    The point mutation format is: WP_XXXXXXXX.1:p.Ser531Leu
    """

    def __init__(self, tsv_path: str, organism: str):
        self.path = tsv_path
        self.organism_lower = organism.lower()

    def parse(self) -> List[ResistanceMutation]:
        mutations: List[ResistanceMutation] = []

        with open(self.path, "rt") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            col = {h: i for i, h in enumerate(header)}

            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < len(header):
                    continue

                method = parts[col.get("method", 9)] if col.get("method") else ""
                if "POINT" not in method.upper():
                    continue

                gene = parts[col.get("gene_symbol", 0)]
                drug_class = parts[col.get("class", 18)] if col.get("class") else ""
                drug = parts[col.get("subclass", 19)] if col.get("subclass") else drug_class

                # Mutation field — try "mutation" column or "sequence_name"
                mut_raw = parts[col.get("mutation", 20)] if "mutation" in col else ""
                if not mut_raw:
                    mut_raw = parts[col.get("sequence_name", 3)]

                # Parse HGVS-style: p.Ser531Leu or p.S531L
                m = re.search(r"p\.([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})", mut_raw)
                if not m:
                    continue

                ref_raw, codon_num, alt_raw = m.group(1), int(m.group(2)), m.group(3)
                ref_aa = AA_3TO1.get(ref_raw, ref_raw) if len(ref_raw) == 3 else ref_raw
                alt_aa = AA_3TO1.get(alt_raw, alt_raw) if len(alt_raw) == 3 else alt_raw

                mutations.append(ResistanceMutation(
                    gene=gene,
                    ref_aa=ref_aa,
                    codon_number=codon_num,
                    alt_aa=alt_aa,
                    drug=drug,
                    drug_class=drug_class,
                    evidence="AMRFinderPlus curated",
                    source="amrfinderplus",
                    hgvs_p=f"p.{ref_aa}{codon_num}{alt_aa}",
                ))

        log.info("AMRFinderPlus parser: %d mutations loaded", len(mutations))
        return mutations


class WhoCatalogueParser:
    """
    Parse WHO TB Mutation Catalogue v1 (2021) or v2 (2023).

    Handles the real exported TSV column names, which differ between editions:

    v1 / community TSV:
        gene | mutation | drug | confidence_grade

    v2 / WHO-UCN-GTB-2023.5 official export:
        drug | gene | mutation | effect | prediction |
        INITIAL CONFIDENCE GRADING | FINAL CONFIDENCE GRADING | ...

    Grade values may be plain integers ("1") or prefixed text
    ("1 - Assoc w R", "1-Assoc w R") — the parser extracts the leading digit.

    Mutation HGVS notation:
        Coding:    p.Ser450Leu  or  p.S450L
        Promoter:  c.-15C>T
        Synonymous: p.= (skipped)
        rRNA:      r.1401a>g
    """

    # All known column name variants, lower-cased, in preference order.
    # The first match found in the actual header wins.
    _GENE_COLS   = ["gene"]
    _MUT_COLS    = ["mutation", "variant", "hgvs_mutation"]
    _DRUG_COLS   = ["drug", "drug_name", "antibiotic"]
    _GRADE_COLS  = [
        "final confidence grading",        # WHO v2 official (preferred)
        "initial confidence grading",      # WHO v2 fallback
        "tier",                            # WHO-UCN-GTB-2023 actual column name
        "confidence_grade",                # community TSV v1
        "who_confidence",                  # alternate community name
        "confidence",
        "grade",
        "grading",
    ]
    # Effect values that mean "not a resistance mutation" — skip these rows
    _SKIP_EFFECTS = {
        "synonymous_variant", "synonymous variant",
        "intergenic_variant", "intergenic variant",
    }

    def __init__(self, tsv_path: str, min_grade: int = 2):
        self.path = tsv_path
        self.min_grade = min_grade

    @staticmethod
    def _find_col(col_map: Dict[str, int], candidates: List[str]) -> Optional[int]:
        """Return the column index for the first matching candidate name."""
        for name in candidates:
            if name in col_map:
                return col_map[name]
        return None

    @staticmethod
    def _parse_grade(raw: str) -> Optional[int]:
        """
        Extract integer grade from raw cell value.
        Handles: "1", "2 - Assoc w R", "1-Assoc w R", " 3 ", "2".
        Returns None if no integer found.
        """
        raw = raw.strip()
        m = re.match(r"^(\d+)", raw)
        return int(m.group(1)) if m else None

    @staticmethod
    def _find_real_header(path: str, max_scan: int = 10) -> int:
        """
        Scan up to max_scan lines to find the one containing gene + drug + mutation.
        The WHO-UCN-GTB-2023 official export has 2 metadata rows before the real header.
        Returns the 0-based line index of the real header (0 = first line).
        """
        with open(path, "rt", encoding="utf-8-sig") as fh:
            for i, line in enumerate(fh):
                if i >= max_scan:
                    break
                cols_lower = {c.strip().lower() for c in line.rstrip("\n").split("\t")}
                if "gene" in cols_lower and "drug" in cols_lower and                         ("mutation" in cols_lower or "variant" in cols_lower):
                    return i
        return 0

    def parse(self) -> List[ResistanceMutation]:
        mutations: List[ResistanceMutation] = []

        # Auto-detect real header row — skips WHO-UCN-GTB-2023 metadata lines
        header_line_idx = self._find_real_header(self.path)
        if header_line_idx > 0:
            log.info("WHO TSV: skipping %d metadata line(s) before real header",
                     header_line_idx)

        with open(self.path, "rt", encoding="utf-8-sig") as fh:
            for _ in range(header_line_idx):
                fh.readline()
            raw_header = fh.readline().rstrip("\n")
            header = raw_header.split("\t")
            col = {h.strip().lower(): i for i, h in enumerate(header)}

        log.info("WHO TSV real header (line %d): %s",
                 header_line_idx + 1, [h.strip() for h in header[:15]])

        # Resolve column indices
        idx_gene   = self._find_col(col, self._GENE_COLS)
        idx_mut    = self._find_col(col, self._MUT_COLS)
        idx_drug   = self._find_col(col, self._DRUG_COLS)
        idx_grade  = self._find_col(col, self._GRADE_COLS)
        idx_effect = col.get("effect", None)  # optional — for synonymous filtering

        missing = []
        if idx_gene  is None: missing.append("gene")
        if idx_mut   is None: missing.append("mutation")
        if idx_drug  is None: missing.append("drug")
        if idx_grade is None: missing.append("confidence grade")

        if missing:
            actual = [h.strip() for h in header]
            raise ValueError(
                f"WHO catalogue TSV is missing required columns: {missing}\n"
                f"Columns found in your file: {actual[:20]}\n"
                f"Expected variants — gene: {self._GENE_COLS}, "
                f"mutation: {self._MUT_COLS}, drug: {self._DRUG_COLS}, "
                f"grade: {self._GRADE_COLS}"
            )

        skipped_grade = skipped_syn = skipped_effect = skipped_fmt = 0

        with open(self.path, "rt", encoding="utf-8-sig") as fh:
            for _ in range(header_line_idx + 1):
                fh.readline()   # skip metadata + header
            for lineno, line in enumerate(fh, start=header_line_idx + 2):
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= max(idx_gene, idx_mut, idx_drug, idx_grade):
                    continue

                gene    = parts[idx_gene].strip()
                mut_raw = parts[idx_mut].strip()
                drug    = parts[idx_drug].strip()
                grade   = self._parse_grade(parts[idx_grade])

                if not gene or not mut_raw or not drug:
                    continue
                if grade is None or grade > self.min_grade:
                    skipped_grade += 1
                    continue

                # Skip synonymous/intergenic rows using the effect column
                if idx_effect is not None and idx_effect < len(parts):
                    effect_val = parts[idx_effect].strip().lower()
                    if effect_val in self._SKIP_EFFECTS:
                        skipped_effect += 1
                        continue

                evidence = f"WHO grade {grade}"

                # --- Protein mutation: p.Ser450Leu or p.S450L ---
                m = re.match(r"p\.([A-Za-z]{1,3})(\d+)([A-Za-z*]{1,3}|=)", mut_raw)
                if m:
                    ref_raw, codon_num, alt_raw = m.group(1), int(m.group(2)), m.group(3)
                    if alt_raw in ("=", "X"):
                        skipped_syn += 1
                        continue
                    ref_aa = AA_3TO1.get(ref_raw, ref_raw) if len(ref_raw) == 3 else ref_raw
                    alt_aa = AA_3TO1.get(alt_raw, alt_raw) if len(alt_raw) == 3 else alt_raw
                    mutations.append(ResistanceMutation(
                        gene=gene, ref_aa=ref_aa, codon_number=codon_num, alt_aa=alt_aa,
                        drug=drug, evidence=evidence, source="who_catalogue", hgvs_p=mut_raw,
                    ))
                    continue

                # --- True promoter: c.-15C>T  (negative offset = upstream) ---
                m = re.match(r"c\.(-\d+)([ACGT])>([ACGT])", mut_raw, re.IGNORECASE)
                if m:
                    offset = int(m.group(1))
                    ref_nuc, alt_nuc = m.group(2).upper(), m.group(3).upper()
                    mutations.append(ResistanceMutation(
                        gene=gene + "_promoter",
                        ref_aa=ref_nuc, codon_number=offset, alt_aa=alt_nuc,
                        drug=drug, evidence=evidence, source="who_catalogue", hgvs_p=mut_raw,
                    ))
                    continue

                # --- Coding sequence nucleotide: c.1349C>T (positive offset) ---
                # These are stored as nucleotide-level entries (like rrs variants)
                m = re.match(r"c\.(\d+)([ACGT])>([ACGT])", mut_raw, re.IGNORECASE)
                if m:
                    offset = int(m.group(1))
                    ref_nuc, alt_nuc = m.group(2).upper(), m.group(3).upper()
                    mutations.append(ResistanceMutation(
                        gene=gene + "_nt",
                        ref_aa=ref_nuc, codon_number=offset, alt_aa=alt_nuc,
                        drug=drug, evidence=evidence, source="who_catalogue", hgvs_p=mut_raw,
                    ))
                    continue

                # --- rRNA: r.1401a>g ---
                m = re.match(r"r\.(\d+)([acgu])>([acgu])", mut_raw, re.IGNORECASE)
                if m:
                    rna_pos = int(m.group(1))
                    ref_nuc, alt_nuc = m.group(2).upper(), m.group(3).upper()
                    mutations.append(ResistanceMutation(
                        gene=gene + "_rRNA",
                        ref_aa=ref_nuc, codon_number=rna_pos, alt_aa=alt_nuc,
                        drug=drug, evidence=evidence, source="who_catalogue", hgvs_p=mut_raw,
                    ))
                    continue

                skipped_fmt += 1

        log.info(
            "WHO catalogue: %d mutations loaded (grade ≤ %d) | "
            "skipped: %d grade-filtered, %d synonymous, %d synonymous-by-effect, "
            "%d unrecognised format",
            len(mutations), self.min_grade,
            skipped_grade, skipped_syn, skipped_effect, skipped_fmt,
        )
        if not mutations:
            log.warning(
                "Zero mutations parsed. Likely cause: column names in your TSV do not "
                "match expected names. Re-run with --debug to see detected columns, "
                "or inspect the first line of your TSV file."
            )
        return mutations


# ---------------------------------------------------------------------------
# VCF-based panel parser
# ---------------------------------------------------------------------------

class VcfPanelParser:
    """
    Build a panel directly from a VCF file of known resistance variants.

    Each VCF record is treated as a resistance-associated SNP.
    Drug/gene information is read from INFO fields (ANN=, DRUG=, GENE=)
    or from a companion TSV annotation file.

    Supports standard GATK/Freebayes/bcftools VCF output.
    INFO fields used (if present):
        GENE=rpoB               Gene name
        DRUG=rifampicin         Drug name
        AA_CHANGE=p.S450L       Pre-annotated amino acid change
        ANN=...                 SnpEff ANN field (parsed for gene/aa)
    """

    def __init__(self, vcf_path: str, annotation_tsv: Optional[str] = None,
                 default_drug: str = "unknown"):
        self.vcf_path = vcf_path
        self.annotation_tsv = annotation_tsv
        self.default_drug = default_drug

    def _load_annotation_tsv(self) -> Dict[str, Dict]:
        """
        Optional companion TSV: pos TAB ref TAB alt TAB gene TAB drug TAB aa_change
        Returns dict keyed by "POS_ALT".
        """
        annot: Dict[str, Dict] = {}
        if not self.annotation_tsv or not os.path.exists(self.annotation_tsv):
            return annot
        with open(self.annotation_tsv) as fh:
            header = fh.readline().rstrip("\n").split("\t")
            col = {h.strip().lower(): i for i, h in enumerate(header)}
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                pos = parts[col.get("pos", col.get("position", 0))].strip()
                alt = parts[col.get("alt", 2)].strip().upper()
                key = f"{pos}_{alt}"
                annot[key] = {
                    "gene":      parts[col.get("gene", 3)].strip() if "gene" in col else "",
                    "drug":      parts[col.get("drug", 4)].strip() if "drug" in col else "",
                    "aa_change": parts[col.get("aa_change", 5)].strip() if "aa_change" in col else "",
                }
        return annot

    @staticmethod
    def _parse_info(info_str: str) -> Dict[str, str]:
        d: Dict[str, str] = {}
        for token in info_str.split(";"):
            if "=" in token:
                k, v = token.split("=", 1)
                d[k.strip()] = v.strip()
            else:
                d[token.strip()] = "true"
        return d

    @staticmethod
    def _parse_snpeff_ann(ann_str: str) -> Tuple[str, str]:
        """Extract gene name and HGVS p. annotation from SnpEff ANN field."""
        # ANN=ALT|effect|impact|GENE|...|HGVS_c|HGVS_p|...
        gene = ""
        hgvs_p = ""
        for allele_ann in ann_str.split(","):
            fields = allele_ann.split("|")
            if len(fields) >= 4:
                gene = fields[3].strip()
            if len(fields) >= 11:
                hgvs_p = fields[10].strip()
            if gene:
                break
        return gene, hgvs_p

    def parse(self) -> List[ResistanceMutation]:
        """
        Parse VCF and return ResistanceMutation list.
        SNPs only (len(REF)==1 and len(ALT)==1).
        """
        annot = self._load_annotation_tsv()
        mutations: List[ResistanceMutation] = []
        skipped_indel = skipped_multiallelic = 0

        with open(self.vcf_path, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue

                chrom, pos_str, _id, ref, alt_field = parts[:5]
                pos = int(pos_str)

                # Skip indels and multi-allelic records
                alt_alleles = [a.strip() for a in alt_field.split(",")]
                alt_alleles = [a for a in alt_alleles if a not in (".", "*")]

                if len(ref) != 1:
                    skipped_indel += 1
                    continue
                if len(alt_alleles) != 1:
                    skipped_multiallelic += 1
                    continue

                alt = alt_alleles[0]
                if len(alt) != 1:
                    skipped_indel += 1
                    continue

                # INFO field
                info_str = parts[7] if len(parts) > 7 else "."
                info = self._parse_info(info_str) if info_str != "." else {}

                # Lookup in companion TSV first
                key = f"{pos}_{alt.upper()}"
                annot_entry = annot.get(key, {})

                # Gene: INFO GENE= > SnpEff ANN > companion TSV > unknown
                gene = info.get("GENE", "")
                hgvs_p = info.get("AA_CHANGE", info.get("HGVS_P", ""))
                if not gene and "ANN" in info:
                    gene, hgvs_p_ann = self._parse_snpeff_ann(info["ANN"])
                    if not hgvs_p:
                        hgvs_p = hgvs_p_ann
                if not gene:
                    gene = annot_entry.get("gene", "unknown")
                if not hgvs_p:
                    hgvs_p = annot_entry.get("aa_change", "")

                drug = info.get("DRUG", annot_entry.get("drug", self.default_drug))

                # Parse HGVS p. if present to get ref_aa/codon/alt_aa
                ref_aa, codon_num, alt_aa = "?", pos, "?"
                if hgvs_p:
                    mp = re.match(r"p\.([A-Za-z]{1,3})(\d+)([A-Za-z*]{1,3})", hgvs_p)
                    if mp:
                        r_raw, codon_num, a_raw = mp.group(1), int(mp.group(2)), mp.group(3)
                        ref_aa = AA_3TO1.get(r_raw, r_raw) if len(r_raw) == 3 else r_raw
                        alt_aa = AA_3TO1.get(a_raw, a_raw) if len(a_raw) == 3 else a_raw

                mutations.append(ResistanceMutation(
                    gene=gene,
                    ref_aa=ref_aa,
                    codon_number=codon_num,
                    alt_aa=alt_aa,
                    drug=drug,
                    evidence="VCF",
                    source="vcf",
                    hgvs_p=hgvs_p or f"g.{pos}{ref}>{alt}",
                ))

        log.info(
            "VCF parser: %d SNP mutations loaded | skipped %d indels, %d multi-allelic",
            len(mutations), skipped_indel, skipped_multiallelic,
        )
        return mutations


# ---------------------------------------------------------------------------
# Auto-calibration of gene_end for minus-strand genes
# ---------------------------------------------------------------------------

ANCHOR_MOTIFS: Dict[str, Tuple[str, int]] = {
    # gene_name → (anchor_single_letter_aa, codon_number_in_published_numbering)
    # Universally conserved residues used to calibrate gene_end for minus-strand genes.
    # rpoB H37Rv: S450 is the RRDR anchor (≡ E. coli S531).
    #   + strand at 761976–761978 = CGA → rev-comp = TCG = Ser ✓
    "rpoB": ("S", 450),
}

def calibrate_gene_end(fasta: pysam.FastaFile, gene: GeneAnnotation,
                        anchor_aa: str, anchor_codon: int,
                        search_window: int = 30) -> Optional[int]:
    """
    Attempt to auto-calibrate gene_end for a minus-strand gene by scanning
    ±search_window codons around the NCBI-annotated end to find gene_end such that
    codon anchor_codon translates to anchor_aa.

    Returns the calibrated gene_end (1-based), or None if calibration fails.
    Only applicable to minus-strand genes.
    """
    if gene.strand != "-":
        return None

    for delta in range(-search_window * 3, search_window * 3 + 1, 3):
        candidate_end = gene.gene_end + delta
        candidate_gene = GeneAnnotation(
            gene_name=gene.gene_name, contig=gene.contig,
            gene_start=gene.gene_start, gene_end=gene.gene_end,
            strand=gene.strand, calibrated_end=candidate_end,
        )
        try:
            codon = extract_ref_codon(fasta, candidate_gene, anchor_codon)
            if translate(codon) == anchor_aa:
                log.info(
                    "  calibrate_gene_end(%s): calibrated_end=%d (delta=%+d) → codon %d = %s (%s) ✓",
                    gene.gene_name, candidate_end, delta, anchor_codon, codon, anchor_aa,
                )
                return candidate_end
        except Exception:
            continue

    log.warning(
        "  calibrate_gene_end(%s): could not anchor codon %d to '%s' within ±%d bp",
        gene.gene_name, anchor_codon, anchor_aa, search_window * 3,
    )
    return None


# ---------------------------------------------------------------------------
# Panel builder — main orchestrator
# ---------------------------------------------------------------------------

class PanelBuilder:
    """
    Orchestrates the full panel construction pipeline.

    Parameters
    ----------
    mutations : list[ResistanceMutation]
        Pre-parsed mutations from CardParser / AmrFinderParser / WhoCatalogueParser.
    annotations : dict[str, GeneAnnotation]
        Gene annotations, keyed by gene name.
    fasta_path : str
        Path to the reference FASTA (must have .fai index or be indexable).
    organism : str
        Organism name for metadata.
    reference_accession : str
        Reference accession for metadata (e.g. "NC_000962.3").
    panel_version : str
        Version string for output JSON.
    """

    def __init__(
        self,
        mutations: List[ResistanceMutation],
        annotations: Dict[str, GeneAnnotation],
        fasta_path: str,
        organism: str = "Unknown",
        reference_accession: str = "Unknown",
        panel_version: str = "1.0",
    ):
        self.mutations = mutations
        self.annotations = annotations
        self.fasta_path = fasta_path
        self.organism = organism
        self.reference_accession = reference_accession
        self.panel_version = panel_version
        self._fasta: Optional[pysam.FastaFile] = None

    def _open_fasta(self) -> pysam.FastaFile:
        if self._fasta is None:
            if not os.path.exists(self.fasta_path + ".fai"):
                log.info("Indexing reference FASTA with pysam...")
                pysam.faidx(self.fasta_path)
            self._fasta = pysam.FastaFile(self.fasta_path)
        return self._fasta

    def _close_fasta(self):
        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None

    # ------------------------------------------------------------------
    # Auto-calibrate gene ends for minus-strand genes
    # ------------------------------------------------------------------
    def _maybe_calibrate(self, gene: GeneAnnotation) -> GeneAnnotation:
        """
        Attempt gene_end calibration for minus-strand genes where we have a
        known anchor (ANCHOR_MOTIFS).  Returns the same GeneAnnotation object,
        possibly with calibrated_end set.
        """
        if gene.strand != "-":
            return gene
        if gene.gene_name not in ANCHOR_MOTIFS:
            return gene

        anchor_aa, anchor_codon = ANCHOR_MOTIFS[gene.gene_name]

        fasta = self._open_fasta()
        cal_end = calibrate_gene_end(fasta, gene, anchor_aa, anchor_codon)
        if cal_end is not None:
            gene.calibrated_end = cal_end
        return gene

    # ------------------------------------------------------------------
    # Helpers exposed for testing
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(entries: List["PanelEntry"]) -> List["PanelEntry"]:
        """
        Deduplicate a list of PanelEntry objects by panel_key (POSITION_ALT).
        When two entries share the same key, the first one wins.
        Used internally and exposed for unit testing.
        """
        seen: Dict[str, "PanelEntry"] = {}
        for entry in entries:
            if entry.panel_key not in seen:
                seen[entry.panel_key] = entry
        return list(seen.values())

    def _process_mutation(
        self, mut: "ResistanceMutation", gene: "GeneAnnotation"
    ) -> List["PanelEntry"]:
        """
        Map a single ResistanceMutation to one or more PanelEntry objects.

        This is the core per-mutation logic, extracted from build() so it
        can be called directly from tests without running the full pipeline.

        Returns a list (may be empty if no SNP route exists or verification fails).
        """
        fasta = self._open_fasta()
        entries: List[PanelEntry] = []

        try:
            ref_codon = extract_ref_codon(fasta, gene, mut.codon_number)
        except Exception as exc:
            log.warning("_process_mutation: could not extract codon for %s %s: %s",
                        mut.gene, mut.hgvs_p, exc)
            return entries

        snps = enumerate_snps(ref_codon, mut.alt_aa)
        for codon_pos_0b, ref_nuc_codon, alt_nuc_codon in snps:
            genomic_pos = codon_pos_to_genomic(mut.codon_number, codon_pos_0b, gene)
            ref_nuc_plus = to_plus_strand_nuc(ref_nuc_codon, gene.strand)
            alt_nuc_plus = to_plus_strand_nuc(alt_nuc_codon, gene.strand)

            mut_codon_list = list(ref_codon)
            mut_codon_list[codon_pos_0b] = alt_nuc_codon
            mut_codon = "".join(mut_codon_list)
            codon_change = f"{ref_codon}>{mut_codon}"
            aa_change = f"p.{mut.ref_aa}{mut.codon_number}{mut.alt_aa}"

            v = verify_entry(
                fasta, gene, mut.codon_number,
                expected_ref_aa=mut.ref_aa,
                expected_alt_aa=mut.alt_aa,
                codon_pos_0based=codon_pos_0b,
            )

            entries.append(PanelEntry(
                gene=mut.gene,
                genomic_pos=genomic_pos,
                ref_nuc=ref_nuc_plus,
                alt_nuc=alt_nuc_plus,
                aa_change=aa_change,
                codon_change=codon_change,
                drug=mut.drug,
                drug_class=mut.drug_class,
                evidence=mut.evidence,
                source=mut.source,
                verification=v,
            ))

        return entries

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, fail_on_verification_error: bool = False) -> dict:
        """
        Run the full panel-build pipeline.

        Returns a dict ready to be written as JSON (the mindv panel format).
        """
        fasta = self._open_fasta()

        # Step 1: Calibrate gene ends
        for gname, gene in self.annotations.items():
            self.annotations[gname] = self._maybe_calibrate(gene)

        # Step 2: Deduplicate mutations (same gene + codon + alt_aa → keep highest evidence)
        seen: Dict[Tuple[str, int, str], ResistanceMutation] = {}
        for mut in self.mutations:
            key = (mut.gene, mut.codon_number, mut.alt_aa)
            if key not in seen:
                seen[key] = mut
            else:
                # Prefer WHO > CARD > AMRFinderPlus; prefer lower grade number
                grade_priority = {"who_catalogue": 0, "card": 1, "amrfinderplus": 2}
                if grade_priority.get(mut.source, 9) < grade_priority.get(seen[key].source, 9):
                    seen[key] = mut
        mutations = list(seen.values())

        log.info("Building panel: %d unique protein-level mutations across %d genes",
                 len(mutations), len(set(m.gene for m in mutations)))

        # Step 3: Map to genomic and verify
        panel_entries: List[PanelEntry] = []
        stats = {"verified": 0, "failed": 0, "skipped_no_annotation": 0,
                 "skipped_no_snp": 0, "promoter": 0}

        for mut in mutations:
            # --- Promoter / rRNA mutations (stored as nucleotide-level) ---
            if mut.gene.endswith("_promoter") or mut.gene.endswith("_rRNA"):
                base_gene = mut.gene.replace("_promoter", "").replace("_rRNA", "")
                gene = self.annotations.get(base_gene)
                if gene is None:
                    log.debug("  No annotation for %s — skipping", base_gene)
                    stats["skipped_no_annotation"] += 1
                    continue
                # Promoter: codon_number is offset from gene start (negative = upstream)
                if mut.gene.endswith("_promoter"):
                    if gene.strand == "+":
                        genomic_pos = gene.gene_start + mut.codon_number
                    else:
                        genomic_pos = gene.gene_start - mut.codon_number
                    ref_nuc = mut.ref_aa  # stored in ref_aa field
                    alt_nuc = mut.alt_aa  # stored in alt_aa field
                # rRNA: codon_number is rRNA position (E. coli numbering)
                # Genomic mapping requires an anchor offset — we store as-is and flag
                else:
                    genomic_pos = gene.gene_start + mut.codon_number - 1
                    ref_nuc = mut.ref_aa
                    alt_nuc = mut.alt_aa

                entry = PanelEntry(
                    gene=mut.gene,
                    genomic_pos=genomic_pos,
                    ref_nuc=ref_nuc,
                    alt_nuc=alt_nuc,
                    aa_change=mut.hgvs_p,
                    codon_change=f"{ref_nuc}>{alt_nuc}",
                    drug=mut.drug,
                    drug_class=mut.drug_class,
                    evidence=mut.evidence,
                    source=mut.source,
                    verification=VerificationResult(
                        status="PROMOTER/rRNA",
                        extracted_codon="N/A",
                        extracted_aa="N/A",
                        expected_ref_aa=ref_nuc,
                        message="Promoter or rRNA position — nucleotide-level verification required",
                    ),
                )
                panel_entries.append(entry)
                stats["promoter"] += 1
                continue

            # --- Protein-coding mutations ---
            gene = self.annotations.get(mut.gene)
            if gene is None:
                log.debug("  No annotation for gene '%s' — skipping %s", mut.gene, mut.hgvs_p)
                stats["skipped_no_annotation"] += 1
                continue

            # Enumerate all SNPs that produce this AA change
            try:
                ref_codon = extract_ref_codon(fasta, gene, mut.codon_number)
            except Exception as exc:
                log.warning("  Could not extract codon for %s %s: %s", mut.gene, mut.hgvs_p, exc)
                stats["failed"] += 1
                continue

            snps = enumerate_snps(ref_codon, mut.alt_aa)
            if not snps:
                log.warning(
                    "  %s %s: no single SNP in codon '%s' produces %s→%s",
                    mut.gene, mut.hgvs_p, ref_codon, mut.ref_aa, mut.alt_aa,
                )
                stats["skipped_no_snp"] += 1
                continue

            for codon_pos_0b, ref_nuc_codon, alt_nuc_codon in snps:
                genomic_pos = codon_pos_to_genomic(mut.codon_number, codon_pos_0b, gene)
                ref_nuc_plus = to_plus_strand_nuc(ref_nuc_codon, gene.strand)
                alt_nuc_plus = to_plus_strand_nuc(alt_nuc_codon, gene.strand)

                # Build mutant codon string for annotation
                mut_codon_list = list(ref_codon)
                mut_codon_list[codon_pos_0b] = alt_nuc_codon
                mut_codon = "".join(mut_codon_list)
                codon_change = f"{ref_codon}>{mut_codon}"
                aa_change = f"p.{mut.ref_aa}{mut.codon_number}{mut.alt_aa}"

                # Verification
                v = verify_entry(
                    fasta, gene, mut.codon_number,
                    expected_ref_aa=mut.ref_aa,
                    expected_alt_aa=mut.alt_aa,
                    codon_pos_0based=codon_pos_0b,
                )

                if v.status == "FAILED":
                    log.error("  VERIFICATION FAILED: %s %s — %s", mut.gene, aa_change, v.message)
                    stats["failed"] += 1
                    if fail_on_verification_error:
                        raise RuntimeError(f"Verification failed for {mut.gene} {aa_change}: {v.message}")
                    continue

                stats["verified"] += 1
                entry = PanelEntry(
                    gene=mut.gene,
                    genomic_pos=genomic_pos,
                    ref_nuc=ref_nuc_plus,
                    alt_nuc=alt_nuc_plus,
                    aa_change=aa_change,
                    codon_change=codon_change,
                    drug=mut.drug,
                    drug_class=mut.drug_class,
                    evidence=mut.evidence,
                    source=mut.source,
                    verification=v,
                )
                panel_entries.append(entry)

        self._close_fasta()

        log.info(
            "Panel build complete: %d entries verified | %d failed | %d skipped (no gene) | "
            "%d skipped (no SNP route) | %d promoter/rRNA",
            stats["verified"], stats["failed"],
            stats["skipped_no_annotation"], stats["skipped_no_snp"], stats["promoter"],
        )

        return self._format_panel(panel_entries, stats)

    # ------------------------------------------------------------------
    # Format output JSON
    # ------------------------------------------------------------------
    def _format_panel(self, entries: List[PanelEntry], stats: dict) -> dict:
        """
        Convert PanelEntry list → mindv panel JSON dict.

        Groups entries by gene, sorts by genomic position within each gene.
        Duplicate position_alt keys within the same gene are impossible by design
        (each SNP produces a unique genomic_pos + alt_nuc pair).
        """
        # Group by gene
        by_gene: Dict[str, List[PanelEntry]] = {}
        for e in entries:
            by_gene.setdefault(e.gene, []).append(e)

        targets = {}
        for gene_name, gene_entries in by_gene.items():
            gene = self.annotations.get(gene_name.replace("_promoter","").replace("_rRNA",""))
            positions = sorted(e.genomic_pos for e in gene_entries)
            scan_start = max(1, min(positions) - 20)
            scan_end = max(positions) + 20

            known_mutations = {}
            for e in sorted(gene_entries, key=lambda x: x.genomic_pos):
                key = e.panel_key  # "position_altNuc"
                known_mutations[key] = {
                    "ref": e.ref_nuc,
                    "alt": e.alt_nuc,
                    "aa_change": e.aa_change,
                    "codon": e.codon_change,
                    "drug": e.drug,
                    "drug_class": e.drug_class,
                    "evidence": e.evidence,
                    "source": e.source,
                    "verification": {
                        "status": e.verification.status,
                        "ref_codon_extracted": e.verification.extracted_codon,
                        "ref_aa_extracted": e.verification.extracted_aa,
                    },
                }

            target_entry = {
                "start": scan_start,
                "end": scan_end,
                "known_mutations": known_mutations,
            }
            if gene:
                target_entry.update({
                    "gene_start": gene.gene_start,
                    "gene_end": gene.effective_end,
                    "strand": gene.strand,
                })
                if gene.calibrated_end is not None:
                    target_entry["calibration_note"] = (
                        f"gene_end calibrated from NCBI annotated {gene.gene_end} "
                        f"to {gene.calibrated_end}"
                    )
            targets[gene_name] = target_entry

        panel = {
            "organism": self.organism,
            "reference": self.reference_accession,
            "panel_version": self.panel_version,
            "panel_build_stats": stats,
            "targets": targets,
        }
        return panel


# ---------------------------------------------------------------------------
# NCBI GFF3 fetcher (optional convenience function)
# ---------------------------------------------------------------------------

def fetch_ncbi_gff3(accession: str, outpath: str):
    """
    Download GFF3 from NCBI FTP for a given assembly accession.
    Example: fetch_ncbi_gff3("GCF_000195855.1", "ref.gff3.gz")
    """
    base = f"https://ftp.ncbi.nlm.nih.gov/genomes/all/{accession[:3]}/{accession[4:7]}/{accession[7:10]}/{accession[10:13]}"
    # NCBI path structure for GCF_ accessions
    url = f"{base}/{accession}_genomic.gff.gz"
    log.info("Fetching GFF3 from %s ...", url)
    urllib.request.urlretrieve(url, outpath)
    log.info("Saved to %s", outpath)


# ---------------------------------------------------------------------------
# Helper: read the first contig name from FASTA (via .fai or header scan)
# ---------------------------------------------------------------------------

def _read_contig_from_fasta(fasta_path: str) -> str:
    """
    Return the first sequence name from a FASTA file.

    Reads from the .fai index if present (one file-open, first line, first
    tab-delimited field) — this is instant even for 4 GB genomes.

    Falls back to scanning the FASTA header lines if no .fai exists.

    This is the correct way to get the contig name for the panel JSON —
    it matches exactly what BWA-MEM writes into BAM @SQ headers, so
    pysam.pileup() will never see a contig mismatch.

    Example
    -------
    GCF_000195955.2_ASM19595v2_genomic.fna  →  NC_000962.3
    NC_002677.1.fasta                       →  NC_002677.1
    """
    fai_path = fasta_path + ".fai"
    if os.path.isfile(fai_path):
        with open(fai_path) as fh:
            first_line = fh.readline()
            if first_line:
                contig = first_line.split("\t")[0].strip()
                if contig:
                    log.info("Contig name from .fai: %s", contig)
                    return contig

    # Fallback: scan FASTA for first header
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                # ">NC_000962.3 Mycobacterium ..." → "NC_000962.3"
                contig = line[1:].split()[0]
                log.info("Contig name from FASTA header: %s", contig)
                return contig

    # Last resort: use filename (old behaviour, but warn loudly)
    fallback = os.path.basename(fasta_path).replace(".fasta","").replace(".fna","").replace(".fa","")
    log.warning(
        "Could not read contig name from %s or %s — "
        "using filename '%s'. This may cause a contig mismatch error during scan. "
        "Run: samtools faidx %s",
        fasta_path, fai_path, fallback, fasta_path
    )
    return fallback


# ---------------------------------------------------------------------------
# Public entry point (called by cli.py)
# ---------------------------------------------------------------------------

def run_panel_build(
    source: str,
    organism: str,
    ref_fasta: str,
    gff_path: str,
    output_path: str,
    card_json: Optional[str] = None,
    amrfinder_tsv: Optional[str] = None,
    who_catalogue_tsv: Optional[str] = None,
    vcf_path: Optional[str] = None,
    vcf_annotation_tsv: Optional[str] = None,
    genes: Optional[List[str]] = None,
    who_min_grade: int = 2,
    panel_version: str = "1.0",
    reference_accession: str = "",
    fail_on_error: bool = False,
    **kwargs,
):
    """
    High-level entry point called by `mindv panel-build`.

    source : "card" | "amrfinderplus" | "who_catalogue" | "vcf" | "combined"
    """
    # --- Parse source ---
    mutations: List[ResistanceMutation] = []

    if source in ("card", "combined"):
        if not card_json:
            raise ValueError("--card-json required for --source card")
        mutations += CardParser(card_json, organism).parse()

    if source in ("amrfinderplus", "combined"):
        if not amrfinder_tsv:
            raise ValueError("--amrfinder-tsv required for --source amrfinderplus")
        mutations += AmrFinderParser(amrfinder_tsv, organism).parse()

    if source in ("who_catalogue", "combined"):
        if not who_catalogue_tsv:
            raise ValueError("--who-catalogue-tsv required for --source who_catalogue")
        mutations += WhoCatalogueParser(who_catalogue_tsv, who_min_grade).parse()

    if source in ("vcf", "combined"):
        if not vcf_path:
            raise ValueError("--vcf required for --source vcf")
        mutations += VcfPanelParser(vcf_path, vcf_annotation_tsv).parse()

    if not mutations:
        raise RuntimeError("No mutations parsed from source — check input files and organism name")

    # Filter to requested genes
    if genes:
        mutations = [m for m in mutations
                     if m.gene in genes
                     or m.gene.replace("_promoter","").replace("_rRNA","") in genes]
        log.info("After gene filter: %d mutations remain", len(mutations))

    # --- Parse GFF3 ---
    gene_names = list(set(
        m.gene.replace("_promoter", "").replace("_rRNA", "")
        for m in mutations
    ))
    annotations = parse_gff3(gff_path, target_genes=gene_names)
    log.info("Gene annotations loaded: %s", list(annotations.keys()))

    if not annotations:
        raise RuntimeError("No gene annotations found in GFF3 for the requested genes")

    # --- Build ---
    if not reference_accession:
        reference_accession = _read_contig_from_fasta(ref_fasta)

    builder = PanelBuilder(
        mutations=mutations,
        annotations=annotations,
        fasta_path=ref_fasta,
        organism=organism,
        reference_accession=reference_accession,
        panel_version=panel_version,
    )
    panel = builder.build(fail_on_verification_error=fail_on_error)

    # --- Write ---
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(panel, fh, indent=4)

    n_verified = panel["panel_build_stats"]["verified"]
    n_failed   = panel["panel_build_stats"]["failed"]
    log.info("Panel written to %s (%d entries, %d failed verification)",
             output_path, n_verified, n_failed)
    return panel
