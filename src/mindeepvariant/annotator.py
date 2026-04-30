"""
annotator.py — The Biological Translator & Tiered Annotation Engine

This module performs two critical functions:

1. CODON TRANSLATION: Converts a DNA-level mutation (e.g., pos 7589, G→T)
   into a protein-level consequence (e.g., p.D91Y, Missense).
   Handles both forward (+) and reverse (-) strand genes.

2. TIERED ANNOTATION: Classifies each variant as:
   - Tier 1 (KNOWN):  Matches a catalogued resistance mutation from the panel JSON.
   - Tier 2 (NOVEL):  Called by the CNN in a drug-resistance region but NOT
                       in the known mutation database. Candidate for investigation.

This two-tier system is the key differentiator from pure "lookup" tools like
NTM-Profiler or Mykrobe, which can only report known variants.
"""

import json
import logging
from dataclasses import dataclass, field

from Bio.Seq import Seq

import pysam

logger = logging.getLogger(__name__)


# Standard codon table (NCBI translation table 1)
# Used as fallback when Biopython is unavailable
CODON_TABLE = {
    "ATA": "I", "ATC": "I", "ATT": "I", "ATG": "M",
    "ACA": "T", "ACC": "T", "ACG": "T", "ACT": "T",
    "AAC": "N", "AAT": "N", "AAA": "K", "AAG": "K",
    "AGC": "S", "AGT": "S", "AGA": "R", "AGG": "R",
    "CTA": "L", "CTC": "L", "CTG": "L", "CTT": "L",
    "CCA": "P", "CCC": "P", "CCG": "P", "CCT": "P",
    "CAC": "H", "CAT": "H", "CAA": "Q", "CAG": "Q",
    "CGA": "R", "CGC": "R", "CGG": "R", "CGT": "R",
    "GTA": "V", "GTC": "V", "GTG": "V", "GTT": "V",
    "GCA": "A", "GCC": "A", "GCG": "A", "GCT": "A",
    "GAC": "D", "GAT": "D", "GAA": "E", "GAG": "E",
    "GGA": "G", "GGC": "G", "GGG": "G", "GGT": "G",
    "TCA": "S", "TCC": "S", "TCG": "S", "TCT": "S",
    "TTC": "F", "TTT": "F", "TTA": "L", "TTG": "L",
    "TAC": "Y", "TAT": "Y", "TAA": "*", "TAG": "*",
    "TGC": "C", "TGT": "C", "TGA": "*", "TGG": "W",
}


@dataclass
class AnnotatedVariant:
    """A variant call enriched with protein-level and tiered annotation."""

    # From the scanner
    patient_id: str
    gene: str
    contig: str
    position: int
    ref_base: str
    alt_base: str
    ref_depth: int
    alt_depth: int
    total_depth: int
    allele_frequency: float
    predicted_class: int
    class_name: str
    confidence: float

    # From the annotator
    aa_change: str = ""
    mutation_type: str = ""  # Missense, Synonymous, Nonsense, Indel
    impact: str = ""         # HIGH, MODERATE, LOW
    tier: str = ""           # KNOWN, NOVEL
    known_drug: str = ""     # Drug association if Tier 1
    known_literature: str = ""  # Citation if Tier 1


def _translate_codon(codon_seq):
    """Translate a 3-letter codon to amino acid using Biopython, with fallback."""
    try:
        return str(Seq(codon_seq).translate())
    except Exception:
        return CODON_TABLE.get(codon_seq.upper(), "?")


def get_amino_acid_change(fasta_path, contig, mutation_pos, alt_base,
                          gene_start, gene_end, strand="+"):
    """
    Translate a DNA mutation into a protein consequence.

    Handles both forward (+) and reverse (-) strand genes correctly
    by computing the reading frame offset and reverse-complementing
    codons for minus-strand genes.

    Parameters
    ----------
    fasta_path : str
        Path to the reference FASTA.
    contig : str
        Contig name.
    mutation_pos : int
        1-based genomic position of the mutation.
    alt_base : str
        The alternate allele ('-' for deletions).
    gene_start : int
        1-based start position of the gene (CDS start).
    gene_end : int
        1-based end position of the gene (CDS end).
    strand : str
        '+' for forward, '-' for reverse strand.

    Returns
    -------
    aa_change : str
        HGVS-like protein notation (e.g., 'p.D91Y').
    mutation_type : str
        'Missense', 'Synonymous', 'Nonsense', or 'Indel'.
    impact : str
        'HIGH', 'MODERATE', or 'LOW'.
    """
    # Handle indels immediately — no codon math needed
    if alt_base == "-":
        return "Frameshift", "Indel", "HIGH"
    if len(alt_base) > 1 or alt_base.startswith("+"):
        return "Frameshift", "Indel", "HIGH"

    fasta = pysam.FastaFile(fasta_path)

    try:
        if strand == "+":
            # Forward strand: count from gene_start
            dist_from_start = mutation_pos - gene_start
            codon_number = dist_from_start // 3
            pos_in_codon = dist_from_start % 3

            # 0-based coordinates for FASTA fetch
            codon_start_0based = (gene_start - 1) + (codon_number * 3)
        else:
            # Reverse strand: count from gene_end
            dist_from_start = gene_end - mutation_pos
            codon_number = dist_from_start // 3
            pos_in_codon = dist_from_start % 3

            # For minus strand, pos_in_codon=0 is the RIGHTMOST genomic base
            # (first base of the coding codon), pos_in_codon=2 is the LEFTMOST.
            # The codon spans from (mutation_pos + pos_in_codon - 2) to
            # (mutation_pos + pos_in_codon) on the forward strand.
            codon_start_0based = mutation_pos - 3 + pos_in_codon

        # Fetch the 3-base codon from the genome
        ref_codon = fasta.fetch(contig, codon_start_0based, codon_start_0based + 3).upper()

        if len(ref_codon) != 3:
            return "Unknown", "Complex", "UNKNOWN"

        # Build the mutant codon
        alt_codon_list = list(ref_codon)
        if strand == "-":
            alt_codon_list[2 - pos_in_codon] = alt_base.upper()
        else:
            alt_codon_list[pos_in_codon] = alt_base.upper()
        alt_codon = "".join(alt_codon_list)

        # For reverse strand, reverse-complement before translating
        if strand == "-":
            ref_codon = str(Seq(ref_codon).reverse_complement())
            alt_codon = str(Seq(alt_codon).reverse_complement())

        # Translate
        ref_aa = _translate_codon(ref_codon)
        alt_aa = _translate_codon(alt_codon)

        aa_pos = codon_number + 1  # 1-based amino acid position
        aa_change = f"p.{ref_aa}{aa_pos}{alt_aa}"

        if alt_aa == "*":
            return aa_change, "Nonsense", "HIGH"
        elif ref_aa != alt_aa:
            return aa_change, "Missense", "HIGH"
        else:
            return aa_change, "Synonymous", "LOW"

    except Exception as e:
        logger.warning(f"Translation error at {contig}:{mutation_pos}: {e}")
        return "Unknown", "Complex", "UNKNOWN"
    finally:
        fasta.close()


def load_panel(panel_path):
    """
    Load an AMR panel configuration from JSON.

    Expected JSON structure:
    {
        "organism": "Mycobacterium leprae",
        "reference": "NC_002677.1",
        "targets": {
            "gyrA_DRDR": {
                "start": 7550, "end": 7650,
                "gene_start": 7318, "gene_end": 11067,
                "strand": "+",
                "drug": "Fluoroquinolones",
                "known_mutations": {
                    "7589": {"ref": "G", "alt": "T", "aa": "p.D91Y",
                             "drug": "Ofloxacin", "literature": "WHO 2023"}
                }
            }
        }
    }

    Returns
    -------
    panel : dict
        The parsed panel dictionary.
    """
    with open(panel_path, "r") as f:
        panel = json.load(f)

    n_targets = len(panel.get("targets", {}))
    n_known = sum(
        len(t.get("known_mutations", {}))
        for t in panel.get("targets", {}).values()
    )
    logger.info(
        f"Loaded panel: {panel.get('organism', 'Unknown')} | "
        f"{n_targets} gene targets | {n_known} known mutations"
    )
    return panel


def annotate_variant(variant_call, panel_targets, fasta_path):
    """
    Annotate a VariantCall with protein consequence and tier classification.

    Tier Logic:
        - If the position+alt matches a known_mutations entry → Tier 1 (KNOWN)
        - If it's in a target region but not catalogued → Tier 2 (NOVEL)
        - If it's a synonymous change → marked as LOW impact regardless

    Parameters
    ----------
    variant_call : VariantCall
        Raw call from the scanner.
    panel_targets : dict
        The "targets" section of the panel JSON.
    fasta_path : str
        Path to reference FASTA for codon translation.

    Returns
    -------
    annotated : AnnotatedVariant
    """
    gene_name = variant_call.gene
    gene_info = panel_targets.get(gene_name, {})

    # Translate the DNA change to protein
    aa_change, mut_type, impact = get_amino_acid_change(
        fasta_path=fasta_path,
        contig=variant_call.contig,
        mutation_pos=variant_call.position,
        alt_base=variant_call.alt_base,
        gene_start=gene_info.get("gene_start", 0),
        gene_end=gene_info.get("gene_end", 0),
        strand=gene_info.get("strand", "+"),
    )

    # Tier classification: check against known mutations
    # Keys in the panel use "position_alt" format (e.g., "7589_T") to support
    # multiple known mutations at the same genomic position.
    known_muts = gene_info.get("known_mutations", {})
    pos_alt_key = f"{variant_call.position}_{variant_call.alt_base.upper()}"
    tier = "NOVEL"
    known_drug = ""
    known_lit = ""

    if pos_alt_key in known_muts:
        known_entry = known_muts[pos_alt_key]
        tier = "KNOWN"
        known_drug = known_entry.get("drug", "")
        known_lit = known_entry.get("literature", "")

    # Synonymous mutations are low-interest regardless of tier
    if mut_type == "Synonymous":
        impact = "LOW"

    return AnnotatedVariant(
        patient_id=variant_call.patient_id,
        gene=variant_call.gene,
        contig=variant_call.contig,
        position=variant_call.position,
        ref_base=variant_call.ref_base,
        alt_base=variant_call.alt_base,
        ref_depth=variant_call.ref_depth,
        alt_depth=variant_call.alt_depth,
        total_depth=variant_call.total_depth,
        allele_frequency=variant_call.allele_frequency,
        predicted_class=variant_call.predicted_class,
        class_name=variant_call.class_name,
        confidence=variant_call.confidence,
        aa_change=aa_change,
        mutation_type=mut_type,
        impact=impact,
        tier=tier,
        known_drug=known_drug,
        known_literature=known_lit,
    )
