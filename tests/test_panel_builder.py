"""
tests/test_panel_builder.py
===========================
Unit tests for mindv panel_builder module.

Run with:
    pytest tests/test_panel_builder.py -v

All tests verified against the actual panel_builder.py API:
  - CODON_TABLE (alias for GENETIC_CODE)
  - VerificationResult(status, extracted_codon, extracted_aa, expected_ref_aa, message="")
  - VerificationResult.is_ok property
  - extract_ref_codon(fasta_or_path, gene, codon_number)
  - codon_genomic_positions(N, gene)
  - enumerate_snps(ref_codon, alt_aa)
  - PanelBuilder._process_mutation(mut, gene) -> List[PanelEntry]
  - PanelBuilder._deduplicate(entries) -> List[PanelEntry]
  - PanelEntry.panel_key -> "POSITION_ALT"
  - WhoCatalogueParser(tsv_path, min_grade)
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List

import pytest

# ---------------------------------------------------------------------------
# Import panel_builder from source tree without installing the package
# ---------------------------------------------------------------------------

import importlib.util as _ilu

_MODULE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "src", "mindeepvariant", "panel_builder.py"
))

_spec = _ilu.spec_from_file_location("panel_builder", _MODULE_PATH)
pb = _ilu.module_from_spec(_spec)
import sys as _sys
_sys.modules["panel_builder"] = pb   # required for @dataclass to resolve the module
_spec.loader.exec_module(pb)

GeneAnnotation = pb.GeneAnnotation
ResistanceMutation = pb.ResistanceMutation
PanelEntry = pb.PanelEntry
VerificationResult = pb.VerificationResult
PanelBuilder = pb.PanelBuilder
CODON_TABLE = pb.CODON_TABLE          # public alias for GENETIC_CODE
enumerate_snps = pb.enumerate_snps
reverse_complement = pb.reverse_complement
codon_genomic_positions = pb.codon_genomic_positions
extract_ref_codon = pb.extract_ref_codon
WhoCatalogueParser = pb.WhoCatalogueParser


# ---------------------------------------------------------------------------
# FASTA helpers
# ---------------------------------------------------------------------------

def _write_fasta(seq: str, contig: str = "TEST") -> str:
    """Write a sequence to a temp FASTA and return the path."""
    path = tempfile.mktemp(suffix=".fasta")
    with open(path, "w") as fh:
        fh.write(f">{contig}\n")
        for i in range(0, len(seq), 60):
            fh.write(seq[i:i+60] + "\n")
    return path


def _index(path: str) -> None:
    """samtools faidx — skip silently if samtools is absent."""
    import subprocess
    subprocess.run(["samtools", "faidx", path], capture_output=True)


# ---------------------------------------------------------------------------
# Codon table
# ---------------------------------------------------------------------------

class TestCodonTable:
    def test_all_64_codons(self):
        bases = "ACGT"
        expected = {b1+b2+b3 for b1 in bases for b2 in bases for b3 in bases}
        assert set(CODON_TABLE.keys()) == expected

    def test_standard_codons(self):
        assert CODON_TABLE["ATG"] == "M"
        assert CODON_TABLE["TAA"] == "*"
        assert CODON_TABLE["TGG"] == "W"
        assert CODON_TABLE["TCG"] == "S"   # rpoB S450/S456 codon
        assert CODON_TABLE["GAC"] == "D"   # Asp — rpoB D435
        assert CODON_TABLE["CAC"] == "H"   # His — rpoB H445
        assert CODON_TABLE["GGC"] == "G"   # Gly — gyrA G89 (M. leprae)

    def test_stop_codons(self):
        for stop in ("TAA", "TAG", "TGA"):
            assert CODON_TABLE[stop] == "*", f"{stop} should map to stop (*)"


# ---------------------------------------------------------------------------
# reverse_complement
# ---------------------------------------------------------------------------

class TestReverseComplement:
    def test_simple(self):
        assert reverse_complement("ATCG") == "CGAT"

    def test_palindrome(self):
        assert reverse_complement("AATT") == "AATT"

    def test_rpoB_TCG(self):
        # TCG (Ser, coding strand) stored as CGA on + strand for minus-strand genes
        assert reverse_complement("TCG") == "CGA"

    def test_round_trip(self):
        seq = "GATTACAATCG"
        assert reverse_complement(reverse_complement(seq)) == seq


# ---------------------------------------------------------------------------
# VerificationResult.is_ok
# ---------------------------------------------------------------------------

class TestVerificationResult:
    def test_verified_is_ok(self):
        v = VerificationResult(
            status="VERIFIED",
            extracted_codon="GAC",
            extracted_aa="D",
            expected_ref_aa="D",
        )
        assert v.is_ok is True

    def test_failed_not_ok(self):
        v = VerificationResult(
            status="FAILED",
            extracted_codon="ATG",
            extracted_aa="M",
            expected_ref_aa="D",
            message="Expected D got M",
        )
        assert v.is_ok is False

    def test_warning_is_ok(self):
        v = VerificationResult(
            status="WARNING",
            extracted_codon="GAT",
            extracted_aa="D",
            expected_ref_aa="D",
            message="Synonymous codon",
        )
        assert v.is_ok is True

    def test_promoter_is_ok(self):
        v = VerificationResult(
            status="PROMOTER/rRNA",
            extracted_codon="N/A",
            extracted_aa="N/A",
            expected_ref_aa="C",
        )
        assert v.is_ok is True


# ---------------------------------------------------------------------------
# codon_genomic_positions
# ---------------------------------------------------------------------------

class TestCodonGenomicPositions:

    def _plus(self) -> GeneAnnotation:
        return GeneAnnotation("gyrA", "NC_000962.3", 7302, 9818, "+")

    def _minus(self) -> GeneAnnotation:
        return GeneAnnotation("rpoB", "NC_000962.3", 759807, 763325, "-")

    def test_plus_codon1(self):
        p1, p2, p3 = codon_genomic_positions(1, self._plus())
        assert (p1, p2, p3) == (7302, 7303, 7304)

    def test_plus_codon_n(self):
        N = 94
        p1, p2, p3 = codon_genomic_positions(N, self._plus())
        expected = 7302 + (N - 1) * 3
        assert p1 == expected and p2 == expected + 1 and p3 == expected + 2

    def test_minus_codon1(self):
        p1, p2, p3 = codon_genomic_positions(1, self._minus())
        assert p1 == 763325
        assert p2 == 763324
        assert p3 == 763323

    def test_minus_codon_n(self):
        N = 450
        p1, p2, p3 = codon_genomic_positions(N, self._minus())
        expected = 763325 - (N - 1) * 3
        assert p1 == expected and p2 == expected - 1 and p3 == expected - 2

    def test_minus_positions_descend(self):
        p1, p2, p3 = codon_genomic_positions(5, self._minus())
        assert p1 > p2 > p3

    def test_plus_positions_ascend(self):
        p1, p2, p3 = codon_genomic_positions(5, self._plus())
        assert p1 < p2 < p3

    def test_rpoB_S450_position(self):
        """H37Rv rpoB S450 codon must land at known positions 761976-761978."""
        # gene_end = 763325, N = 450
        # p1 = 763325 - 449*3 = 763325 - 1347 = 761978... wait:
        # Actually: p1 = gene_end - (N-1)*3 = 763325 - 449*3 = 763325 - 1347 = 761978
        # p2 = 761977, p3 = 761976
        gene = GeneAnnotation("rpoB", "NC_000962.3", 759807, 763325, "-")
        p1, p2, p3 = codon_genomic_positions(450, gene)
        assert p1 == 761978
        assert p2 == 761977
        assert p3 == 761976


# ---------------------------------------------------------------------------
# enumerate_snps
# ---------------------------------------------------------------------------

class TestEnumerateSNPs:

    def test_ser_to_leu_TCG(self):
        snps = enumerate_snps("TCG", "L")
        assert len(snps) >= 1
        # TTG = Leu: pos 1, C→T
        assert any(pos == 1 and r == "C" and a == "T" for pos, r, a in snps)

    def test_ser_to_trp_TCG(self):
        snps = enumerate_snps("TCG", "W")
        assert any(pos == 1 and r == "C" and a == "G" for pos, r, a in snps)

    def test_his_to_tyr_CAC(self):
        snps = enumerate_snps("CAC", "Y")
        assert any(pos == 0 and r == "C" and a == "T" for pos, r, a in snps)

    def test_his_to_asp_CAC(self):
        snps = enumerate_snps("CAC", "D")
        assert any(pos == 0 and r == "C" and a == "G" for pos, r, a in snps)

    def test_stop_codon(self):
        snps = enumerate_snps("CAG", "*")
        assert any(pos == 0 and r == "C" and a == "T" for pos, r, a in snps)

    def test_unreachable_returns_empty(self):
        # ATG (Met) → W (Trp=TGG) requires 2 changes
        assert enumerate_snps("ATG", "W") == []

    def test_no_ref_equals_alt(self):
        for pos, r, a in enumerate_snps("GAC", "N"):
            assert r != a

    def test_all_alts_produce_target(self):
        for pos, r, a in enumerate_snps("GCA", "V"):
            mut = list("GCA")
            mut[pos] = a
            assert CODON_TABLE["".join(mut)] == "V"


# ---------------------------------------------------------------------------
# extract_ref_codon (path string interface)
# ---------------------------------------------------------------------------

class TestExtractRefCodon:

    @pytest.fixture
    def plus_fasta(self):
        """Gene at 101-310: ATG GCA GCG GCC GAC ..."""
        prefix = "N" * 100
        codons = ["ATG", "GCA", "GCG", "GCC", "GAC"] + ["GCT"] * 65
        seq = prefix + "".join(codons) + "N" * 90
        path = _write_fasta(seq, "SYNTH")
        _index(path)
        yield path
        for f in [path, path + ".fai"]:
            if os.path.exists(f):
                os.unlink(f)

    @pytest.fixture
    def minus_fasta(self):
        """+ strand at 201-203 = CGA → rev-comp = TCG (Ser)."""
        seq = "N" * 200 + "CGA" + "N" * 200
        path = _write_fasta(seq, "SYNTH2")
        _index(path)
        yield path
        for f in [path, path + ".fai"]:
            if os.path.exists(f):
                os.unlink(f)

    def test_plus_codon1(self, plus_fasta):
        gene = GeneAnnotation("g", "SYNTH", 101, 310, "+")
        assert extract_ref_codon(plus_fasta, gene, 1) == "ATG"

    def test_plus_codon2(self, plus_fasta):
        gene = GeneAnnotation("g", "SYNTH", 101, 310, "+")
        assert extract_ref_codon(plus_fasta, gene, 2) == "GCA"

    def test_plus_codon5(self, plus_fasta):
        gene = GeneAnnotation("g", "SYNTH", 101, 310, "+")
        assert extract_ref_codon(plus_fasta, gene, 5) == "GAC"

    def test_minus_codon1(self, minus_fasta):
        gene = GeneAnnotation("g", "SYNTH2", 199, 203, "-")
        codon = extract_ref_codon(minus_fasta, gene, 1)
        assert codon == "TCG", f"Expected TCG (Ser), got {codon!r}"

    def test_accepts_string_path(self, plus_fasta):
        gene = GeneAnnotation("g", "SYNTH", 101, 310, "+")
        codon = extract_ref_codon(plus_fasta, gene, 1)
        assert len(codon) == 3


# ---------------------------------------------------------------------------
# PanelEntry.panel_key
# ---------------------------------------------------------------------------

class TestPanelEntry:

    def _entry(self, pos=7589, alt="T"):
        return PanelEntry(
            gene="gyrA",
            genomic_pos=pos,
            ref_nuc="C",
            alt_nuc=alt,
            aa_change="p.A91V",
            codon_change="GCA>GTA",
            drug="fluoroquinolones",
            verification=VerificationResult("VERIFIED", "GCA", "A", "A"),
        )

    def test_key_format(self):
        assert self._entry(7589, "T").panel_key == "7589_T"

    def test_different_alts_different_keys(self):
        assert self._entry(7600, "A").panel_key != self._entry(7600, "T").panel_key

    def test_same_pos_different_alts(self):
        assert self._entry(7600, "A").panel_key == "7600_A"
        assert self._entry(7600, "T").panel_key == "7600_T"


# ---------------------------------------------------------------------------
# PanelBuilder._deduplicate
# ---------------------------------------------------------------------------

class TestDeduplicate:

    def _make(self, pos: int, alt: str) -> PanelEntry:
        return PanelEntry(
            gene="geneX", genomic_pos=pos, ref_nuc="G", alt_nuc=alt,
            aa_change="p.X1Y", codon_change="GXX>GYY", drug="drug_A",
            verification=VerificationResult("VERIFIED", "GXX", "X", "X"),
        )

    def test_distinct_alts_all_preserved(self):
        entries = [self._make(100, "A"), self._make(100, "T"), self._make(100, "C")]
        result = PanelBuilder._deduplicate(entries)
        assert {e.panel_key for e in result} == {"100_A", "100_T", "100_C"}

    def test_exact_duplicate_deduplicated(self):
        entries = [self._make(200, "A"), self._make(200, "A")]
        result = PanelBuilder._deduplicate(entries)
        assert len(result) == 1 and result[0].panel_key == "200_A"

    def test_50_distinct_positions_preserved(self):
        entries = [self._make(i, "A") for i in range(50)]
        assert len(PanelBuilder._deduplicate(entries)) == 50


# ---------------------------------------------------------------------------
# PanelBuilder._process_mutation
# ---------------------------------------------------------------------------

class TestProcessMutation:

    @pytest.fixture
    def simple_fasta(self):
        """Plus-strand gene: codon 2 = GCA (Ala) at positions 104-106."""
        prefix = "N" * 100
        gene_seq = "ATG" + "GCA" + "GCG" * 68
        seq = prefix + gene_seq + "N" * 100
        path = _write_fasta(seq, "SYNTH")
        _index(path)
        yield path
        for f in [path, path + ".fai"]:
            if os.path.exists(f):
                os.unlink(f)

    def _builder(self, fasta_path, gene):
        return PanelBuilder(
            mutations=[],
            annotations={gene.gene_name: gene},
            fasta_path=fasta_path,
        )

    def test_correct_mutation_produces_entries(self, simple_fasta):
        gene = GeneAnnotation("testGene", "SYNTH", 101, 310, "+")
        # codon 2 = GCA (Ala); Ala→Val via GCA→GTA: pos 1 in codon, + strand pos 105
        mut = ResistanceMutation("testGene", "A", 2, "V", "test_drug",
                                 source="synthetic", hgvs_p="p.A2V")
        entries = self._builder(simple_fasta, gene)._process_mutation(mut, gene)
        assert len(entries) >= 1
        keys = [e.panel_key for e in entries]
        assert "105_T" in keys, f"Expected '105_T' among {keys}"

    def test_verification_passes_for_correct_ref(self, simple_fasta):
        gene = GeneAnnotation("testGene", "SYNTH", 101, 310, "+")
        mut = ResistanceMutation("testGene", "A", 2, "V", "drug",
                                 source="synthetic", hgvs_p="p.A2V")
        entries = self._builder(simple_fasta, gene)._process_mutation(mut, gene)
        for e in entries:
            assert e.verification.is_ok, (
                f"Expected VERIFIED, got {e.verification.status}: {e.verification.message}"
            )

    def test_wrong_ref_aa_fails(self, simple_fasta):
        gene = GeneAnnotation("testGene", "SYNTH", 101, 310, "+")
        # codon 2 = Ala; claim ref="K" (Lys) — WRONG
        mut = ResistanceMutation("testGene", "K", 2, "E", "drug",
                                 source="synthetic", hgvs_p="p.K2E")
        entries = self._builder(simple_fasta, gene)._process_mutation(mut, gene)
        for e in entries:
            assert not e.verification.is_ok

    def test_unreachable_aa_returns_empty(self, simple_fasta):
        gene = GeneAnnotation("testGene", "SYNTH", 101, 310, "+")
        # codon 1 = ATG (Met); Met→Trp requires 2 SNPs
        mut = ResistanceMutation("testGene", "M", 1, "W", "drug",
                                 source="synthetic", hgvs_p="p.M1W")
        entries = self._builder(simple_fasta, gene)._process_mutation(mut, gene)
        assert entries == []


# ---------------------------------------------------------------------------
# WhoCatalogueParser
# ---------------------------------------------------------------------------

class TestWhoCatalogueParser:

    def _tsv(self, rows: list) -> str:
        path = tempfile.mktemp(suffix=".tsv")
        with open(path, "w") as fh:
            fh.write("drug\tgene\tmutation\tconfidence_grade\n")
            fh.writelines(rows)
        return path

    def test_parses_hgvs_long_form(self):
        path = self._tsv(["rifampicin\trpoB\tp.Ser450Leu\t1\n"])
        muts = WhoCatalogueParser(path, 2).parse()
        assert any("p.Ser450Leu" in m.hgvs_p for m in muts)
        os.unlink(path)

    def test_parses_hgvs_short_form(self):
        path = self._tsv(["rifampicin\trpoB\tp.H445Y\t1\n"])
        muts = WhoCatalogueParser(path, 2).parse()
        assert len(muts) == 1 and muts[0].ref_aa == "H" and muts[0].alt_aa == "Y"
        os.unlink(path)

    def test_grade_filter(self):
        path = self._tsv([
            "rifampicin\trpoB\tp.S450L\t1\n",   # grade 1
            "rifampicin\trpoB\tp.H445Y\t2\n",   # grade 2
            "rifampicin\trpoB\tp.D435Y\t3\n",   # grade 3 — excluded by min_grade=2
        ])
        muts_strict = WhoCatalogueParser(path, 1).parse()
        muts_lenient = WhoCatalogueParser(path, 2).parse()
        assert len(muts_strict) == 1
        assert len(muts_lenient) == 2
        os.unlink(path)

    def test_synonymous_excluded(self):
        path = self._tsv([
            "rifampicin\trpoB\tp.S450=\t1\n",   # synonymous
            "rifampicin\trpoB\tp.S450L\t1\n",
        ])
        muts = WhoCatalogueParser(path, 1).parse()
        assert len(muts) == 1
        os.unlink(path)

    def test_promoter_parsed(self):
        path = self._tsv(["isoniazid\tinhA\tc.-15C>T\t1\n"])
        muts = WhoCatalogueParser(path, 1).parse()
        assert len(muts) == 1
        m = muts[0]
        assert "promoter" in m.gene
        assert m.hgvs_p == "c.-15C>T"
        assert m.ref_aa == "C" and m.alt_aa == "T" and m.codon_number == -15
        os.unlink(path)

    def test_rrna_parsed(self):
        path = self._tsv(["amikacin\trrs\tr.1401a>g\t1\n"])
        muts = WhoCatalogueParser(path, 1).parse()
        assert len(muts) == 1
        m = muts[0]
        assert "rRNA" in m.gene
        assert m.ref_aa == "A" and m.alt_aa == "G" and m.codon_number == 1401
        os.unlink(path)

    def test_drug_field_preserved(self):
        path = self._tsv(["isoniazid\tkatG\tp.S315T\t1\n"])
        muts = WhoCatalogueParser(path, 1).parse()
        assert muts[0].drug == "isoniazid"
        os.unlink(path)


# ---------------------------------------------------------------------------
# TB panel JSON structural invariants
# ---------------------------------------------------------------------------

class TestTBPanel:

    @pytest.fixture(scope="class")
    def tb_panel(self):
        path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "configs", "tb_h37rv.json"
        ))
        if not os.path.exists(path):
            pytest.skip("tb_h37rv.json not found")
        with open(path) as fh:
            return json.load(fh)

    def test_required_keys(self, tb_panel):
        for k in ("organism", "reference", "panel_version", "targets"):
            assert k in tb_panel

    def test_organism_is_tuberculosis(self, tb_panel):
        assert "tuberculosis" in tb_panel["organism"].lower()

    def test_reference_is_h37rv(self, tb_panel):
        assert "NC_000962" in tb_panel["reference"]

    def test_all_targets_have_required_fields(self, tb_panel):
        for tname, tdata in tb_panel["targets"].items():
            for f in ("start", "end", "gene_start", "gene_end", "strand", "known_mutations"):
                assert f in tdata, f"{tname} missing '{f}'"

    def test_no_duplicate_mutation_keys(self, tb_panel):
        for tname, tdata in tb_panel["targets"].items():
            keys = list(tdata["known_mutations"].keys())
            dupes = {k for k in keys if keys.count(k) > 1}
            assert not dupes, f"Duplicate keys in {tname}: {dupes}"

    def test_mutation_key_format(self, tb_panel):
        import re
        pat = re.compile(r"^\d+_[ACGT]$")
        for tname, tdata in tb_panel["targets"].items():
            for key in tdata["known_mutations"]:
                assert pat.match(key), f"Bad key in {tname}: '{key}'"

    def test_each_mutation_has_ref_alt_aa(self, tb_panel):
        for tname, tdata in tb_panel["targets"].items():
            for key, mdata in tdata["known_mutations"].items():
                for field in ("ref", "alt", "aa_change"):
                    assert field in mdata, f"{tname}/{key} missing '{field}'"

    def test_valid_strands(self, tb_panel):
        for tname, tdata in tb_panel["targets"].items():
            assert tdata["strand"] in ("+", "-"), f"{tname}: bad strand"

    def test_start_less_than_end(self, tb_panel):
        for tname, tdata in tb_panel["targets"].items():
            assert tdata["start"] < tdata["end"]

    def test_rpob_target_present(self, tb_panel):
        assert any("rpoB" in t for t in tb_panel["targets"])

    def test_rpob_S450L(self, tb_panel):
        for tname, tdata in tb_panel["targets"].items():
            if "rpoB" in tname:
                changes = {v["aa_change"] for v in tdata["known_mutations"].values()}
                assert "p.S450L" in changes, f"p.S450L missing from {tname}: {changes}"
                return
        pytest.fail("No rpoB target found")

    def test_katG_S315T(self, tb_panel):
        for tname, tdata in tb_panel["targets"].items():
            if "katG" in tname:
                changes = {v["aa_change"] for v in tdata["known_mutations"].values()}
                assert "p.S315T" in changes, f"p.S315T missing from {tname}"
                return
        pytest.fail("No katG target found")


# ---------------------------------------------------------------------------
# Calibration smoke test
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_minus_strand_codon_at_calibrated_end(self):
        """
        Synthetic: + strand at 228-230 (0-based 227-229) = CGA → rev-comp = TCG (Ser).
        gene_end=230, codon 10 on − strand reads positions 230, 229, 228.
        """
        seq = list("N" * 300)
        seq[227] = "C"
        seq[228] = "G"
        seq[229] = "A"
        path = _write_fasta("".join(seq), "CALIB")
        _index(path)
        gene = GeneAnnotation("g", "CALIB", 100, 230, "-")
        codon = extract_ref_codon(path, gene, 10)
        assert codon == "TCG", f"Expected TCG (Ser), got {codon!r}"
        for f in [path, path + ".fai"]:
            if os.path.exists(f):
                os.unlink(f)


# ---------------------------------------------------------------------------
# CLI argparse smoke test
# ---------------------------------------------------------------------------

class TestCLIParser:

    @pytest.fixture(scope="class")
    def clip(self):
        path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "src", "mindeepvariant", "cli_panelbuild.py"
        ))
        spec = _ilu.spec_from_file_location("clip", path)
        m = _ilu.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_who_catalogue_accepted(self, clip, tmp_path):
        import argparse
        p = argparse.ArgumentParser()
        clip.add_panel_build_subparser(p.add_subparsers())
        args = p.parse_args([
            "panel-build", "--source", "who_catalogue",
            "--who-catalogue-tsv", "dummy.tsv",
            "--organism", "Mycobacterium tuberculosis",
            "--ref", "ref.fasta", "--gff", "ref.gff3",
            "--output", str(tmp_path / "panel.json"),
        ])
        assert args.source == "who_catalogue"

    def test_card_accepted(self, clip, tmp_path):
        import argparse
        p = argparse.ArgumentParser()
        clip.add_panel_build_subparser(p.add_subparsers())
        args = p.parse_args([
            "panel-build", "--source", "card",
            "--card-json", "card.json",
            "--organism", "Mycobacterium tuberculosis",
            "--ref", "ref.fasta", "--gff", "ref.gff3",
            "--output", str(tmp_path / "panel.json"),
        ])
        assert args.source == "card"

    def test_source_required(self, clip):
        import argparse
        p = argparse.ArgumentParser()
        clip.add_panel_build_subparser(p.add_subparsers())
        with pytest.raises(SystemExit):
            p.parse_args([
                "panel-build",
                "--organism", "Mycobacterium tuberculosis",
                "--ref", "ref.fasta", "--gff", "ref.gff3", "--output", "out.json",
            ])


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
