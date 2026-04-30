"""
scanner.py — The Pileup Scanner

This module extracts real pileup data from BAM files and converts
them into tensors for CNN inference. It also implements the heuristic
"fast-forward" scanner that skips wild-type positions and only triggers
the CNN when a minimum number of alternate reads are observed.

Key fixes over the original notebook implementation:
    1. The original code iterated pileup columns with a manual counter,
       which drifted out of sync when pysam skipped zero-coverage positions.
       This version uses the actual genomic coordinate from each pileup
       column to place data in the correct tensor column.

    2. Quality filtering at the pileup level: low base-quality (BQ) and
       low mapping-quality (MQ) reads are excluded before any allele
       counting. This is applied via pysam's native min_base_quality and
       min_mapping_quality kwargs, which means filtered reads are
       invisible to both the heuristic counter AND the tensor extractor —
       they don't pollute downstream stages.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import pysam

from .model import BASE_TO_PIXEL, CLASS_NAMES

logger = logging.getLogger(__name__)


@dataclass
class VariantCall:
    """A single variant call produced by the scanner."""

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
    probabilities: list  # Per-class softmax scores


@contextmanager
def open_genomic_files(bam_path, fasta_path):
    """
    Context manager that opens and closes BAM/FASTA handles properly.

    This prevents file descriptor leaks during batch processing — the
    original code opened new handles on every tensor extraction call
    without closing them, which crashes after ~1000 open files.
    """
    bam = pysam.AlignmentFile(bam_path, "rb")
    fasta = pysam.FastaFile(fasta_path)
    try:
        yield bam, fasta
    finally:
        bam.close()
        fasta.close()


def validate_inputs(bam_path, fasta_path, contig):
    """
    Check that input files are valid before scanning.

    Raises
    ------
    FileNotFoundError
        If BAM or FASTA file doesn't exist.
    ValueError
        If BAM isn't indexed or contig isn't in the reference.
    """
    import os

    if not os.path.isfile(bam_path):
        raise FileNotFoundError(f"BAM file not found: {bam_path}")
    if not os.path.isfile(fasta_path):
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    # Check BAM index — pysam handles .bai, .csi, and .bam.bai automatically
    try:
        test_bam = pysam.AlignmentFile(bam_path, "rb")
        if not test_bam.has_index():
            test_bam.close()
            raise ValueError(
                f"BAM index not found for {bam_path}. "
                f"Run: samtools index {bam_path}"
            )
        test_bam.close()
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Cannot open BAM {bam_path}: {e}")

    # Check contig exists in reference
    with pysam.FastaFile(fasta_path) as fasta:
        if contig not in fasta.references:
            raise ValueError(
                f"Contig '{contig}' not found in {fasta_path}. "
                f"Available: {list(fasta.references[:5])}..."
            )


def extract_pileup_tensor(
    bam, fasta, contig, center_pos,
    window=21, depth=30,
    min_bq=20, min_mq=20,
):
    """
    Extract a pileup tensor from real BAM data.

    This is the core "image generation" step — it converts a stack of
    aligned reads centered on a genomic position into a grayscale matrix
    that the CNN can classify.

    Quality filtering: reads with mapping quality below `min_mq` and
    individual bases with base quality below `min_bq` are excluded by
    pysam *before* the pileup column reaches us. This means the tensor
    only ever contains high-confidence evidence — the same evidence the
    heuristic counter sees in scan_region().

    Parameters
    ----------
    bam : pysam.AlignmentFile
    fasta : pysam.FastaFile
    contig : str
    center_pos : int
        1-based genomic position to center the window on.
    window : int
        Width of the tensor in base pairs.
    depth : int
        Maximum number of read rows.
    min_bq : int
        Minimum base quality (Phred). Default 20 = 1% per-base error.
    min_mq : int
        Minimum mapping quality. Default 20 excludes ambiguously-mapped reads.

    Returns
    -------
    tensor : torch.Tensor
        Shape (depth+1, window), float32.
    """
    half = window // 2
    start_0based = (center_pos - 1) - half  # Convert to 0-based
    end_0based = start_0based + window

    # Clamp to valid coordinates
    start_0based = max(0, start_0based)

    # Row 0: Reference sequence
    ref_string = fasta.fetch(contig, start_0based, end_0based).upper()
    grid = [[0.0] * window for _ in range(depth + 1)]

    for i, base in enumerate(ref_string):
        if i < window:
            grid[0][i] = BASE_TO_PIXEL.get(base, 0.0)

    # Read rows: use actual genomic coordinate to find column index.
    # Quality filtering applied at pileup level via pysam kwargs.
    pileup_iter = bam.pileup(
        contig, start_0based, end_0based,
        truncate=True,
        min_base_quality=min_bq,
        min_mapping_quality=min_mq,
    )

    for pileup_col in pileup_iter:
        col_idx = pileup_col.reference_pos - start_0based

        if col_idx < 0 or col_idx >= window:
            continue

        row_idx = 1
        for pileup_read in pileup_col.pileups:
            if row_idx > depth:
                break
            if pileup_read.is_del:
                grid[row_idx][col_idx] = BASE_TO_PIXEL["-"]
            elif not pileup_read.is_refskip:
                qpos = pileup_read.query_position
                if qpos is not None:
                    base = pileup_read.alignment.query_sequence[qpos].upper()
                    grid[row_idx][col_idx] = BASE_TO_PIXEL.get(base, 0.0)
            row_idx += 1

    return torch.tensor(grid, dtype=torch.float32)


def scan_region(
    model,
    bam,
    fasta,
    contig,
    start_pos,
    end_pos,
    patient_id="unknown",
    gene_name="unknown",
    min_alt_reads=3,
    min_total_depth=15,
    window=21,
    depth=30,
    min_bq=20,
    min_mq=20,
):
    """
    Scan a genomic region for variants using the heuristic + CNN approach.

    The heuristic filter counts alternate alleles at each position and
    only triggers the (expensive) CNN inference when a minimum number
    of non-reference reads are observed. This dramatically reduces
    compute time on wild-type regions.

    Quality filtering (min_bq, min_mq) is applied at the pysam pileup
    level, so low-quality reads/bases are invisible to both the heuristic
    counter and the downstream tensor extractor. This is the right place
    for it — filtering after counting would create inconsistencies where
    the CNN sees different reads than the heuristic did.

    Parameters
    ----------
    model : MinDeepVariantCNN
        Trained model in eval mode.
    bam : pysam.AlignmentFile
    fasta : pysam.FastaFile
    contig : str
    start_pos, end_pos : int
        1-based inclusive scan region.
    patient_id, gene_name : str
        Annotation strings.
    min_alt_reads : int
        Minimum alternate read count to trigger CNN.
    min_total_depth : int
        Minimum total depth to trigger CNN.
    window, depth : int
        Tensor dimensions.
    min_bq : int
        Minimum base quality (Phred). Default 20.
    min_mq : int
        Minimum mapping quality. Default 20.

    Yields
    ------
    VariantCall
        One per position that passes the heuristic filter and gets
        classified by the CNN. Includes class-0 (wild-type) calls for
        transparency — downstream filtering decides what to report.
    """
    pileup_iter = bam.pileup(
        contig, start_pos - 1, end_pos,
        truncate=True,
        min_base_quality=min_bq,
        min_mapping_quality=min_mq,
    )

    for pileup_col in pileup_iter:
        pos = pileup_col.reference_pos + 1  # Back to 1-based
        ref_base = fasta.fetch(
            contig, pileup_col.reference_pos, pileup_col.reference_pos + 1
        ).upper()

        # Count alleles
        ref_count = 0
        alt_counts = {}

        for pileup_read in pileup_col.pileups:
            if pileup_read.is_del:
                alt_counts["-"] = alt_counts.get("-", 0) + 1
            elif not pileup_read.is_refskip:
                qpos = pileup_read.query_position
                if qpos is not None:
                    read_base = pileup_read.alignment.query_sequence[qpos].upper()
                    if read_base == ref_base:
                        ref_count += 1
                    elif read_base != "N":
                        alt_counts[read_base] = alt_counts.get(read_base, 0) + 1

        total_alt = sum(alt_counts.values())
        total_dp = ref_count + total_alt

        # Heuristic filter: skip positions without enough evidence
        if total_dp < min_total_depth or total_alt < min_alt_reads:
            continue

        # This position has enough alt reads — trigger the CNN
        top_alt_base = max(alt_counts, key=alt_counts.get)
        af = total_alt / total_dp if total_dp > 0 else 0.0

        tensor = extract_pileup_tensor(
            bam, fasta, contig, pos, window, depth,
            min_bq=min_bq, min_mq=min_mq,
        )
        input_tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        predicted_class, confidence, probs = model.predict_with_confidence(input_tensor)

        yield VariantCall(
            patient_id=patient_id,
            gene=gene_name,
            contig=contig,
            position=pos,
            ref_base=ref_base,
            alt_base=top_alt_base,
            ref_depth=ref_count,
            alt_depth=total_alt,
            total_depth=total_dp,
            allele_frequency=round(af, 4),
            predicted_class=predicted_class,
            class_name=CLASS_NAMES[predicted_class],
            confidence=round(confidence, 4),
            probabilities=[round(p, 4) for p in probs.tolist()],
        )
