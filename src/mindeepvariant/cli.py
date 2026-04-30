"""
cli.py — Command-Line Interface for minDeepVariant

Usage:
    # Train a new model on synthetic data
    mindeepvariant train --output weights.pth --epochs 30

    # Scan a cohort of patients against an AMR panel (default settings)
    mindeepvariant scan --bam_dir /path/to/bams --ref genome.fna \\
        --panel leprae.json --outdir results/

    # Strict clinical reporting: only near-fixed, high-confidence calls
    mindeepvariant scan ... --preset clinical

    # Sensitive heteroresistance / mixed-population mode
    mindeepvariant scan ... --preset sensitive

Subcommands:
    train   Train the CNN on synthetic pileup data.
    scan    Scan BAM files against an AMR gene panel.
"""

import argparse
import csv
import glob
import logging
import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for batch PDF generation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .model import load_model, CLASS_NAMES
from .train import train_model
from .scanner import (
    open_genomic_files,
    validate_inputs,
    extract_pileup_tensor,
    scan_region,
)
from .annotator import load_panel, annotate_variant

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Filter presets
# ─────────────────────────────────────────────────────────────────────
# Each preset is a dict of (cli arg name -> value) that overrides
# whatever defaults the user didn't explicitly set. Documented in README.
PRESETS = {
    "default": {
        "min_af": 0.10,
        "min_dp": 10,
        "min_ad": 4,
        "min_bq": 20,
        "min_mq": 20,
        "min_confidence": 0.75,
    },
    "clinical": {
        # Strict — for clinical reporting on clonal isolates.
        # Only call near-fixed variants with strong model confidence.
        "min_af": 0.70,
        "min_dp": 20,
        "min_ad": 10,
        "min_bq": 20,
        "min_mq": 30,
        "min_confidence": 0.90,
    },
    "sensitive": {
        # Permissive — for heteroresistance and mixed-population studies.
        # Catches subclonal variants but expect more noise; manual review
        # of low-AF calls is recommended.
        "min_af": 0.02,
        "min_dp": 10,
        "min_ad": 3,
        "min_bq": 20,
        "min_mq": 20,
        "min_confidence": 0.60,
    },
    "raw": {
        # No filtering — show every CNN-positive candidate.
        # Reproduces pre-v1.0 behavior. Use for debugging, method
        # validation, and the tutorial notebook's "what does noise
        # look like?" demonstration. NOT for clinical interpretation.
        "min_af": 0.0,
        "min_dp": 15,
        "min_ad": 3,
        "min_bq": 20,
        "min_mq": 20,
        "min_confidence": 0.0,
    },
}


def _apply_preset(args):
    """Apply preset values, but never overwrite explicit user flags."""
    if not args.preset:
        return
    preset = PRESETS[args.preset]
    for key, value in preset.items():
        # Only apply preset value if user didn't pass the flag explicitly.
        # We detect "explicit" by checking whether the current value
        # differs from our argparse default (None sentinel).
        if getattr(args, key) is None:
            setattr(args, key, value)
    logger.info(f"Applied '{args.preset}' preset")


def _fill_defaults(args):
    """Backstop: if neither preset nor explicit flag set a value, use 'default'."""
    fallback = PRESETS["default"]
    for key, value in fallback.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def cmd_train(args):
    """Train subcommand handler."""
    model, history = train_model(
        window=args.window,
        depth=args.depth,
        epochs=args.epochs,
        samples_per_epoch=args.samples,
        learning_rate=args.lr,
        output_path=args.output,
    )
    logger.info("Training complete.")

def cmd_test(args):
    """
    Test subcommand — validates the trained CNN on synthetic pileup data.

    Generates synthetic pileups for each of the four variant classes,
    runs inference, and reports a confusion matrix with overall accuracy.
    This tests the model weights and inference pipeline without requiring
    any BAM files, reference genomes, or external data.
    """
    model = load_model(args.weights, window=args.window, depth=args.depth)
    logger.info(f"Model loaded from {args.weights}")

    from .train import generate_synthetic_pileup

    n_per_class = args.samples_per_class
    n_classes = len(CLASS_NAMES)
    total = n_per_class * n_classes

    logger.info(f"Generating {total} synthetic pileups ({n_per_class} per class)...")

    confusion = [[0] * n_classes for _ in range(n_classes)]
    correct = 0

    for true_class in range(n_classes):
        for _ in range(n_per_class):
            tensor = generate_synthetic_pileup(true_class, args.window, args.depth)
            input_tensor = tensor.unsqueeze(0).unsqueeze(0)
            predicted_class, confidence, _ = model.predict_with_confidence(input_tensor)
            confusion[true_class][predicted_class] += 1
            if predicted_class == true_class:
                correct += 1

    accuracy = correct / total * 100

    short_names = ["Ref", "Het", "Hom", "Del"]
    print(f"\n{'='*50}")
    print(f"  mindv Synthetic Test")
    print(f"{'='*50}")
    print(f"\nConfusion Matrix ({n_per_class} samples per class):\n")
    print(f"{'':>10}  {'--- Predicted ---':^28}")
    print(f"{'Actual':>10}  {short_names[0]:>6} {short_names[1]:>6} {short_names[2]:>6} {short_names[3]:>6}")
    for i in range(n_classes):
        row_vals = "  ".join(f"{confusion[i][j]:>4}" for j in range(n_classes))
        print(f"{short_names[i]:>10}  {row_vals}")

    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")

    threshold = 85.0
    if accuracy >= threshold:
        print(f"PASS (>= {threshold}%)\n")
    else:
        print(f"FAIL (< {threshold}%) — retrain with: mindv train\n")
        sys.exit(1)

def cmd_scan(args):
    """Scan subcommand handler — the main clinical pipeline."""

    # 0. Resolve filter parameters from preset + explicit flags
    _apply_preset(args)
    _fill_defaults(args)

    logger.info(
        f"Filters: AF>={args.min_af} DP>={args.min_dp} AD>={args.min_ad} "
        f"BQ>={args.min_bq} MQ>={args.min_mq} CONF>={args.min_confidence}"
    )

    # 1. Load the model
    model = load_model(args.weights, window=args.window, depth=args.depth)
    logger.info(f"Model loaded from {args.weights}")

    # 2. Load the AMR panel
    panel = load_panel(args.panel)
    targets = panel["targets"]
    contig = panel.get("reference", args.contig)

    # 3. Find BAM files and prioritize GATK-deduplicated files
    import collections
    all_bams = glob.glob(os.path.join(args.bam_dir, "**", "*.bam"), recursive=True)

    # Group BAMs by their parent directory (each patient gets a folder)
    patient_folders = collections.defaultdict(list)
    for b in all_bams:
        if not b.endswith(".bai"):
            patient_folders[os.path.dirname(b)].append(b)

    bam_files = []
    for folder, bams_in_folder in patient_folders.items():
        dedup_bams = [b for b in bams_in_folder if "_marked_duplicates.bam" in os.path.basename(b)]
        if dedup_bams:
            bam_files.append(dedup_bams[0])
        elif bams_in_folder:
            bams_in_folder.sort()
            bam_files.append(bams_in_folder[0])

    bam_files.sort()

    if not bam_files:
        logger.error(f"No BAM files found in {args.bam_dir}")
        sys.exit(1)

    # 4. Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # 5. Accumulate all results for the master CSV
    all_annotated = []          # variants that pass ALL filters
    total_candidates = 0        # everything the CNN classified as variant (pre-filter)

    # 6. Process each patient
    for bam_path in bam_files:
        patient_id = (
            os.path.basename(bam_path)
            .replace("_marked_duplicates.bam", "")
            .replace(".bam", "")
        )

        try:
            validate_inputs(bam_path, args.ref, contig)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Skipping {patient_id}: {e}")
            continue

        logger.info(f"Scanning patient: {patient_id}")

        txt_path = os.path.join(args.outdir, f"{patient_id}_report.txt")
        pdf_path = os.path.join(args.outdir, f"{patient_id}_plots.pdf")

        patient_variants = []
        patient_filtered_count = 0  # how many candidates were dropped

        with open_genomic_files(bam_path, args.ref) as (bam, fasta), \
             open(txt_path, "w") as txt_out, \
             PdfPages(pdf_path) as pdf_doc:

            txt_out.write(f"{'='*70}\n")
            txt_out.write(f"  minDeepVariant AMR Report | Patient: {patient_id}\n")
            txt_out.write(f"  Organism: {panel.get('organism', 'N/A')}\n")
            txt_out.write(
                f"  Filters: AF>={args.min_af} DP>={args.min_dp} "
                f"AD>={args.min_ad} BQ>={args.min_bq} MQ>={args.min_mq} "
                f"CONF>={args.min_confidence}\n"
            )
            txt_out.write(f"{'='*70}\n\n")

            for gene_name, coords in targets.items():
                txt_out.write(f"--- {gene_name} ({contig}:{coords['start']}-{coords['end']}) ---\n")
                txt_out.write(
                    f"{'POS':<10} {'REF':<4} {'ALT':<4} {'DP':<6} "
                    f"{'AF':<6} {'CLASS':<16} {'CONF':<6} "
                    f"{'AA_CHANGE':<14} {'TYPE':<12} {'TIER':<6} {'DRUG'}\n"
                )
                txt_out.write("-" * 100 + "\n")

                gene_hits = 0

                for call in scan_region(
                    model=model,
                    bam=bam,
                    fasta=fasta,
                    contig=contig,
                    start_pos=coords["start"],
                    end_pos=coords["end"],
                    patient_id=patient_id,
                    gene_name=gene_name,
                    min_alt_reads=args.min_ad,        # heuristic floor uses min_ad
                    min_total_depth=args.min_dp,      # heuristic floor uses min_dp
                    window=args.window,
                    depth=args.depth,
                    min_bq=args.min_bq,
                    min_mq=args.min_mq,
                ):
                    # Annotate with protein consequence + tier
                    ann = annotate_variant(call, targets, args.ref)

                    # Skip wild-type calls entirely (don't even write to text report)
                    if ann.predicted_class == 0:
                        continue

                    total_candidates += 1

                    # ─── Filter cascade ─────────────────────────────
                    # Reasons logged in patient text report for transparency.
                    fail_reasons = []
                    if ann.allele_frequency < args.min_af:
                        fail_reasons.append(f"AF={ann.allele_frequency:.4f}<{args.min_af}")
                    if ann.total_depth < args.min_dp:
                        fail_reasons.append(f"DP={ann.total_depth}<{args.min_dp}")
                    if call.alt_depth < args.min_ad:
                        fail_reasons.append(f"AD={call.alt_depth}<{args.min_ad}")
                    if ann.confidence < args.min_confidence:
                        fail_reasons.append(f"CONF={ann.confidence:.2f}<{args.min_confidence}")

                    if fail_reasons:
                        patient_filtered_count += 1
                        txt_out.write(
                            f"{ann.position:<10} {ann.ref_base:<4} {ann.alt_base:<4} "
                            f"{ann.total_depth:<6} {ann.allele_frequency:<6.4f} "
                            f"{ann.class_name:<16} {ann.confidence:<6.2f} "
                            f"{ann.aa_change:<14} {ann.mutation_type:<12} "
                            f"{'FILT':<6} [filtered: {', '.join(fail_reasons)}]\n"
                        )
                        continue

                    # ─── Passed all filters ─────────────────────────
                    txt_out.write(
                        f"{ann.position:<10} {ann.ref_base:<4} {ann.alt_base:<4} "
                        f"{ann.total_depth:<6} {ann.allele_frequency:<6.4f} "
                        f"{ann.class_name:<16} {ann.confidence:<6.2f} "
                        f"{ann.aa_change:<14} {ann.mutation_type:<12} "
                        f"{ann.tier:<6} {ann.known_drug}\n"
                    )

                    gene_hits += 1
                    patient_variants.append(ann)
                    all_annotated.append(ann)

                    # Generate PDF plot for this variant
                    tensor = extract_pileup_tensor(
                        bam, fasta, contig, ann.position,
                        args.window, args.depth,
                        min_bq=args.min_bq, min_mq=args.min_mq,
                    )
                    fig = plt.figure(figsize=(10, 6))
                    plt.imshow(tensor.numpy(), cmap="gray", aspect="auto")

                    tier_tag = f"[{ann.tier}]" if ann.tier == "KNOWN" else f"[{ann.tier} - INVESTIGATE]"
                    title = (
                        f"{gene_name} | {contig}:{ann.position} | "
                        f"Depth: {ann.total_depth}x | AF: {ann.allele_frequency:.3f}\n"
                        f"{ann.class_name} (conf: {ann.confidence:.2f}) | "
                        f"{ann.aa_change} | {ann.mutation_type}\n"
                        f"{tier_tag} {ann.known_drug}"
                    )
                    plt.title(title, fontsize=11, fontweight="bold", pad=15)
                    plt.ylabel("Row 0: Reference | Rows 1+: Patient Reads")
                    plt.xlabel(f"Genomic Window (center = {ann.position})")
                    plt.axhline(y=0.5, color="red", linewidth=2)
                    plt.axvline(x=args.window // 2, color="cyan",
                                linestyle="--", linewidth=2, alpha=0.7)
                    plt.colorbar(label="Base Intensity")
                    plt.tight_layout()
                    pdf_doc.savefig(fig)
                    plt.close(fig)

                if gene_hits == 0:
                    txt_out.write("  Wild-Type (no variants passed filters)\n")
                txt_out.write("\n")

            # Patient summary
            known_count = sum(1 for v in patient_variants if v.tier == "KNOWN")
            novel_count = sum(1 for v in patient_variants if v.tier == "NOVEL")
            txt_out.write(f"\n{'='*70}\n")
            txt_out.write(f"  SUMMARY\n")
            txt_out.write(f"  Reported variants:    {len(patient_variants)}\n")
            txt_out.write(f"    Tier 1 (Known):     {known_count}\n")
            txt_out.write(f"    Tier 2 (Novel):     {novel_count}\n")
            txt_out.write(f"  Filtered out:         {patient_filtered_count}\n")
            txt_out.write(f"{'='*70}\n")

        logger.info(
            f"  {patient_id}: {len(patient_variants)} reported "
            f"({known_count} known, {novel_count} novel), "
            f"{patient_filtered_count} filtered"
        )

    # 7. Write master CSV
    csv_path = os.path.join(args.outdir, "Master_Clinical_Summary.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "PATIENT", "GENE", "CONTIG", "POS", "REF", "ALT",
            "TOTAL_DP", "AF", "CLASS", "CONFIDENCE",
            "AA_CHANGE", "MUT_TYPE", "IMPACT", "TIER",
            "KNOWN_DRUG", "LITERATURE",
        ])
        for v in all_annotated:
            writer.writerow([
                v.patient_id, v.gene, v.contig, v.position,
                v.ref_base, v.alt_base, v.total_depth,
                v.allele_frequency, v.class_name, v.confidence,
                v.aa_change, v.mutation_type, v.impact, v.tier,
                v.known_drug, v.known_literature,
            ])

    filtered_total = total_candidates - len(all_annotated)
    logger.info(
        f"Cohort summary: {total_candidates} CNN-positive candidates, "
        f"{len(all_annotated)} reported, {filtered_total} filtered out"
    )
    logger.info(f"Master CSV saved: {csv_path} ({len(all_annotated)} total variants)")
    logger.info("Cohort scan complete.")


def main():
    """Entry point for the mindeepvariant CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="mindv",
        description=(
            "minDeepVariant: A minimal, from-scratch deep learning variant caller. "
            "Treats variant calling as image classification, inspired by Google's "
            "DeepVariant and Karpathy's minGPT philosophy."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Train subcommand ──────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Train the CNN on synthetic data")
    train_parser.add_argument("--output", default="mindv_weights.pth",
                              help="Output path for trained weights (default: mindv_weights.pth)")
    train_parser.add_argument("--epochs", type=int, default=30,
                              help="Number of training epochs (default: 30)")
    train_parser.add_argument("--samples", type=int, default=1000,
                              help="Synthetic samples per epoch (default: 1000)")
    train_parser.add_argument("--lr", type=float, default=0.0005,
                              help="Learning rate (default: 0.0005)")
    train_parser.add_argument("--window", type=int, default=21,
                              help="Genomic window width (default: 21)")
    train_parser.add_argument("--depth", type=int, default=30,
                              help="Read depth for pileup tensors (default: 30)")

# ── Test subcommand ──────────────────────────────────
    test_parser = subparsers.add_parser("test", help="Test the CNN on synthetic data")
    test_parser.add_argument("--weights", default="mindv_weights.pth",
                             help="Path to trained model weights (default: mindv_weights.pth)")
    test_parser.add_argument("--samples-per-class", dest="samples_per_class",
                             type=int, default=25,
                             help="Synthetic samples per class (default: 25)")
    test_parser.add_argument("--window", type=int, default=21,
                             help="Genomic window width (default: 21)")
    test_parser.add_argument("--depth", type=int, default=30,
                             help="Read depth for pileup tensors (default: 30)")

    # ── Scan subcommand ───────────────────────────────────────
    scan_parser = subparsers.add_parser("scan", help="Scan BAM files against an AMR panel")
    scan_parser.add_argument("--bam_dir", required=True,
                             help="Directory containing BAM files (searched recursively)")
    scan_parser.add_argument("--ref", required=True,
                             help="Path to reference FASTA file")
    scan_parser.add_argument("--panel", required=True,
                             help="Path to AMR panel JSON config")
    scan_parser.add_argument("--outdir", required=True,
                             help="Output directory for reports")
    scan_parser.add_argument("--weights", default="mindv_weights.pth",
                             help="Path to trained model weights (default: mindv_weights.pth)")
    scan_parser.add_argument("--contig", default=None,
                             help="Override contig name (default: read from panel JSON)")
    scan_parser.add_argument("--window", type=int, default=21,
                             help="Genomic window width (default: 21)")
    scan_parser.add_argument("--depth", type=int, default=30,
                             help="Read depth for pileup tensors (default: 30)")

    # Filter / quality knobs.
    # Defaults are deliberately None — we resolve them from the preset
    # (or the 'default' preset as a backstop) in cmd_scan(). This lets us
    # detect which flags the user set explicitly so presets don't
    # silently overwrite them.
    filt = scan_parser.add_argument_group("Variant filtering")
    filt.add_argument("--preset", choices=["default", "clinical", "sensitive", "raw"],
                      default=None,
                      help="Filter preset: 'default' (balanced, AF>=0.10), "
                           "'clinical' (strict, AF>=0.70 — for reporting), "
                           "'sensitive' (AF>=0.02 — for heteroresistance). "
                           "Individual --min-* flags override preset values.")
    filt.add_argument("--min-af", dest="min_af", type=float, default=None,
                      help="Minimum allele frequency. Default: 0.10 "
                           "(haploid clonal organism noise floor).")
    filt.add_argument("--min-dp", dest="min_dp", type=int, default=None,
                      help="Minimum total read depth at position. Default: 10.")
    filt.add_argument("--min-ad", dest="min_ad", type=int, default=None,
                      help="Minimum alternate-allele supporting reads. "
                           "Also serves as the heuristic trigger floor. Default: 4.")
    filt.add_argument("--min-bq", dest="min_bq", type=int, default=None,
                      help="Minimum base quality (Phred) for a base to count "
                           "in the pileup. Applied at pysam level. Default: 20.")
    filt.add_argument("--min-mq", dest="min_mq", type=int, default=None,
                      help="Minimum mapping quality for a read to be included. "
                           "Applied at pysam level. Default: 20.")
    filt.add_argument("--min-confidence", dest="min_confidence", type=float, default=None,
                      help="Minimum CNN softmax confidence for the called class. "
                           "Default: 0.75.")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "scan":
        cmd_scan(args)
    elif args.command == "test":
        cmd_test(args)


if __name__ == "__main__":
    main()
