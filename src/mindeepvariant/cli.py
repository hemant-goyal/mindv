"""
mindv CLI — entry point for all subcommands.

Subcommands
-----------
train        Train the CNN on synthetic pileup images.
test         Smoke-test the trained model; prints confusion matrix.
scan         Run AMR variant calling on a directory of BAM files.
panel-build  Build and verify a mindv panel JSON from a resistance database.

Class layout (from model.py):
    CLASS_NAMES = ["Hom-Ref", "Het-SNP", "Hom-Alt-SNP", "Deletion"]  # 4 classes

Integration notes:
    - model.py  : MinDeepVariantCNN(window, depth, n_classes=4)
                  load_model(weights_path, window, depth, n_classes)
    - train.py  : train_model(window, depth, epochs, samples_per_epoch,
                              learning_rate, output_path) → (model, history)
                  generate_synthetic_pileup(class_label, window, depth)
    - scanner.py: scan_region(model, bam, fasta, contig, start_pos, end_pos,
                              patient_id, gene_name, min_alt_reads,
                              min_total_depth, window, depth, min_bq, min_mq)
                              → yields VariantCall
                  open_genomic_files(bam_path, fasta_path) → context manager
                  VariantCall fields: patient_id, gene, contig, position,
                      ref_base, alt_base, ref_depth, alt_depth, total_depth,
                      allele_frequency, predicted_class, class_name,
                      confidence, probabilities
    - annotator.py: annotate_variant(variant_call, panel_targets, fasta_path)
                                    → AnnotatedVariant
                    load_panel(panel_path) → full panel dict
                    AnnotatedVariant adds: aa_change, mutation_type, impact,
                        tier, known_drug, known_literature
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Filter presets
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict] = {
    "raw": {
        "min_af": 0.0, "min_dp": 15, "min_ad": 3,
        "min_bq": 20, "min_mq": 20, "min_confidence": 0.0,
    },
    "sensitive": {
        "min_af": 0.02, "min_dp": 10, "min_ad": 3,
        "min_bq": 20, "min_mq": 20, "min_confidence": 0.60,
    },
    "default": {
        "min_af": 0.10, "min_dp": 10, "min_ad": 4,
        "min_bq": 20, "min_mq": 20, "min_confidence": 0.75,
    },
    "clinical": {
        "min_af": 0.70, "min_dp": 20, "min_ad": 10,
        "min_bq": 20, "min_mq": 30, "min_confidence": 0.90,
    },
}

ALL_FILTER_PARAMS = ["min_af", "min_confidence", "min_dp", "min_ad", "min_bq", "min_mq"]


def _apply_preset(args: argparse.Namespace) -> None:
    preset_name = getattr(args, "preset", None) or "default"
    if preset_name not in PRESETS:
        sys.exit(f"[mindv] Unknown preset '{preset_name}'. Choose: {list(PRESETS)}")
    for k, v in PRESETS[preset_name].items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)


def _fill_defaults(args: argparse.Namespace) -> None:
    for k, v in PRESETS["default"].items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)


# ---------------------------------------------------------------------------
# BAM discovery
# ---------------------------------------------------------------------------

def _discover_bams(bam_dir: str) -> List[str]:
    import glob
    all_bams = [b for b in glob.glob(os.path.join(bam_dir, "**", "*.bam"), recursive=True)
                if not b.endswith(".bai")]
    by_folder: Dict[str, List[str]] = collections.defaultdict(list)
    for b in all_bams:
        by_folder[os.path.dirname(b)].append(b)
    selected = []
    for bams in by_folder.values():
        dedup = [b for b in bams if "_marked_duplicates.bam" in os.path.basename(b)]
        selected.append(dedup[0] if dedup else bams[0])
    return sorted(selected)


def _patient_id(bam_path: str) -> str:
    name = os.path.basename(bam_path)
    for sfx in ("_marked_duplicates.bam", ".bam"):
        if name.endswith(sfx):
            return name[:-len(sfx)]
    return name


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    from .train import train_model  # type: ignore

    # train_model(window, depth, epochs, samples_per_epoch, learning_rate, output_path)
    train_model(
        window=21,
        depth=30,
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        learning_rate=args.lr,
        output_path=args.output,
    )


def _add_train_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="Train the CNN on synthetic pileup images.")
    p.add_argument("--output", default="mindv_weights.pth",
                   help="Path to save trained weights (default: mindv_weights.pth)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--samples-per-epoch", type=int, default=1000, dest="samples_per_epoch",
                   help="Synthetic samples per epoch (default: 1000)")
    p.add_argument("--lr", type=float, default=0.0005,
                   help="Learning rate (default: 0.0005)")
    p.set_defaults(func=cmd_train)


# ---------------------------------------------------------------------------
# Subcommand: test
# ---------------------------------------------------------------------------

def cmd_test(args: argparse.Namespace) -> None:
    import torch
    from .model import MinDeepVariantCNN, CLASS_NAMES, load_model  # type: ignore
    from .train import generate_synthetic_pileup                   # type: ignore

    weights_path = Path(args.weights)
    if not weights_path.exists():
        sys.exit(f"[mindv test] Weights not found: {weights_path}. Run `mindv train` first.")

    # 4 classes: Hom-Ref, Het-SNP, Hom-Alt-SNP, Deletion
    n_classes = len(CLASS_NAMES)
    model = load_model(str(weights_path))
    model.eval()

    n = args.n_per_class
    correct = 0
    total = 0
    confusion: Dict[int, Dict[int, int]] = {
        i: {j: 0 for j in range(n_classes)} for i in range(n_classes)
    }

    print(f"[mindv test] {n * n_classes} synthetic samples, {n_classes} classes …")
    with torch.no_grad():
        for true_label in range(n_classes):
            for _ in range(n):
                tensor = generate_synthetic_pileup(true_label)
                x = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                logits = model(x)
                pred = int(logits.argmax(dim=1).item())
                confusion[true_label][pred] += 1
                if pred == true_label:
                    correct += 1
                total += 1

    accuracy = correct / total
    print(f"\nAccuracy: {accuracy:.1%}  ({correct}/{total})\n")
    header = f"{'':14s}" + "".join(f"{c:>14s}" for c in CLASS_NAMES)
    print(header)
    for i in range(n_classes):
        row = f"{CLASS_NAMES[i]:14s}" + "".join(f"{confusion[i][j]:>14d}" for j in range(n_classes))
        print(row)
    print()

    if accuracy < 0.85:
        print(f"[WARN] Accuracy {accuracy:.1%} < 85 %. Consider retraining.")
        sys.exit(1)
    else:
        print(f"[OK] Accuracy {accuracy:.1%} ≥ 85 %.")


def _add_test_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("test", help="Smoke-test the model with synthetic data.")
    p.add_argument("--weights", default="mindv_weights.pth")
    p.add_argument("--n-per-class", type=int, default=200, dest="n_per_class")
    p.set_defaults(func=cmd_test)


# ---------------------------------------------------------------------------
# Subcommand: scan
# ---------------------------------------------------------------------------

def cmd_scan(args: argparse.Namespace) -> None:
    from .model import load_model                                   # type: ignore
    from .scanner import open_genomic_files, scan_region           # type: ignore
    from .annotator import annotate_variant, load_panel
    from .tb_resistance_classifier import write_cohort_classification            # type: ignore

    _apply_preset(args)
    _fill_defaults(args)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        sys.exit(f"[mindv scan] Weights not found: {weights_path}. Run `mindv train` first.")

    model = load_model(str(weights_path))
    panel = load_panel(args.panel)

    print(f"[mindv scan] {panel.get('organism','?')}  ref={panel.get('reference','?')}  "
          f"version={panel.get('panel_version','?')}")
    print(f"[mindv scan] Filters → preset={args.preset or 'default'}  "
          f"AF≥{args.min_af}  DP≥{args.min_dp}  AD≥{args.min_ad}  "
          f"conf≥{args.min_confidence}")

    bam_files = _discover_bams(args.bam_dir)
    if not bam_files:
        sys.exit(f"[mindv scan] No BAM files found in {args.bam_dir}")
    print(f"[mindv scan] Found {len(bam_files)} BAM file(s)\n")

    args.outdir = _resolve_outdir(args.outdir, args.overwrite)
    panel_targets = panel.get("targets", {})
    contig = panel.get("reference", "")

    all_rows = []
    total_candidates = total_reported = total_filtered = 0

    for bam_path in bam_files:
        pid = _patient_id(bam_path)
        print(f"  Scanning {pid} …")

        rows = []
        candidates = reported = filtered = 0

        # open_genomic_files is a context manager yielding (bam, fasta)
        with open_genomic_files(bam_path, args.ref) as (bam, fasta):
            for target_name, target in panel_targets.items():
                # Per-target MQ override: rRNA genes have low-MQ reads
                # by design (repetitive sequence). Using the global min_mq
                # filters ALL reads from rrs/rrl, giving zero depth.
                # The panel JSON can set "min_mq": 0 to override this.
                target_min_mq   = target.get("min_mq",   args.min_mq)
                target_min_bq   = target.get("min_bq",   args.min_bq)
                # rRNA targets may need a lower confidence threshold because
                # the CNN was trained on coding-sequence pileups only
                target_min_conf = target.get("min_confidence", args.min_confidence)

                # scan_region takes open handles, not paths
                for call in scan_region(
                    model=model,
                    bam=bam,
                    fasta=fasta,
                    contig=contig,
                    start_pos=target["start"],
                    end_pos=target["end"],
                    patient_id=pid,
                    gene_name=target_name,
                    min_bq=target_min_bq,
                    min_mq=target_min_mq,
                ):
                    candidates += 1

                    # annotate_variant(call, panel_targets_dict, fasta_path)
                    ann = annotate_variant(call, panel_targets, args.ref)

                    filt_reason = _filter_call(call, args,
                                                   min_confidence=target_min_conf)

                    # Special case: CNN said Hom-Ref but it matches a KNOWN panel
                    # mutation. Keep it if AF >= 5% — resistance at sub-clonal AF is
                    # clinically important. Below 5%, it is almost certainly sequencing
                    # noise and should remain filtered.
                    if (filt_reason == "Hom-Ref" and ann.tier == "KNOWN"
                            and call.allele_frequency >= 0.05):
                        filt_reason = None
                    rows.append({
                        "patient":      pid,
                        "target":       target_name,
                        "pos":          call.position,
                        "ref":          call.ref_base,
                        "alt":          call.alt_base,
                        "dp":           call.total_depth,
                        "af":           call.allele_frequency,      # allele_frequency not allele_freq
                        "ad":           call.alt_depth,
                        "class":        call.class_name,             # class_name not call_type
                        "confidence":   call.confidence,
                        "aa_change":    ann.aa_change,
                        "mut_type":     ann.mutation_type,           # mutation_type not effect
                        "impact":       ann.impact,
                        "tier":         ann.tier if not filt_reason else "FILT",
                        "known_drug":   ann.known_drug,
                        "filter":       filt_reason or "PASS",
                    })
                    if filt_reason:
                        filtered += 1
                    else:
                        reported += 1

        total_candidates += candidates
        total_reported   += reported
        total_filtered   += filtered
        all_rows.extend(rows)

        report_path = os.path.join(args.outdir, f"{pid}_mindv.txt")
        _write_patient_report(report_path, rows, pid)

    csv_path = os.path.join(args.outdir, "cohort_mindv.csv")
    _write_cohort_csv(csv_path, all_rows)

    print(f"\nCohort: {total_candidates} candidates  |  "
          f"{total_reported} reported  |  {total_filtered} filtered")
    # WHO resistance classification — auto-runs after every TB scan
    organism = panel.get("organism", "").lower() if isinstance(panel, dict) else ""
    cohort_csv = os.path.join(args.outdir, "cohort_mindv.csv")
    cls_tsv    = os.path.join(args.outdir, "cohort_resistance_classification.tsv")
    if os.path.exists(cohort_csv) and "tuberculosis" in organism:
        try:
            write_cohort_classification(cohort_csv, cls_tsv)
            print(f"WHO classification → {cls_tsv}")
        except Exception as e:
            print(f"[warn] WHO classification skipped: {e}")
    print(f"Results → {args.outdir}/")


def _filter_call(call, args, min_confidence: float = None) -> Optional[str]:
    # Use actual VariantCall field names
    # min_confidence can be overridden per-target (e.g. lower for rRNA regions)
    effective_min_conf = min_confidence if min_confidence is not None else args.min_confidence

    # Class 0 (Hom-Ref): CNN classified this position as reference.
    # Only report Hom-Ref if it is a KNOWN panel mutation (rare but possible
    # when a resistance mutation is at sub-clonal frequency and the CNN hedges).
    # For all other Hom-Ref calls, filter immediately — they are noise.
    if call.predicted_class == 0:
        return "Hom-Ref"   # will be overridden to PASS for KNOWN entries in cli

    if call.allele_frequency < args.min_af:
        return f"AF={call.allele_frequency:.4f}<{args.min_af}"
    if call.total_depth < args.min_dp:
        return f"DP={call.total_depth}<{args.min_dp}"
    if call.alt_depth < args.min_ad:
        return f"AD={call.alt_depth}<{args.min_ad}"
    if call.confidence < effective_min_conf:
        return f"conf={call.confidence:.2f}<{effective_min_conf}"
    return None


def _write_patient_report(path: str, rows: list, pid: str) -> None:
    with open(path, "w") as fh:
        fh.write(f"# mindv report — {pid}\n")
        fh.write(f"# {'POS':>10}  {'REF':>3}  {'ALT':>3}  {'DP':>6}  {'AD':>5}  "
                 f"{'AF':>8}  {'CLASS':>12}  {'CONF':>6}  {'AA_CHANGE':>15}  "
                 f"{'MUT_TYPE':>12}  {'TIER':>5}  {'DRUG':>20}  FILTER\n")
        for r in rows:
            fh.write(
                f"  {r['pos']:>10}  {r['ref']:>3}  {r['alt']:>3}  "
                f"{r['dp']:>6}  {r['ad']:>5}  {r['af']:>8.4f}  "
                f"{r['class']:>12}  {r['confidence']:>6.2f}  "
                f"{r['aa_change']:>15}  {r['mut_type']:>12}  "
                f"{r['tier']:>5}  {r['known_drug']:>20}  {r['filter']}\n"
            )


def _resolve_outdir(requested: str, overwrite: bool) -> str:
    """
    Decide the actual output directory to use.

    Rules
    -----
    --overwrite given  : use the requested path as-is (delete contents if needed)
    directory absent   : create and use it
    directory present  : add a timestamp suffix  →  mindv_results_20250513_1430
    permission denied  : fall back to ~/mindv_TIMESTAMP

    This prevents accidental overwriting of previous results while keeping
    the directory name human-readable and chronologically sortable.
    """
    requested = os.path.abspath(requested)

    if overwrite:
        os.makedirs(requested, exist_ok=True)
        print(f"[mindv] Output → {requested}  (overwrite mode)")
        return requested

    if not os.path.exists(requested):
        os.makedirs(requested)
        print(f"[mindv] Output → {requested}")
        return requested

    # Directory exists — add timestamp suffix
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    new_dir = f"{requested}_{stamp}"
    try:
        os.makedirs(new_dir, exist_ok=True)
        print(f"[mindv] Output directory '{os.path.basename(requested)}' already exists.")
        print(f"[mindv] Writing to new directory → {new_dir}")
        print(f"[mindv] Tip: use --overwrite to write into the existing directory.")
        return new_dir
    except PermissionError:
        # Windows NTFS /mnt/d/ can be read-only for new dirs
        fallback = os.path.join(
            os.path.expanduser("~"),
            f"mindv_results_{stamp}"
        )
        os.makedirs(fallback, exist_ok=True)
        print(f"[mindv] WARNING: permission denied on {new_dir}")
        print(f"[mindv] Writing to home directory instead → {fallback}")
        return fallback


def _write_cohort_csv(path: str, rows: list) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _add_scan_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("scan", help="Run AMR variant calling on BAM files.")
    p.add_argument("--bam-dir", required=True, dest="bam_dir")
    p.add_argument("--ref", required=True)
    p.add_argument("--panel", required=True)
    p.add_argument("--outdir", default="mindv_results")
    p.add_argument("--weights", default="mindv_weights.pth")
    p.add_argument("--preset", choices=list(PRESETS), default=None)
    p.add_argument("--min-af",         type=float, default=None, dest="min_af")
    p.add_argument("--min-dp",         type=int,   default=None, dest="min_dp")
    p.add_argument("--min-ad",         type=int,   default=None, dest="min_ad")
    p.add_argument("--min-bq",         type=int,   default=None, dest="min_bq")
    p.add_argument("--min-mq",         type=int,   default=None, dest="min_mq")
    p.add_argument("--min-confidence", type=float, default=None, dest="min_confidence")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Overwrite existing output directory instead of creating a new timestamped one")
    p.set_defaults(func=cmd_scan)


# ---------------------------------------------------------------------------
# Subcommand: panel-build
# ---------------------------------------------------------------------------

def _add_panel_build_parser(sub: argparse._SubParsersAction) -> None:
    from .cli_panelbuild import add_panel_build_subparser  # type: ignore
    add_panel_build_subparser(sub)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mindv",
        description="mindv — Minimal Deep Variant caller for haploid AMR profiling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="mindv 1.1.0")
    sub = parser.add_subparsers(dest="subcommand", metavar="COMMAND")
    sub.required = True

    _add_train_parser(sub)
    _add_test_parser(sub)
    _add_scan_parser(sub)
    _add_panel_build_parser(sub)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
