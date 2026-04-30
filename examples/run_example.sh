#!/bin/bash
# ============================================================
# minDeepVariant — Example Run Script
# ============================================================
# This script demonstrates the full workflow:
#   1. Train the CNN on synthetic data
#   2. Scan patient BAMs against the M. leprae AMR panel
#   3. Review output files
#
# Prerequisites:
#   - pip install mindeepvariant
#   - Indexed BAM files in a directory
#   - Reference FASTA with .fai index
# ============================================================

set -e  # Exit on error

# --- Configuration ---
BAM_DIR="/path/to/aligned_bams"
REF="/path/to/GCF_000195855.1_ASM19585v1_genomic.fna"
PANEL="configs/leprae.json"
OUTDIR="results"
WEIGHTS="mindv_weights.pth"

# --- Step 1: Train the model ---
echo "═══════════════════════════════════════════════════"
echo "  Step 1: Training minDeepVariant CNN"
echo "═══════════════════════════════════════════════════"

if [ -f "$WEIGHTS" ]; then
    echo "Weights file already exists, skipping training."
else
    mindeepvariant train \
        --epochs 30 \
        --samples 1000 \
        --lr 0.0005 \
        --output "$WEIGHTS"
fi

# --- Step 2: Scan the patient cohort ---
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Step 2: Scanning patient cohort"
echo "═══════════════════════════════════════════════════"

mindeepvariant scan \
    --bam_dir "$BAM_DIR" \
    --ref "$REF" \
    --panel "$PANEL" \
    --weights "$WEIGHTS" \
    --outdir "$OUTDIR" \
    --min_alt 3 \
    --min_depth 15

# --- Step 3: Review results ---
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Step 3: Results"
echo "═══════════════════════════════════════════════════"

echo "Output directory:"
ls -la "$OUTDIR"/

echo ""
echo "Master Summary (first 20 lines):"
head -20 "$OUTDIR/Master_Clinical_Summary.csv"

echo ""
echo "Count of variants by tier:"
echo -n "  KNOWN: " && grep -c "KNOWN" "$OUTDIR/Master_Clinical_Summary.csv" || echo "0"
echo -n "  NOVEL: " && grep -c "NOVEL" "$OUTDIR/Master_Clinical_Summary.csv" || echo "0"

echo ""
echo "Done! Check $OUTDIR/ for full reports and PDF tensor plots."
