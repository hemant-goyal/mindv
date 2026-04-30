# minDeepVariant

**A minimal, from-scratch deep learning variant caller — the [minGPT](https://github.com/karpathy/minGPT) of genomics.**

minDeepVariant demonstrates [Google DeepVariant's](https://github.com/google/deepvariant) core insight in under 500 lines of PyTorch: **variant calling is image classification**. It converts DNA pileups into grayscale images and trains a CNN to classify them as wild-type, heterozygous SNP, homozygous SNP, or deletion.

Built for **education and rapid AMR profiling**, not to replace production variant callers.

---

## How It Works

### The Core Insight

Traditional variant callers use statistical models (GATK's HaplotypeCaller, bcftools). DeepVariant showed that if you *visualize* a pileup — stacking aligned reads as rows with each base mapped to a color — mutations appear as **vertical streaks** that a convolutional neural network can detect.

```
Position:    ...  8  9  10  11  12  ...
Reference:   ...  A  C  [G]  T   A  ...    ← Row 0
Read 1:      ...  A  C  [G]  T   A  ...    ← matches ref
Read 2:      ...  A  C  [T]  T   A  ...    ← ALT at center!
Read 3:      ...  A  C  [T]  T   A  ...    ← ALT at center!
Read 4:      ...  A  C  [G]  T   A  ...    ← matches ref
```

When encoded as pixel intensities (A=0.25, C=0.50, G=0.75, T=1.00), that `G→T` change creates a visible vertical discontinuity in the image that the CNN learns to detect.

### The Pipeline

```
BAM File → Pileup Extraction → Grayscale Tensor → CNN Classification → Annotation
                                    ↓
                              [21 × 31 image]
                              Row 0: Reference
                              Rows 1-30: Reads
                                    ↓
                              4-class output:
                              0: Hom-Ref (wild-type)
                              1: Het-SNP
                              2: Hom-Alt SNP
                              3: Deletion
```

### Architecture

```
Input (1×31×21) → Conv2d(16) → ReLU → MaxPool
                → Conv2d(32) → ReLU → MaxPool
                → Linear(64) → ReLU → Dropout(0.3)
                → Linear(4) → Softmax
```

### Tiered Annotation

Variants are classified into two tiers:

| Tier | Meaning | Action |
|------|---------|--------|
| **KNOWN** | Matches a catalogued resistance mutation in the panel JSON | Report as confirmed resistance |
| **NOVEL** | Called by the CNN in a drug-target region but not in the database | Flag for investigation |

This tiered approach is what distinguishes minDeepVariant from pure "lookup" tools like Mykrobe or NTM-Profiler, which can only report variants already in their databases.

---

## Installation

```bash
# From source
git clone https://github.com/hemant-goyal/minDeepVariant.git
cd minDeepVariant
pip install -e .

# Or with Docker
docker build -t mindeepvariant .
```

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.10
- pysam ≥ 0.19
- Biopython ≥ 1.79
- matplotlib ≥ 3.5

---

## Quick Start

### 1. Train the model

```bash
mindeepvariant train --epochs 30 --samples 1000 --output mindv_weights.pth
```

This generates synthetic pileup images and trains the CNN. Takes ~2 minutes on CPU.

### 2. Scan a patient cohort

```bash
mindeepvariant scan \
    --bam_dir /path/to/aligned_bams/ \
    --ref /path/to/reference.fna \
    --panel configs/leprae.json \
    --weights mindv_weights.pth \
    --outdir results/
```

### 3. Check results

```
results/
├── Patient1_report.txt          # Per-patient text report
├── Patient1_plots.pdf           # Pileup tensor visualizations
├── Patient2_report.txt
├── Patient2_plots.pdf
└── Master_Clinical_Summary.csv  # Cohort-wide CSV with all variants
```

---

## Panel Configuration

minDeepVariant is **organism-agnostic**. Supply a JSON panel for any bacterium:

```json
{
    "organism": "Mycobacterium leprae",
    "reference": "NC_002677.1",
    "targets": {
        "gyrA_DRDR": {
            "start": 7550,
            "end": 7650,
            "gene_start": 7318,
            "gene_end": 11067,
            "strand": "+",
            "drug": "Fluoroquinolones",
            "known_mutations": {
                "7589": {
                    "ref": "G", "alt": "T",
                    "aa": "p.D91Y",
                    "drug": "Ofloxacin",
                    "literature": "WHO 2023"
                }
            }
        }
    }
}
```

A pre-built panel for **M. leprae** (gyrA, folP1, rpoB) ships in `configs/leprae.json`.

---

## Limitations (Honest Assessment)

This is an **educational implementation**. Key differences from production DeepVariant:

| Aspect | minDeepVariant | Google DeepVariant |
|--------|---------------|-------------------|
| Training data | Synthetic pileups | Real labeled variants (GIAB truth sets) |
| Architecture | 2-layer CNN | Inception-v3 (deeper) |
| Input channels | 1 (grayscale) | 6+ (base, quality, strand, mapping quality...) |
| Variant types | SNPs + simple deletions | SNPs, indels, structural variants |
| Base quality | Not modeled | Encoded as a separate image channel |

These limitations are intentional — the goal is clarity, not competition.

---

## Project Structure

```
minDeepVariant/
├── src/mindeepvariant/
│   ├── model.py        # CNN architecture + pixel encoding
│   ├── train.py        # Synthetic data engine + training loop
│   ├── scanner.py      # BAM pileup extraction + heuristic filter
│   ├── annotator.py    # Codon translation + Known/Novel tiers
│   └── cli.py          # Command-line interface
├── configs/
│   └── leprae.json     # M. leprae AMR panel
├── tests/
│   └── test_core.py    # Unit tests
├── Dockerfile
└── setup.py
```

---

## Citation

If you use minDeepVariant in your work:

```
@software{mindeepvariant,
    title = {minDeepVariant: A minimal deep learning variant caller for genomics education},
    author = {Manjyot},
    year = {2025},
    url = {https://github.com/hemant-goyal//minDeepVariant}
}
```

---

## Acknowledgments

- **Google DeepVariant** (Poplin et al., 2018) for the pileup-as-image paradigm
- **Andrej Karpathy's minGPT** for the "minimal reimplementation" philosophy
- **Chris Hayduk's minAlphaFold2** for demonstrating the approach in structural biology

---

## License

MIT
