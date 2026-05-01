# mindv — A Minimal Deep Learning Variant Caller

[![Install](https://img.shields.io/badge/install-pip%20install%20git%2Bgithub.com%2Fhemant--goyal%2Fmindv-blue)](https://github.com/hemant-goyal/mindv)

**mindv** (minDeepVariant) is a from-scratch, educational implementation of image-based variant calling, inspired by [Google's DeepVariant](https://github.com/google/deepvariant) and built in the spirit of Karpathy's [minGPT](https://github.com/karpathy/minGPT): minimal code, maximal clarity, real results.

It treats variant calling as **image classification** — converting pileups of aligned sequencing reads into grayscale tensors and classifying them with a small convolutional neural network (CNN). The entire codebase is ~1,250 lines of Python across 5 modules, each small enough to read in one sitting.

mindv was developed for antimicrobial resistance (AMR) profiling in a *Mycobacterium* species, but its organism-agnostic JSON panel system means it can be pointed at any haploid genome with a known set of resistance-associated loci.

**NOTE:** Any help would be greatly appreciated in terms of testing and also in generating AMR panels for WHO priority pathogens. 

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Training](#training)
  - [Scanning](#scanning)
  - [Filter Presets](#filter-presets)
  - [Custom Filtering](#custom-filtering)
- [AMR Panel Configuration](#amr-panel-configuration)
- [Output Files](#output-files)
- [Understanding the Filters](#understanding-the-filters)
  - [Allele Frequency (AF)](#allele-frequency-af)
  - [Read Depth (DP) and Alternate Depth (AD)](#read-depth-dp-and-alternate-depth-ad)
  - [Base Quality (BQ) and Mapping Quality (MQ)](#base-quality-bq-and-mapping-quality-mq)
  - [Model Confidence](#model-confidence)
- [Validation Results](#validation-results)
- [Testing Your Installation](#testing-your-installation)
- [Reproducing Results with Public Data](#reproducing-results-with-public-data)
- [Architecture](#architecture)
- [Limitations and Honest Comparison with DeepVariant](#limitations-and-honest-comparison-with-deepvariant)
- [Future Directions](#future-directions)
- [Citation](#citation)
- [License](#license)

---

## Why This Exists

Google's DeepVariant demonstrated that variant calling can be reframed as image classification. Instead of hand-tuning statistical models for every sequencing platform and organism, you render aligned reads as an image and let a neural network learn what a real variant "looks like."

But DeepVariant itself is a large, production-grade system written in hundreds of thousands of lines of code, complex build systems, GPU infrastructure requirements, and training pipelines that assume access to large labeled human genome datasets. If you're a bioinformatics student, specifically a microbiologist working on a non-model organism, or a researcher who wants to understand *how* the method works rather than just *run* it, DeepVariant is opaque.

**mindv fills that gap.** It implements DeepVariant's core idea in code simple enough to understand completely, trains on synthetic data (no labeled genome datasets required), and produces real, validated results on clinical sequencing data. It is:

- **Educational**: every design decision is documented, every module fits in your head
- **Practical**: successfully profiled AMR mutations across tens of clinical *M. leprae* cohort with Sanger-validated results
- **Extensible**: swap in a new JSON panel to profile any organism, any set of loci
- **Honest**: see [Limitations](#limitations-and-honest-comparison-with-deepvariant) for what it can and cannot do compared to production tools

---

## How It Works

mindv follows a three-stage pipeline: **Train → Scan → Annotate**.

### Stage 1: Training on Synthetic Data

The model has never seen real sequencing data during training. Instead, `train.py` generates thousands of synthetic pileup images that simulate the four variant classes:

| Class | Label | What the synthetic image looks like |
|-------|-------|-------------------------------------|
| 0 | Hom-Ref (Wild-type) | All read rows match the reference row |
| 1 | Het-SNP | ~50% of reads show an alternate base at center |
| 2 | Hom-Alt-SNP | ~100% of reads show an alternate base at center |
| 3 | Deletion | Reads show gap characters at and around center |

Each synthetic image incorporates biological realism with variable read depths, background sequencing noise, strand-specific base distributions, and realistic allele frequency ranges. The CNN learns to distinguish these patterns without needing any organism-specific training data.

### Stage 2: Scanning Real BAM Files

Given a BAM file and a reference FASTA, the scanner (`scanner.py`) walks through each position in the target gene panel. At every position, it performs a fast heuristic check: are there enough alternate-allele reads (AD) at sufficient depth (DP) to warrant CNN evaluation? Most positions are wild-type and get skipped instantly. The few that pass the heuristic threshold get converted into a real pileup tensor (a grayscale image of the reference row + aligned reads) and fed to the trained CNN for classification.

Quality filtering happens at this stage, at the pysam pileup level. Reads with mapping quality below `--min-mq` and individual bases with base quality below `--min-bq` are excluded by pysam *before* any allele counting occurs. This means low-quality evidence is invisible to both the heuristic counter and the CNN — they always see the same filtered read stack, which prevents inconsistencies between the triggering logic and the classification logic.

The scanner also implements **GATK-aware BAM selection**: when a sample directory contains both a raw BAM (`sample.bam`) and a GATK MarkDuplicates output (`sample_marked_duplicates.bam`), the deduplicated file is automatically preferred. This prevents false-positive allele frequencies caused by PCR amplification bias.

### Stage 3: Annotation and Tiered Classification

Every CNN-positive call is annotated by `annotator.py` with:

- **Protein consequence**: codon translation using the gene's reading frame, correctly handling both forward and reverse strands
- **Mutation type**: Missense, Synonymous, Nonsense, or Frameshift
- **Tier classification**: KNOWN (matches a characterized resistance mutation in the panel JSON) or NOVEL (new candidate requiring investigation)
- **Drug association**: for KNOWN-tier variants, which drug the mutation confers resistance to
- **Literature reference**: the source publication for KNOWN-tier variants, which is integrated into the AMR panel.

Post-CNN filtering then applies the user's chosen thresholds (AF, DP, AD, confidence) to produce the final reported variant set. Filtered variants are logged in per-sample text reports with the specific reason(s) they failed, so nothing is silently discarded.

---

mindv is optimized for **haploid clonal organisms** — bacteria, mycobacteria, and other microbes where the genome is a single copy and the infecting population is typically clonal. This biological context profoundly affects how variant calling works and how results should be interpreted.

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch ≥ 1.9
- pysam ≥ 0.19
- Biopython ≥ 1.79
- matplotlib ≥ 3.4

### Install directly from GitHub

```bash
pip install git+https://github.com/hemant-goyal/mindv.git
```
After installation, the `mindv` command is available system-wide:

```bash
mindv --help
```
# Developer install (editable)

```bash
git clone https://github.com/hemant-goyal/mindv.git
cd mindv
pip install -e .
```

### Docker (optional)

```bash
docker build -t mindv .
docker run -v /path/to/data:/data mindv scan \
    --bam_dir /data/bams --ref /data/ref.fna \
    --panel /data/leprae.json --outdir /data/results
```

---

## Quick Start

### 1. Train the model

```bash
mindv train --output mindv_weights.pth --epochs 30
```

This generates synthetic pileup images and trains the CNN. Takes ~5 minutes on CPU. The output is a `.pth` weights file used by the scanner.

### 2. Scan a sample cohort

```bash
mindv scan \
    --bam_dir /path/to/aligned/ \
    --ref /path/to/reference.fna \
    --panel configs/leprae.json \
    --outdir results/
```

This recursively finds BAM files, prioritizes GATK-deduplicated files (`*_marked_duplicates.bam`) when available, scans each sample against the gene panel, and produces per-sample reports plus a master CSV.

### 3. Review results

```bash
# Master summary across all samples
cat results/Master_Clinical_Summary.csv

# Per-sample detailed report (includes filtered variants with reasons)
cat results/sample_01_report.txt

# Per-sample tensor visualization PDFs
open results/sample_01_plots.pdf
```

---

## Usage Guide

### Training

```bash
mindv train [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output` | `mindv_weights.pth` | Output path for trained model weights |
| `--epochs` | 30 | Number of training epochs |
| `--samples` | 1000 | Synthetic training samples generated per epoch |
| `--lr` | 0.0005 | Learning rate (Adam optimizer) |
| `--window` | 21 | Genomic window width in base pairs |
| `--depth` | 30 | Maximum read rows in pileup tensor |

The window and depth parameters define the CNN input dimensions. If you change them here, you must pass the same values to `scan`. The defaults (21×31 tensor: 30 read rows + 1 reference row) work well for targeted AMR profiling where you're looking at point mutations in known genes.

### Scanning

```bash
mindv scan [OPTIONS]
```

**Required arguments:**

| Option | Description |
|--------|-------------|
| `--bam_dir` | Directory containing BAM files (searched recursively) |
| `--ref` | Path to reference FASTA (must be indexed: `.fai` file present) |
| `--panel` | Path to AMR panel JSON configuration |
| `--outdir` | Output directory for reports, CSVs, and PDFs |

**Optional arguments:**

| Option | Default | Description |
|--------|---------|-------------|
| `--weights` | `mindv_weights.pth` | Path to trained model weights |
| `--contig` | *(from panel JSON)* | Override contig/chromosome name |
| `--window` | 21 | Must match the value used during training |
| `--depth` | 30 | Must match the value used during training |
| `--preset` | *(default values)* | Filter preset: `raw`, `sensitive`, `default`, `clinical` |

**BAM file selection logic:**

The scanner searches `--bam_dir` recursively for `.bam` files, grouped by parent directory (one folder per sample). If a folder contains both a raw BAM (`sample.bam`) and a GATK-deduplicated BAM (`sample_marked_duplicates.bam`), the deduplicated file is used. This prevents false-positive allele frequencies caused by PCR amplification bias. The BAM index can be in any standard format (`.bam.bai`, `.bai`, or `.csi`) — pysam detects all three automatically.

### Filter Presets

Presets configure all six filter parameters at once for common use cases. Individual `--min-*` flags override preset values, so you can use a preset as a starting point and adjust one parameter.

```bash
# Balanced defaults — recommended for most research use
mindv scan ... --preset default

# Strict clinical reporting — only near-fixed, high-confidence calls
mindv scan ... --preset clinical

# Heteroresistance / mixed-population mode — catches low-AF subclones
mindv scan ... --preset sensitive

# Unfiltered — shows everything the CNN flagged (for debugging/validation)
mindv scan ... --preset raw
```

| Parameter | `raw` | `sensitive` | `default` | `clinical` |
|-----------|-------|-------------|-----------|------------|
| `--min-af` | 0.0 | 0.02 | **0.10** | 0.70 |
| `--min-dp` | 15 | 10 | **10** | 20 |
| `--min-ad` | 3 | 3 | **4** | 10 |
| `--min-bq` | 20 | 20 | **20** | 20 |
| `--min-mq` | 20 | 20 | **20** | 30 |
| `--min-confidence` | 0.0 | 0.60 | **0.75** | 0.90 |

**When to use which preset:**

- **`default`**: General-purpose research. Catches all near-fixed and moderate-AF variants while rejecting the Illumina noise floor. Start here.
- **`clinical`**: When the output goes into a clinical report and you only want high-certainty, actionable calls. Sacrifices sensitivity for specificity.
- **`sensitive`**: When you're specifically hunting for heteroresistance, minor resistant subpopulations, or mixed infections. Expect more candidates that require manual review — check the AF column and tensor PDFs.
- **`raw`**: Debugging and method validation only. Shows every CNN-positive candidate with standard read-quality filtering but no post-CNN AF/confidence filtering. Use this to see "what the model saw" before filtering. **Not for biological interpretation.**

### Custom Filtering

Override individual parameters on top of any preset (or the implicit default):

```bash
# Default preset but with a lower AF threshold to catch one specific sample
mindv scan ... --preset default --min-af 0.05

# Clinical strictness but accepting lower model confidence
mindv scan ... --preset clinical --min-confidence 0.80

# Fully manual — set every parameter explicitly
mindv scan ... \
    --min-af 0.15 --min-dp 20 --min-ad 5 \
    --min-bq 25 --min-mq 25 --min-confidence 0.85
```

---

## AMR Panel Configuration

mindv uses JSON files to define which genomic regions to scan and which mutations are known resistance markers. This makes the tool organism-agnostic — swap the JSON, scan a different pathogen.

### Panel JSON structure

```json
{
    "organism": "Mycobacterium leprae",
    "reference": "NC_002677.1",
    "targets": {
        "gyrA_DRDR": {
            "start": 7540,
            "end": 7650,
            "gene_start": 7318,
            "gene_end": 11067,
            "strand": "+",
            "known_mutations": {
                "7589_T": {
                    "ref": "C", "alt": "T",
                    "aa_change": "p.A91V",
                    "drug": "Ofloxacin",
                    "literature": "WHO Leprosy AMR Surveillance, 2023"
                },
                "7600_T": {
                    "ref": "G", "alt": "T",
                    "aa_change": "p.D95Y",
                    "drug": "Ofloxacin",
                    "literature": "WHO 2023"
                }
            }
        },
        "rpoB_RRDR": { "..." : "..." },
        "folP1_DRDR": { "..." : "..." }
    }
}
```

**Field descriptions:**

| Field | Description |
|-------|-------------|
| `organism` | Species name (used in report headers) |
| `reference` | Default contig name (overridable with `--contig`) |
| `targets` | Dictionary of gene regions to scan |
| `start` / `end` | 1-based coordinates of the drug-resistance-determining region (DRDR) to scan |
| `gene_start` / `gene_end` | Full gene coordinates (used for codon translation) |
| `strand` | `"+"` or `"-"` (determines codon reading frame and reverse-complement logic) |
| `known_mutations` | Position-keyed dictionary of characterized resistance mutations |

**Note on known_mutations Formatting:** 
> The keys in the known_mutations dictionary must use the POSITION_ALT format (e.g., "7589_T" or "7600_A"). Because many resistance-determining genes can have multiple different resistance mutations at the exact same genomic position (e.g., D95N and D95Y both occurring at position 7600), appending the alternate base to the key prevents JSON parsers from silently overwriting duplicate positions.

### Creating panels for other organisms

To profile a different pathogen, create a new JSON file following the structure above. You need:

1. The reference genome accession and contig name
2. Coordinates of the resistance-associated gene regions (from literature or databases like CARD, WHO AMR catalogues)
3. Known mutations with their drug associations

For example, an *M. tuberculosis* panel might target `rpoB` (rifampicin), `katG` (isoniazid), `gyrA`/`gyrB` (fluoroquinolones), and `embB` (ethambutol). The model does not need retraining — the CNN learned generic pileup patterns (match vs. mismatch vs. deletion) that transfer across organisms.

---

## Output Files

After scanning, the output directory contains:

```
results/
├── Master_Clinical_Summary.csv      # Cohort-wide variant table
├── sample_01_report.txt            # Detailed per-sample text report
├── sample_01_plots.pdf             # Tensor visualization for each variant
├── sample_02_report.txt
├── sample_02_plots.pdf
└── ...
```

### Master CSV columns

| Column | Description |
|--------|-------------|
| `sample` | Sample identifier (derived from BAM filename) |
| `GENE` | Target gene region name |
| `CONTIG` | Reference contig |
| `POS` | 1-based genomic position |
| `REF` / `ALT` | Reference and alternate allele |
| `TOTAL_DP` | Total read depth at this position |
| `AF` | Allele frequency (alternate reads / total reads) |
| `CLASS` | CNN classification: Hom-Ref, Het-SNP, Hom-Alt-SNP, or Deletion |
| `CONFIDENCE` | CNN softmax probability for the called class |
| `AA_CHANGE` | Protein consequence (e.g., `p.A91V`) |
| `MUT_TYPE` | Missense, Synonymous, Nonsense, or Frameshift |
| `IMPACT` | HIGH (Missense/Nonsense/Frameshift) or LOW (Synonymous) |
| `TIER` | KNOWN (matches panel) or NOVEL (new candidate) |
| `KNOWN_DRUG` | Associated drug (KNOWN tier only) |
| `LITERATURE` | Source reference (KNOWN tier only) |

### Per-sample text reports

Each text report includes a header with the applied filter parameters, a per-gene breakdown with one row per evaluated position, and a summary section listing reported vs. filtered counts. **Filtered variants are not silently discarded** — they appear in the text report with a `FILT` tag and the specific reason(s) they failed:

```
7565       G    A    1635   0.0018 Hom-Alt-SNP      0.94   p.G83D         Missense     FILT   [filtered: AF=0.0018<0.1]
```

This transparency is deliberate. When tuning thresholds, you can grep the text reports for `FILT` to see what you're missing, and adjust accordingly.

### Tensor visualization PDFs

For each reported variant, a PDF page shows the raw pileup tensor (grayscale image) that the CNN classified. The reference sequence is Row 0 (separated by a red horizontal line), sample reads are below. A cyan vertical dashed line marks the center position. The title shows depth, allele frequency, CNN classification, confidence, protein consequence, tier, and drug association.

These plots serve as visual sanity checks — you can confirm that the CNN is seeing a clean column of alternate bases (real variant) rather than scattered noise (false positive).

---

## Understanding the Filters

This section explains the biological rationale behind each filter parameter. Understanding these is essential for choosing the right preset and interpreting results correctly.

### Allele Frequency (AF)

**What it is:** The fraction of reads at a position that carry the alternate (non-reference) base. `AF = alternate_reads / total_reads`.

**Why it matters:** AF encodes ploidy and population structure. For a haploid clonal organism, a real variant carried by the entire bacterial population should sit at AF ≈ 1.0. In practice, mapping artifacts and stray reads bring this to 0.90–0.99, but the signal is unambiguous.

The critical question is what happens *below* AF = 0.10. Illumina sequencing has an inherent per-base error rate of roughly 0.1–1% (Q20–Q30). At high coverage (1000x+), this error rate produces 1–10 reads at every position showing the "wrong" base purely by chance. Therefore, these stochastic errors get reported as variants like false-positive pattern that produces hundreds of noise calls in deep-sequenced samples.

**The AF threshold is the single most important filter parameter.** The right value depends on what you're trying to detect:

| Biological scenario | Expected AF | Recommended `--min-af` |
|---------------------|-------------|------------------------|
| Fixed clonal variant | 0.90 – 1.00 | 0.10 or higher |
| Mixed infection / heteroresistance | 0.05 – 0.50 | 0.02 – 0.05 |
| Minor resistant subpopulation | 0.01 – 0.05 | 0.01 (with manual review) |
| Sequencing noise | 0.001 – 0.01 | — (reject) |

For routine AMR profiling of clonal isolates, `--min-af 0.10` (the default) cleanly separates real variants from noise. For clinical reporting where you only want actionable, near-fixed mutations, `--min-af 0.70` eliminates ambiguity. For heteroresistance research, `--min-af 0.02` lets through minor subclones, but every call below AF = 0.10 should be treated as a candidate requiring orthogonal confirmation (Sanger sequencing, targeted deep sequencing, or repeat extraction).

### Read Depth (DP) and Alternate Depth (AD)

**DP** is the total number of reads covering a position. **AD** is the number of those reads that carry the alternate allele.

**Why they matter:** Statistical power. At DP = 5, even AF = 0.60 means only 3 alternate reads — you can't confidently distinguish that from a noisy wild-type. At DP = 500, AF = 0.01 means 5 alternate reads — the fraction is tiny but the absolute evidence is measurable (though likely noise).

DP and AD work as complementary guards:

- **DP catches low-coverage positions** where no AF claim is reliable. A position with 3 total reads is not informative regardless of what those 3 reads show.
- **AD catches positions where AF looks reasonable but absolute evidence is thin.** At DP = 30, AF = 0.10 means only 3 alt reads — enough to trigger the heuristic but not enough to trust.

The defaults (DP ≥ 10, AD ≥ 4) are appropriate for typical Illumina WGS at 50x+ coverage. For very deep sequencing (500x+), AD becomes the more important filter.

### Base Quality (BQ) and Mapping Quality (MQ)

These are applied at the **pysam pileup level**, before any allele counting or CNN evaluation. This is a critical design decision as filtering at the source ensures that the heuristic counter, the AF calculation, the tensor extractor, and the CNN all see exactly the same set of high-quality evidence.

**Base quality (BQ)** is the Phred-scaled probability that a specific base call is wrong. BQ = 20 means a 1-in-100 chance of error; BQ = 30 means 1-in-1000. Bases below `--min-bq` are excluded from the pileup entirely. The default (BQ ≥ 20) is the standard threshold used by virtually all variant callers.

**Mapping quality (MQ)** is the aligner's confidence that a read belongs at this position in the genome. Reads with MQ = 0 could equally well map to multiple locations (common in repetitive regions, PE/PPE genes, IS elements). Including them creates false variants from misaligned reads. The default (MQ ≥ 20) excludes ambiguously-mapped reads. The clinical preset raises this to MQ ≥ 30 for extra stringency.

**A practical note:** if you're working with a genome that has extensive repetitive regions (e.g.,*M. tuberculosis* PE/PPE family, which comprises ~10% of the genome), consider raising `--min-mq` to 30 even in default mode for gene regions near repeats.

### Model Confidence

The CNN outputs a softmax probability distribution across the four classes (Hom-Ref, Het-SNP, Hom-Alt-SNP, Deletion). The **confidence** score is the probability assigned to the winning class. A confidence of 0.95 means the model is 95% sure this pileup belongs to the predicted class; 0.55 means it's barely more confident than a coin flip.

The default threshold (≥ 0.75) requires the model to be substantially confident, not merely above 50%. This catches cases where the pileup is ambiguous for example, a low-AF variant that looks somewhat like a Het-SNP but also somewhat like a Hom-Ref. The model might call it Het-SNP with confidence 0.58, which correctly reflects the ambiguity but shouldn't be reported as a confident variant call.

The clinical preset raises this to ≥ 0.90. The sensitive preset lowers it to ≥ 0.60, accepting more uncertain calls in exchange for not missing potential minor variants.

---

## Validation Results

mindv was validated on a cohort of *Mycobacterium* species whole-genome sequencing samples, with Sanger sequencing as the orthogonal gold standard.

### Sanger-validated resistance mutations

5 Samples were confirmed by Sanger sequencing to carry the gyrA resistance mutation. One additional low-frequency call (sample_05, AF = 0.009) fell below Sanger's limit of detection (~15–20% AF) and could not be confirmed or excluded.

### Performance across filter presets

The same sample cohort was scanned under all four presets to characterize the sensitivity-specificity tradeoff:

| Preset | Total reported | Sanger-confirmed detected (of 4) | Sanger-discordant calls | Notes |
|--------|---------------|----------------------------------|------------------------|-------|
| `raw` | 58 | 4/4 | 1 (below Sanger LOD) | Full noise floor visible |
| `sensitive` | 8 | 3/4 | 0 | Includes novel candidates at AF 0.03–0.06 |
| `default` | 5 | 4/4 | 0 | **Recommended**: all confirmed hits, no false positives |
| `clinical` | 2 | 2/4 | 0 | Strictest: only near-fixed, highest-confidence calls |

**Key findings:**

1. **Default preset achieved 4/4 sensitivity with 0 false positives** against Sanger-confirmed resistance mutations. This is the recommended preset for routine research use.

2. The clinical preset's reduced sensitivity (2/4) reflects a deliberate design tradeoff: Sample_01's call had a model confidence of 0.77 (below the clinical threshold of 0.90), and Sample_03's mixed-population AF of 0.40 fell below the clinical AF threshold of 0.70. Both are real variants excluded by design confirm that clinical mode prioritizes certainty over completeness.

3. Sample_05's AF = 0.009 call at the known gyrA codon illustrates a fundamental detection limit: at AF < 0.02, orthogonal Sanger confirmation is impossible, so mindv's default preset conservatively excludes such calls. The sensitive preset retains them for researcher guided review.

4. Sample_03 exhibited **compound resistance** — both known gyrA (AF = 0.40) and other position (AF = 0.16) in gyrA — detectable by mindv's sensitive preset. This mixed-population, multi-mutation profile would be missed by Sanger screens targeting only a single codon, demonstrating the value of deep sequencing + sensitive variant calling for heteroresistance surveillance.

5. The progression from raw (58 variants) to default (5 variants) demonstrates that >90% of unfiltered CNN-positive calls are sequencing noise at the Illumina error floor (AF < 0.01), concentrated in high-coverage samples. Quality-aware filtering is essential for interpretable results.

---

## Testing Your Installation

### Synthetic model test

After installation, verify the model and inference pipeline work correctly:

```bash
mindv test
```

This command generates 100 synthetic pileup tensors (25 per class), runs them through the CNN, and prints a per-class confusion matrix. Expected output on a correctly trained model:

```
=== mindv Synthetic Test ===
Generating 100 synthetic pileups (25 per class)...
Running inference...

Confusion Matrix:
              Predicted
Actual     Ref   Het   Hom   Del
Ref         24     1     0     0
Het          0    23     2     0
Hom          0     1    24     0
Del          0     0     0    25

Overall accuracy: 96/100 (96.0%)
✓ Test passed (accuracy >= 85%)
```

If accuracy drops below 85%, retrain the model (`mindeepvariant train`). The synthetic test does not require any BAM files, reference genomes, or external data.

### Manual verification with real data

To verify the full end-to-end pipeline (pysam integration, panel loading, annotation, filtering), you need a BAM file and reference genome. See [Reproducing Results with Public Data](#reproducing-results-with-public-data) below.

---

## Reproducing Results with Public Data

The validation cohort described above uses samples that cannot be redistributed. To independently test mindv on real data, you can use publicly available whole-genome sequencing datasets from NCBI's Sequence Read Archive (SRA).

### Step 1: Download public *M. leprae* WGS data

Several published studies have deposited *M. leprae* WGS data with known resistance profiles. For example:

```bash
# Install SRA Toolkit if not already present
https://github.com/ncbi/sra-tools

#Install my own TurboSRA
https://github.com/hemant-goyal/TurboSRA

# Download a sample with known mutations (example accession — 
# check the associated publication for resistance genotype)
fastq-dump --split-files SRR_ACCESSION
```

Suitable datasets can be found by searching NCBI SRA for `"Mycobacterium leprae" AND "whole genome sequencing"` and filtering for Illumina paired-end data with associated AMR metadata.

### Step 2: Align to the reference genome

```bash
# Download M. leprae reference
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/195/855/GCF_000195855.1_ASM19585v1/GCF_000195855.1_ASM19585v1_genomic.fna.gz
gunzip GCF_000195855.1_ASM19585v1_genomic.fna.gz

REF=GCF_000195855.1_ASM19585v1_genomic.fna

# Index reference
bwa index $REF
samtools faidx $REF

# Align
bwa mem -t 4 $REF sample_1.fastq sample_2.fastq | \
    samtools sort -o sample.bam
samtools index sample.bam

# (Optional but recommended) Mark duplicates with GATK
gatk MarkDuplicates \
    -I sample.bam \
    -O sample_marked_duplicates.bam \
    -M sample_duplicate_metrics.txt
samtools index sample_marked_duplicates.bam
```

### Step 3: Run mindv

```bash
mindv scan \
    --bam_dir /path/to/aligned/ \
    --ref $REF \
    --panel configs/leprae.json \
    --outdir results/
```

### Step 4: Cross-reference with published results

Compare the mindv output against the resistance genotype reported in the source publication. For well-characterized public isolates, you should see concordant results at the default preset.

---

## Architecture

```
mindv/
├── configs/
│   └── *.json              # * AMR panel
├── src/
│   └── mindeepvariant/
│       ├── __init__.py
│       ├── model.py             # CNN definition, base-to-pixel encoding
│       ├── train.py             # Synthetic data engine + training loop
│       ├── scanner.py           # BAM pileup extraction + heuristic filter
│       ├── annotator.py         # Protein annotation + tiered classification
│       └── cli.py               # Command-line interface + filter presets
├── tests/                       # Synthetic test suite
├── tutorial/                    # Educational notebook walkthrough
├── setup.py                     # pip install -e .
├── Dockerfile                   # Containerized deployment
├── CHANGELOG.md
└── README.md
```

### Module-by-module summary

**`model.py`** (~100 lines) — Defines `MinDeepVariantCNN`, a two-layer convolutional neural network with dynamic input dimensions. Also defines the `BASE_TO_PIXEL` encoding and `CLASS_NAMES`. Includes `predict_with_confidence()` which returns the predicted class, its softmax confidence, and the full probability distribution.

**`train.py`** (~250 lines) — The synthetic data engine. Generates realistic pileup images for all four variant classes, incorporating variable depth, background noise, strand-specific base distributions, and realistic AF ranges. Runs the training loop with Adam optimizer and cross-entropy loss. Outputs a `.pth` weights file.

**`scanner.py`** (~250 lines) — Opens BAM/FASTA files via a context manager (preventing file descriptor leaks), validates inputs (checks BAM index, contig existence), extracts pileup tensors using **actual genomic coordinates** (fixing a column-drift bug in the original implementation), and runs the heuristic + CNN scan loop. Quality filtering (BQ, MQ) is applied at the pysam pileup level, before any allele counting.

**`annotator.py`** (~250 lines) — Loads the JSON panel, performs strand-aware codon translation to determine protein consequence, classifies mutations as Known vs. Novel based on panel entries, and attaches drug/literature metadata.

**`cli.py`** (~350 lines) — The glue. Parses arguments, resolves filter presets, orchestrates the train/scan pipeline, writes per-sample text reports and PDFs, applies the post-CNN filter cascade (AF, DP, AD, confidence), and produces the master CSV.

---

## Limitations and Honest Comparison with DeepVariant

mindv is an educational tool that produces real results. It is **not** a replacement for production-grade variant callers. Here is an honest accounting of where it falls short:

### What mindv does well

- **Teaches the core insight**: if you understand mindv, you understand why DeepVariant works
- **Zero labeled data requirement**: synthetic training means you can apply it to any organism on day one, with no labeled training set needed
- **Fast on targeted panels**: scanning hundreds of samples across 3~5 genes takes ~60 seconds total
- **Transparent**: every call is accompanied by a tensor plot, filter reasons, and confidence scores

### What DeepVariant does that mindv does not

| Capability | DeepVariant | mindv |
|-----------|-------------|-------|
| Training data | Millions of labeled human variants (GIAB) | Synthetic pileups only |
| Input channels | 6-channel tensor (base, quality, strand, mapping quality, etc.) | 1-channel grayscale (base identity only) |
| Model architecture | Inception-v3 (~25M parameters) | 2-layer CNN (~15K parameters) |
| Variant types | SNPs, insertions, deletions, MNPs | SNPs + simple deletions only |
| Indel handling | Realignment-aware representation | Naive gap encoding |
| Whole-genome calling | Yes (parallelized, GPU-accelerated) | No (targeted panels only) |
| Strand bias detection | Built into tensor channels | Not implemented (planned for v2) |
| Population filtering | Multi-sample joint calling available | Single-sample only |
| Regulatory approval | Used in clinical pipelines | Research and education only |

### Known limitations to be aware of

1. **Synthetic training gap**: the CNN has never seen real sequencing artifacts (adapter contamination, GC-bias coverage drops, polymerase slippage in homopolymers). It may miscall positions where these artifacts mimic variant patterns.

2. **No indel realignment**: complex indels near the edge of the pileup window may be misrepresented in the tensor. Simple deletions work; multi-base insertions are unreliable.

3. **No strand bias filter**: this is the most impactful missing feature (planned for v2). Until then, check tensor plots manually for variants supported by only one strand direction.

4. **Single-channel input**: by encoding only base identity (not base quality, mapping quality, or strand), the CNN has less information than DeepVariant's 6-channel tensor. The BQ/MQ filtering in scanner.py partially compensates by excluding low-quality evidence before the tensor is built, but per-read quality information is lost.

5. **AF floor is post-hoc**: the AF filter is applied after CNN classification, not during. The CNN itself has no notion of allele frequency — it classifies the pileup image regardless. This means the model can confidently call a "Het-SNP" at AF = 0.003, which is biologically impossible in a haploid organism. The filter catches this, but a more principled approach would incorporate AF into the tensor or model architecture.

---

## Future Directions

Features planned for v2 and beyond, roughly in priority order:

- **Strand bias filter (`--max-strand-bias`)**: Fisher's exact test on the 2×2 forward/reverse × ref/alt table. The single most impactful quality filter not yet implemented.
- **Variant proximity filter**: flag or drop variant clusters (multiple calls within N bases), which often indicate alignment artifacts in repetitive regions.
- **Multi-channel tensors**: add base quality, mapping quality, and strand direction as separate channels in the pileup image, approaching DeepVariant's 6-channel representation.
- **Fine-tuning on real data**: optional transfer learning from labeled variant sets (e.g., GIAB for human, or curated AMR databases for pathogens).
- **Additional organism panels**: *M. tuberculosis* (TB), *S. aureus* (MRSA), *N. gonorrhoeae* (gonorrhoea) — community contributions welcome.
- **CARD database integration**: automated panel generation from the Comprehensive Antibiotic Resistance Database.
- **VCF output**: standard VCF format alongside the current CSV, for compatibility with downstream tools like SnpEff, SnpSift, and IGV.
- **Simulated BAM test suite**: `wgsim`/`ART`-generated test BAMs with known spiked-in mutations for end-to-end pipeline testing without real sample data.

---

## Citation

If you use mindv in your research, please cite:

> *mindv: A Minimal PyTorch Implementation of Image-Based Variant Calling for Genomics Education and Rapid AMR Profiling.* [Hemant Goyal].(https://github.com/hemant-goyal/mindv) (Manuscript in preparation)

---

## License

[MIT License](LICENSE)

---

## Acknowledgements

- [Google DeepVariant](https://github.com/google/deepvariant) — the original insight that variant calling can be image classification
- [Andrej Karpathy's minGPT](https://github.com/karpathy/minGPT) — the philosophy that minimal implementations can be maximally educational.
