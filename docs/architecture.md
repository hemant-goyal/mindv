# Architecture Guide

## System Overview

minDeepVariant consists of four modules that form a linear pipeline:

```
[BAM + FASTA]  →  scanner.py  →  model.py  →  annotator.py  →  [Reports]
                   (extract)      (classify)    (translate)
```

Each module is self-contained and can be used independently via the Python API.

---

## Module 1: model.py — The CNN

### Input Encoding

DNA bases are mapped to scalar pixel intensities:

| Base | Pixel Value | Rationale |
|------|------------|-----------|
| N (unknown) | 0.00 | No information → black |
| - (deletion) | 0.10 | Gap → near-black, distinct from N |
| A | 0.25 | |
| C | 0.50 | Evenly spaced to maximize |
| G | 0.75 | contrast between all bases |
| T | 1.00 | |

The specific values are arbitrary. What matters is that each base has a unique, consistent intensity so the CNN can learn base-specific patterns.

### Tensor Shape

```
Rows:    depth + 1 = 31  (row 0 = reference, rows 1-30 = reads)
Columns: window    = 21  (10bp flanking each side of the target position)
Channels: 1             (grayscale)

Final shape: (batch, 1, 31, 21)
```

### Network Architecture

```
Layer           | Input Shape      | Output Shape     | Parameters
----------------|------------------|------------------|----------
Conv2d(1→16)    | (1, 1, 31, 21)  | (1, 16, 31, 21) | 160
ReLU            | —                | —                | 0
MaxPool2d(2,2)  | (1, 16, 31, 21) | (1, 16, 15, 10) | 0
Conv2d(16→32)   | (1, 16, 15, 10) | (1, 32, 15, 10) | 4,640
ReLU            | —                | —                | 0
MaxPool2d(2,2)  | (1, 32, 15, 10) | (1, 32, 7, 5)   | 0
Flatten         | (1, 32, 7, 5)   | (1, 1120)        | 0
Linear(→64)     | (1, 1120)        | (1, 64)          | 71,744
ReLU            | —                | —                | 0
Dropout(0.3)    | —                | —                | 0
Linear(→4)      | (1, 64)          | (1, 4)           | 260
```

**Total: ~76,804 parameters** (vs. ~23M for Google DeepVariant's Inception-v3).

### Dynamic Dimension Computation

Unlike the original notebook (which hardcoded `32 * 7 * 5`), the FC layer size is computed at init by passing a dummy tensor through the conv layers:

```python
with torch.no_grad():
    dummy = torch.zeros(1, 1, img_h, img_w)
    flat_size = self.features(dummy).numel()
```

This means changing `window` or `depth` never silently breaks the model.

### Confidence Scoring

`predict_with_confidence()` returns the full softmax distribution, not just the argmax. This enables downstream filtering:

- **High confidence (>0.9):** Strong CNN evidence for this variant class
- **Medium confidence (0.6–0.9):** Plausible but should be validated
- **Low confidence (<0.6):** Ambiguous — may be noise

---

## Module 2: train.py — Synthetic Data Engine

### Class Definitions

| Class | Label | Simulation Rule |
|-------|-------|----------------|
| Hom-Ref | 0 | All reads match reference; ~2% random noise at any position |
| Het-SNP | 1 | 30-70% of reads carry a random alt base at center |
| Hom-Alt | 2 | ~95% of reads carry the alt at center |
| Deletion | 3 | ~90% of reads show gap pixels; deletion spans 1-3 bases |

### Key Improvements Over the Original

1. **Noise in Hom-Ref class:** 2% random base changes per position simulate sequencing errors, teaching the model to tolerate noise without calling false variants.

2. **Variable allele frequency for Het-SNP:** Instead of a fixed 50% coin flip, AF is sampled uniformly from [0.30, 0.70]. This teaches the model that heterozygous variants don't always show exactly 50/50.

3. **Multi-position deletions:** Deletions can span 1-3 consecutive bases (weighted toward 1), which is more realistic than single-position gaps.

### Known Limitations

- No base quality modeling (real DeepVariant uses quality as a separate channel)
- No strand bias simulation
- No mapping quality variation
- Insertions are not modeled (only deletions)
- Error profiles are uniform (don't model ONT homopolymer errors or Illumina cycle-specific bias)

---

## Module 3: scanner.py — Pileup Extraction & Heuristic Filter

### The Critical Bug Fix

The original notebook iterated pileup columns with a manual counter:

```python
# BROKEN — col_idx drifts when pysam skips zero-coverage positions
col_idx = 0
for pileupcolumn in bam.pileup(...):
    grid[row][col_idx] = ...
    col_idx += 1  # ← assumes every position is returned!
```

pysam's `pileup()` skips positions with zero coverage. If position 5 has no reads, the counter jumps from column 4 to column 5 but is actually receiving data for position 6. Every subsequent column is shifted.

The fix uses the actual genomic coordinate:

```python
# FIXED — uses real coordinate to compute column index
for pileup_col in bam.pileup(...):
    col_idx = pileup_col.reference_pos - start_0based  # ← always correct
```

### Heuristic Filter Logic

The scanner uses a two-threshold filter before invoking the CNN:

1. **Minimum total depth (default: 15x):** Positions with fewer than 15 reads don't have enough evidence to call anything.

2. **Minimum alternate reads (default: 3):** At least 3 non-reference reads must be present. This prevents the CNN from being triggered by 1-2 random sequencing errors.

Only positions passing both thresholds get the expensive tensor extraction + CNN inference.

### File Handle Management

The original code opened BAM/FASTA handles on every call without closing them:

```python
# BROKEN — opens a new file handle every call, never closes
def get_tensor(bam_path, fasta_path, ...):
    bam = pysam.AlignmentFile(bam_path, "rb")  # ← leaked!
    fasta = pysam.FastaFile(fasta_path)          # ← leaked!
```

After ~1000 calls (common in batch processing), the OS runs out of file descriptors. The fix uses a context manager that opens handles once per patient:

```python
with open_genomic_files(bam_path, fasta_path) as (bam, fasta):
    for gene in panel:
        for call in scan_region(model, bam, fasta, ...):
            ...  # Same handles reused for all genes
```

---

## Module 4: annotator.py — Biological Translation

### Codon Translation (Forward Strand)

```
Gene Start: 7318 (1-based)
Mutation:   7589

Distance = 7589 - 7318 = 271
Codon #  = 271 ÷ 3 = 90 (0-indexed) → amino acid 91 (1-indexed)
Position = 271 % 3 = 1 (middle base of the codon)

Codon start (0-based) = (7318 - 1) + (90 × 3) = 7587
Fetch 3 bases from FASTA: positions 7587-7589 → "GAC"

Substitute position 1: G[A→T]C → "GTC"
Translate: GAC → D (Asp), GTC → V (Val)
Result: p.D91V (Missense, HIGH impact)
```

### Codon Translation (Reverse Strand)

For genes like rpoB on the minus strand:

```
Gene End:   2276812 (1-based)
Mutation:   2275207

Distance = 2276812 - 2275207 = 1605
Codon #  = 1605 ÷ 3 = 535 (0-indexed) → amino acid 536
Position = 1605 % 3 = 0 (first position in the codon)

Codon start (0-based) = (2275207 - 1) - 0 = 2275206
Fetch 3 bases: "CTC"

After reverse-complement: CTC → GAG
Substitute and reverse-complement the mutant codon too
Compare amino acids
```

### Tiered Classification

```
For each CNN-called variant at position P with alt allele A:
    1. Look up P in panel["targets"][gene]["known_mutations"]
    2. If found AND known_mutations[P]["alt"] == A:
         → TIER 1 (KNOWN): Confirmed resistance
    3. Else:
         → TIER 2 (NOVEL): In drug-target region but not catalogued
```

The tier, along with confidence score and allele frequency, enables prioritized clinical review.

---

## Output Schema

### Master CSV Columns

| Column | Type | Description |
|--------|------|-------------|
| PATIENT | str | Patient identifier |
| GENE | str | Target gene name |
| CONTIG | str | Chromosome/contig |
| POS | int | 1-based genomic position |
| REF | str | Reference allele |
| ALT | str | Alternate allele |
| TOTAL_DP | int | Total read depth |
| AF | float | Allele frequency (alt/total) |
| CLASS | str | CNN class name |
| CONFIDENCE | float | Softmax probability (0-1) |
| AA_CHANGE | str | HGVS protein notation |
| MUT_TYPE | str | Missense/Synonymous/Nonsense/Indel |
| IMPACT | str | HIGH/LOW/UNKNOWN |
| TIER | str | KNOWN or NOVEL |
| KNOWN_DRUG | str | Associated drug (Tier 1 only) |
| LITERATURE | str | Citation (Tier 1 only) |

---

## Comparison with Google DeepVariant

| Feature | minDeepVariant | Google DeepVariant |
|---------|---------------|-------------------|
| Architecture | 2-layer CNN (~77K params) | Inception-v3 (~23M params) |
| Input channels | 1 (base identity only) | 6+ (base, quality, strand, mapping quality, etc.) |
| Training data | Synthetic pileups | Real variants (GIAB truth sets) |
| Variant types | SNPs, simple deletions | SNPs, indels, structural variants |
| Output | 4-class + confidence + tier | Genotype likelihoods (VCF) |
| AMR annotation | Built-in (panel JSON) | Not included |
| Purpose | Education + rapid AMR screening | Production variant calling |
