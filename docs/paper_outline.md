# Paper Outline

## Working Title

**minDeepVariant: A Minimal PyTorch Implementation of Image-Based Variant Calling for Genomics Education and Rapid AMR Profiling**

## Target Journals (ranked by fit)

1. **Bioinformatics (Oxford)** — Applications Note (2 pages, 1 figure)
   - Perfect format: short, code-focused, peer-reviewed
   - Precedent: many "mini" tools published here

2. **PLOS Computational Biology** — Education section
   - Longer format allows the full tutorial narrative
   - Good for pedagogical tools with code

3. **BMC Bioinformatics** — Software article
   - Open access, code-focused
   - Less competitive than Bioinformatics

4. **Journal of Open Source Software (JOSS)**
   - Peer-reviews the software itself, not just the paper
   - Very fast turnaround, good for GitHub-first projects

---

## Abstract (Draft)

Variant calling — identifying genetic differences between a sample and a reference genome — is a cornerstone of genomics. Google's DeepVariant demonstrated that this problem can be reframed as image classification by encoding read pileups as pixel matrices. However, DeepVariant's complexity (23M-parameter Inception-v3 network, multi-channel tensor encoding, extensive training infrastructure) makes it inaccessible as a teaching tool. Here we present minDeepVariant, a minimal reimplementation of the pileup-as-image paradigm in under 500 lines of PyTorch. The tool trains a lightweight CNN (~77K parameters) on synthetic pileup images, classifies variants into four classes (wild-type, heterozygous SNP, homozygous SNP, deletion), and annotates results using a tiered system that distinguishes known resistance mutations from novel candidates. We provide a JSON-configurable panel system that makes the tool organism-agnostic, with a pre-built configuration for *Mycobacterium leprae* antimicrobial resistance profiling. minDeepVariant is freely available at [GitHub URL] and includes an interactive tutorial notebook designed for bioinformatics courses.

---

## Structure

### 1. Introduction (300 words)

**The problem:** Variant calling is taught theoretically but students rarely implement callers from scratch. Existing tools (GATK, bcftools, DeepVariant) are too complex to serve as pedagogical implementations.

**The "mini" philosophy:** Cite Karpathy's minGPT, Hayduk's minAlphaFold2 as precedent for minimal reimplementations that prioritize understanding over performance.

**The DeepVariant insight:** Poplin et al. (2018) showed pileups can be treated as images. We distill this into the simplest possible implementation.

**The AMR application:** Demonstrate practical utility by applying the tool to M. leprae drug resistance profiling, where variant calling tools are limited.

### 2. Implementation (500 words)

**2.1 Pileup Encoding**
- Base-to-pixel mapping (Table 1)
- Tensor dimensions: (depth+1) × window
- Why grayscale is sufficient for education (vs. DeepVariant's 6-channel encoding)

**2.2 CNN Architecture**
- 2 convolutional layers, ~77K parameters
- Dynamic FC layer computation
- Comparison table with DeepVariant's Inception-v3 (Table 2)

**2.3 Synthetic Training Data**
- Why synthetic: avoids the need for truth sets
- Class definitions and simulation rules
- Honest discussion of limitations

**2.4 Heuristic Scanner**
- Two-threshold filter (min depth, min alt reads)
- Coordinate-indexed tensor extraction (the bug fix)

**2.5 Tiered Annotation**
- JSON panel schema (organism-agnostic)
- Tier 1 (Known) vs. Tier 2 (Novel) classification
- Strand-aware codon translation

### 3. Results (400 words)

**3.1 Synthetic Accuracy**
- Training curve (Figure 1A)
- Confusion matrix on held-out synthetic data (Figure 1B)
- Per-class accuracy and confidence calibration

**3.2 M. leprae Application**
- Tested on N=34 clinical isolates from PGIMER, Chandigarh
- Recovered known gyrA D91Y mutation in expected samples
- [If applicable] Novel variants flagged by Tier 2

**3.3 Comparison**
- Same samples processed with NTM-Profiler / GATK
- Concordance on known mutations
- Novel calls unique to minDeepVariant

### 4. Discussion (300 words)

**What this tool is:** An educational implementation and rapid screening tool.

**What this tool is NOT:** A replacement for production variant callers. Limitations:
- Synthetic training data doesn't capture real error profiles
- Single-channel encoding loses base quality information
- Not validated on large truth sets

**Educational value:**
- Used in [course/workshop] at [institution]
- Students report improved understanding of deep learning in genomics
- The tutorial notebook + architecture docs serve as supplementary teaching material

**Future work:**
- Fine-tuning on real labeled variants
- Multi-channel encoding (base quality, strand, mapping quality)
- Integration with RGI/CARD for Tier 3 cross-species annotation
- Transformer-based architecture experiment

### 5. Availability

- GitHub: [URL]
- License: MIT
- Docker: `docker pull [image]`
- Tutorial notebook included

---

## Figures

### Figure 1 (Main — 2 panels)

**A) The Pileup-as-Image Concept**
Four synthetic pileup tensors (Hom-Ref, Het-SNP, Hom-Alt, Deletion) rendered as grayscale images with the center column highlighted. Annotated arrows showing "vertical streak = mutation."

**B) Training Curve + Confusion Matrix**
Left: Loss vs. epoch. Right: 4×4 confusion matrix on synthetic test set.

### Figure 2 (Supplementary)

**The Tiered Annotation Pipeline**
Flowchart: BAM → Scanner → CNN → {Class 0: skip} / {Class 1-3: annotate} → Check panel JSON → Tier 1 (KNOWN) or Tier 2 (NOVEL) → Report.

### Table 1

Base-to-pixel encoding scheme.

### Table 2

Architecture comparison: minDeepVariant vs. Google DeepVariant.

---

## Key Selling Points for Reviewers

1. **Fills a genuine gap:** No pedagogical implementation of DeepVariant's approach exists.
2. **Reproducible:** <500 lines, pip-installable, Docker image, tutorial notebook.
3. **Practical:** Not just a toy — produces clinically interpretable AMR reports.
4. **Organism-agnostic:** JSON panel system means it works beyond M. leprae.
5. **Honest:** Clearly states limitations and doesn't overclaim performance.

---

## Timeline

| Task | Duration | Notes |
|------|----------|-------|
| Finalize code + tests | 1 week | You're almost here |
| Run benchmarks (synthetic accuracy) | 2-3 days | Automated in the notebook |
| Run on 34 M. leprae samples | 1 day | Already done, just needs clean output |
| Compare with NTM-Profiler | 3-5 days | Install and run on same samples |
| Write manuscript | 1-2 weeks | Outline above is the skeleton |
| Figures | 3 days | Mostly generated by the code |
| Submit | Target: 6-8 weeks from now | |
