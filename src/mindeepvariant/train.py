"""
train.py — The Synthetic Data Engine & Training Loop

Since labeled variant-calling training data is expensive to produce
(it requires validated truth sets like Genome in a Bottle), this module
generates synthetic pileup images to bootstrap the CNN.

How Synthetic Data Works:
    1. Generate a random reference sequence of length `window`.
    2. For each variant class, simulate reads with characteristic patterns:
       - Class 0 (Hom-Ref):  All reads match the reference.
       - Class 1 (Het-SNP):  ~50% of reads show an alternate base at center.
       - Class 2 (Hom-Alt):  All reads show an alternate base at center.
       - Class 3 (Deletion): Most reads show a gap ('-') at center.
    3. Encode the pileup as a grayscale tensor using BASE_TO_PIXEL.

Limitations (important for the paper):
    - Synthetic data does not capture real sequencing error profiles.
    - Base quality scores are not modeled.
    - Real deletions span multiple positions; here they are single-position.
    - Allele frequencies are idealized (50/100%) vs. real biological variation.

These limitations are discussed in the manuscript and represent opportunities
for future work (fine-tuning on real labeled data).
"""

import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from .model import MinDeepVariantCNN, BASE_TO_PIXEL

logger = logging.getLogger(__name__)


def generate_synthetic_pileup(class_label, window=21, depth=30):
    """
    Generate a single synthetic pileup tensor for training.

    Parameters
    ----------
    class_label : int
        0 = Hom-Ref, 1 = Het-SNP, 2 = Hom-Alt, 3 = Deletion.
    window : int
        Width of the genomic window.
    depth : int
        Number of simulated reads (rows 1..depth).

    Returns
    -------
    tensor : torch.Tensor
        Shape (depth+1, window), float32. Row 0 is the reference.
    """
    bases = ["A", "C", "G", "T"]
    ref_sequence = [random.choice(bases) for _ in range(window)]
    center = window // 2

    # Pick an alternate base that differs from the reference at center
    ref_center = ref_sequence[center]
    alt_base = random.choice([b for b in bases if b != ref_center])

    # Row 0: Reference sequence
    grid = [[BASE_TO_PIXEL[b] for b in ref_sequence]]

    for _ in range(depth):
        read = [BASE_TO_PIXEL[b] for b in ref_sequence]

        if class_label == 0:
            # Hom-Ref: reads match reference, with slight noise
            # ~2% random sequencing error at any position
            for j in range(window):
                if random.random() < 0.02:
                    noise_base = random.choice(bases)
                    read[j] = BASE_TO_PIXEL[noise_base]

        elif class_label == 1:
            # Het-SNP: ~30-70% of reads carry the alt allele at center
            # (not always exactly 50% — models biological variation)
            af = random.uniform(0.30, 0.70)
            if random.random() < af:
                read[center] = BASE_TO_PIXEL[alt_base]

        elif class_label == 2:
            # Hom-Alt: nearly all reads carry the alt allele
            # (~5% may still show ref due to mapping artifacts)
            if random.random() > 0.05:
                read[center] = BASE_TO_PIXEL[alt_base]

        elif class_label == 3:
            # Deletion: most reads show a gap at center,
            # spanning 1-3 consecutive positions for realism
            if random.random() > 0.10:
                del_length = random.choice([1, 1, 1, 2, 3])
                for offset in range(del_length):
                    pos = center + offset
                    if pos < window:
                        read[pos] = BASE_TO_PIXEL["-"]

        grid.append(read)

    return torch.tensor(grid, dtype=torch.float32)


def train_model(
    window=21,
    depth=30,
    epochs=30,
    samples_per_epoch=1000,
    learning_rate=0.0005,
    output_path="mindv_weights.pth",
):
    """
    Train the MinDeepVariant CNN on synthetic data.

    Parameters
    ----------
    window : int
        Genomic window width.
    depth : int
        Number of read rows per pileup.
    epochs : int
        Number of training epochs.
    samples_per_epoch : int
        Number of synthetic pileups generated per epoch.
    learning_rate : float
        Adam optimizer learning rate.
    output_path : str
        Where to save the trained weights.

    Returns
    -------
    model : MinDeepVariantCNN
        The trained model.
    history : list[float]
        Per-epoch average loss values.
    """
    model = MinDeepVariantCNN(window=window, depth=depth)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    logger.info(
        f"Training minDeepVariant | {epochs} epochs x {samples_per_epoch} samples | "
        f"window={window}, depth={depth}"
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for _ in range(samples_per_epoch):
            label = random.randint(0, 3)
            tensor = generate_synthetic_pileup(label, window, depth)
            inputs = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            targets = torch.tensor([label])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / samples_per_epoch
        history.append(avg_loss)
        logger.info(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}")

    # Save trained weights
    torch.save(model.state_dict(), output_path)
    logger.info(f"Weights saved to {output_path}")

    return model, history
