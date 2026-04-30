"""
model.py — The Brain of minDeepVariant

This module implements the core insight from Google's DeepVariant:
treat variant calling as an IMAGE CLASSIFICATION problem.

DNA pileups (stacks of aligned reads) are encoded as grayscale images
where each base maps to a pixel intensity. Mutations appear as vertical
color discontinuities that a CNN can detect.

Architecture:
    Input:  1 x (depth+1) x window grayscale tensor
            Row 0 = reference sequence
            Rows 1..depth = aligned reads
    Output: 4-class softmax (Hom-Ref, Het-SNP, Hom-Alt-SNP, Deletion)

Reference:
    Poplin et al., "A universal SNP and small-indel variant caller
    using deep neural networks." Nature Biotechnology, 2018.
"""

import torch
import torch.nn as nn

# ──────────────────────────────────────────────
# The DNA Pixel Encoding
# ──────────────────────────────────────────────
# Each nucleotide is mapped to a grayscale intensity.
# The values are arbitrary but fixed — the CNN learns
# to associate specific intensities with specific bases.
# '-' (deletion) gets a distinct low value so the model
# can visually distinguish gaps from unknown bases (N=0).
BASE_TO_PIXEL = {
    "A": 0.25,
    "C": 0.50,
    "G": 0.75,
    "T": 1.00,
    "N": 0.00,
    "-": 0.10,
}

# Variant class labels
CLASS_NAMES = [
    "Hom-Ref",       # Class 0: Wild-type, no variant
    "Het-SNP",       # Class 1: Heterozygous single nucleotide polymorphism
    "Hom-Alt-SNP",   # Class 2: Homozygous alternate SNP
    "Deletion",      # Class 3: Deletion detected
]


class MinDeepVariantCNN(nn.Module):
    """
    A minimal CNN for pileup image classification.

    Two convolutional layers with max-pooling detect local patterns
    (vertical streaks = consistent mismatches across reads), followed
    by a fully connected layer that maps to 4 variant classes.

    Unlike the original model, input dimensions are computed dynamically
    so the architecture adapts to different window/depth settings.

    Parameters
    ----------
    window : int
        Width of the genomic window in base pairs (default: 21).
    depth : int
        Number of read rows in the pileup, excluding the reference
        row (default: 30). Total image height = depth + 1.
    n_classes : int
        Number of output classes (default: 4).
    """

    def __init__(self, window=21, depth=30, n_classes=4):
        super().__init__()

        self.window = window
        self.depth = depth
        img_h = depth + 1  # +1 for the reference row
        img_w = window

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Dynamically compute the flattened size after convolutions
        # by passing a dummy tensor through the feature extractor.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_h, img_w)
            flat_size = self.features(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 1, height, width).

        Returns
        -------
        logits : torch.Tensor
            Shape (batch, n_classes). Raw scores before softmax.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def predict_with_confidence(self, x):
        """
        Run inference and return both the predicted class and
        the softmax probability (confidence score).

        Returns
        -------
        predicted_class : int
        confidence : float
            Softmax probability of the predicted class (0.0–1.0).
        probabilities : torch.Tensor
            Full probability distribution over all classes.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
        return predicted.item(), confidence.item(), probs.squeeze()


def load_model(weights_path, window=21, depth=30, n_classes=4):
    """
    Safely load a trained model from disk.

    Parameters
    ----------
    weights_path : str
        Path to the saved state_dict (.pth file).
    window : int
        Must match the window used during training.
    depth : int
        Must match the depth used during training.

    Returns
    -------
    model : MinDeepVariantCNN
        Model in eval mode, ready for inference.

    Raises
    ------
    FileNotFoundError
        If the weights file does not exist.
    """
    import os
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at '{weights_path}'. "
            f"Train a model first using `mindeepvariant train`."
        )

    model = MinDeepVariantCNN(window=window, depth=depth, n_classes=n_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model
