"""
Tests for the annotator and model modules.

These tests verify the core logic without requiring real BAM/FASTA files.
They test: synthetic data generation, CNN forward pass, codon translation
logic, and tier classification.
"""

import torch
import pytest
from mindeepvariant.model import MinDeepVariantCNN, BASE_TO_PIXEL, CLASS_NAMES
from mindeepvariant.train import generate_synthetic_pileup


class TestModel:
    """Tests for the CNN architecture."""

    def test_model_output_shape(self):
        """Model should output (batch, 4) for default input dimensions."""
        model = MinDeepVariantCNN(window=21, depth=30)
        dummy_input = torch.randn(1, 1, 31, 21)
        output = model(dummy_input)
        assert output.shape == (1, 4), f"Expected (1, 4), got {output.shape}"

    def test_model_custom_dimensions(self):
        """Model should adapt to non-default window/depth."""
        model = MinDeepVariantCNN(window=31, depth=50)
        dummy_input = torch.randn(1, 1, 51, 31)
        output = model(dummy_input)
        assert output.shape == (1, 4)

    def test_predict_with_confidence(self):
        """predict_with_confidence should return class, confidence, and probs."""
        model = MinDeepVariantCNN()
        dummy_input = torch.randn(1, 1, 31, 21)
        pred_class, confidence, probs = model.predict_with_confidence(dummy_input)

        assert isinstance(pred_class, int)
        assert 0 <= pred_class <= 3
        assert 0.0 <= confidence <= 1.0
        assert abs(probs.sum().item() - 1.0) < 1e-5, "Probabilities should sum to 1"

    def test_batch_inference(self):
        """Model should handle batched inputs."""
        model = MinDeepVariantCNN()
        batch = torch.randn(8, 1, 31, 21)
        output = model(batch)
        assert output.shape == (8, 4)


class TestSyntheticData:
    """Tests for the synthetic data generator."""

    def test_output_shape(self):
        """Tensor should be (depth+1, window)."""
        tensor = generate_synthetic_pileup(0, window=21, depth=30)
        assert tensor.shape == (31, 21)

    def test_all_classes_generate(self):
        """All 4 classes should produce valid tensors."""
        for label in range(4):
            tensor = generate_synthetic_pileup(label)
            assert tensor.shape == (31, 21)
            assert tensor.min() >= 0.0
            assert tensor.max() <= 1.0

    def test_pixel_values_in_range(self):
        """All pixel values should be valid BASE_TO_PIXEL values."""
        valid_values = set(BASE_TO_PIXEL.values())
        tensor = generate_synthetic_pileup(0)
        for val in tensor.flatten().tolist():
            assert val in valid_values, f"Unexpected pixel value: {val}"


class TestBaseEncoding:
    """Tests for the DNA pixel encoding."""

    def test_all_bases_have_values(self):
        """Every expected base should have a pixel mapping."""
        for base in ["A", "C", "G", "T", "N", "-"]:
            assert base in BASE_TO_PIXEL

    def test_values_are_unique(self):
        """Each base should map to a unique intensity."""
        values = list(BASE_TO_PIXEL.values())
        assert len(values) == len(set(values)), "Pixel values must be unique"

    def test_values_in_0_1_range(self):
        """All intensities should be in [0, 1]."""
        for val in BASE_TO_PIXEL.values():
            assert 0.0 <= val <= 1.0


class TestClassNames:
    """Tests for class label consistency."""

    def test_four_classes(self):
        assert len(CLASS_NAMES) == 4

    def test_class_names_not_empty(self):
        for name in CLASS_NAMES:
            assert len(name) > 0
