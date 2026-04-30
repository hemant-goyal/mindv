"""minDeepVariant — A minimal deep learning variant caller."""

__version__ = "0.1.0"
__author__ = "hemant-goyal"

from .model import MinDeepVariantCNN, BASE_TO_PIXEL, CLASS_NAMES, load_model
from .train import train_model, generate_synthetic_pileup
from .scanner import extract_pileup_tensor, scan_region, VariantCall
from .annotator import annotate_variant, get_amino_acid_change, AnnotatedVariant, load_panel
