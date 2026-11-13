"""DoubleAdapt multi-factor prediction model."""
from .model import DoubleAdaptModel
from .data import load_panel, validate_panel
__all__ = ["DoubleAdaptModel", "load_panel", "validate_panel"]
__version__ = "0.1.0"
