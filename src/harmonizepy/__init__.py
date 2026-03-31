"""HarmonizePy — batch-effect harmonization in pure Python."""

from .core import harmonize
from .combat import combat
from .limma_wrapper import remove_batch_effect

__all__ = ["__version__", "harmonize", "combat", "remove_batch_effect"]

__version__ = "0.0.0"
