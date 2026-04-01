"""HarmonizePy - batch-effect harmonization in pure Python."""

from .core import harmonize
from .combat import combat
from .combat_wrapper import adjust_combat
from .limma_wrapper import remove_batch_effect, adjust_limma
from .types import HarmonizeConfig

__all__ = [
    "__version__",
    "harmonize",
    "combat",
    "adjust_combat",
    "remove_batch_effect",
    "adjust_limma",
    "HarmonizeConfig",
]

__version__ = "0.1.0"
