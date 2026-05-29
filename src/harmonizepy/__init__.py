"""HarmonizePy - batch-effect harmonization in pure Python."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Public API, importable directly from `harmonizepy`
# ---------------------------------------------------------------------------
from .combat import combat
from .combat_wrapper import adjust_combat
from .core import harmonize
from .limma_wrapper import adjust_limma, remove_batch_effect
from .types import HarmonizeConfig

# __all__ intentionally grouped by semantic tier, not alphabetical.
# ruff: noqa: RUF022
__all__ = [
    "__version__",
    # Pipeline
    "harmonize",
    # Engines
    "combat",
    "remove_batch_effect",
    # Wrappers (advanced / custom-pipeline use)
    "adjust_combat",
    "adjust_limma",
    # Config
    "HarmonizeConfig",
]

# ---------------------------------------------------------------------------
# Semi-public API, importable from submodules for custom pipelines:
#   harmonizepy.sorting.sort_batches
#   harmonizepy.blocking.build_block_list
#   harmonizepy.affiliation.remove_unique_combinations
# ---------------------------------------------------------------------------

__version__ = "0.3.2"
