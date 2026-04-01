"""HarmonizePy - batch-effect harmonization in pure Python."""

# ---------------------------------------------------------------------------
# Public API — importable directly from `harmonizepy`
# ---------------------------------------------------------------------------
from .combat import combat
from .combat_wrapper import adjust_combat
from .core import harmonize
from .limma_wrapper import adjust_limma, remove_batch_effect
from .types import HarmonizeConfig

__all__ = [  # noqa: RUF022 — grouped by semantic tier, not alphabetical
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
# Semi-public API — importable from submodules for custom pipelines:
#   harmonizepy.sorting.sort_batches
#   harmonizepy.blocking.build_block_list
#   harmonizepy.affiliation.remove_unique_combinations
# ---------------------------------------------------------------------------

__version__ = "0.2.0"
