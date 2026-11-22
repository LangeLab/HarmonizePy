"""Batch-effect correction for omics matrices with missing values, in pure Python.

The public names are listed in ``__all__``: the ``harmonize()`` pipeline, the
ComBat and limma engines, their DataFrame wrappers, the run configuration,
and the input-scale warning. Custom pipelines can also use
``harmonizepy.sorting.sort_batches``, ``harmonizepy.blocking.build_block_list``,
and ``harmonizepy.affiliation.remove_unique_combinations``.
"""

from __future__ import annotations

from .combat import combat
from .combat_wrapper import adjust_combat
from .core import harmonize
from .limma_wrapper import adjust_limma, remove_batch_effect
from .types import HarmonizeConfig, HarmonizeResult
from .validation import InputScaleWarning

# ruff: noqa: RUF022  (__all__ is grouped by role, not sorted)
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
    # Config and results
    "HarmonizeConfig",
    "HarmonizeResult",
    # Warnings
    "InputScaleWarning",
]

__version__ = "0.4.0"
