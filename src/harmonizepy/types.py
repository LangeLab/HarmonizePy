"""Shared data structures for the HarmonizePy pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from .validation import _validate_core_args


@dataclass(frozen=True, slots=True)
class HarmonizeConfig:
    """Configuration for a single harmonization run.

    Parameters
    ----------
    algorithm : str
        ``"ComBat"`` or ``"limma"``.
    combat_mode : int
        ComBat mode 1-4 (ignored when *algorithm* is ``"limma"``).
    needed_values : int or None
        Minimum non-missing values per batch for a feature to be included.
        ``None`` (default) auto-selects: 2 for modes 1, 3 and limma; 1 for
        modes 2, 4.  Mirrors the *needed_values* default in ``harmonize``.
    sort_strategy : str or None
        Batch sorting strategy: ``"sparsity"``, ``"jaccard"``,
        ``"seriation"``, or ``None`` (no sort).
    block_size : int or None
        Number of consecutive batches to group into one block.  Must be
        >= 2 and < total number of unique batches.  ``None`` disables
        blocking.
    unique_removal : bool
        When ``True`` (default), rescue singleton features whose batch
        combination is unique by cropping to the nearest shared pattern.

    Raises
    ------
    ValueError
        On invalid *algorithm*, *combat_mode*, *needed_values*,
        *sort_strategy*, or *block_size*.
    TypeError
        If *unique_removal* is not a bool.

    Examples
    --------
    >>> from harmonizepy import HarmonizeConfig
    >>> cfg = HarmonizeConfig(algorithm="limma")
    >>> cfg.algorithm
    'limma'
    >>> cfg2 = HarmonizeConfig(sort_strategy="sparsity", block_size=2)
    >>> cfg2.sort_strategy
    'sparsity'
    """

    algorithm: str = "ComBat"
    combat_mode: int = 1
    needed_values: int | None = None
    sort_strategy: str | None = None
    block_size: int | None = None
    unique_removal: bool = True

    def __post_init__(self) -> None:
        _validate_core_args(
            self.algorithm, self.combat_mode, self.needed_values,
            self.sort_strategy, self.block_size, self.unique_removal,
        )
