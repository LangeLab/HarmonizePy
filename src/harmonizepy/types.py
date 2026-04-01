"""Shared data structures for the HarmonizePy pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from .validation import _VALID_SORT_STRATEGIES


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
        modes 2, 4.  Mirrors the *needed_values* default in :func:`harmonize`.
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
        if self.algorithm not in ("ComBat", "limma"):
            raise ValueError(f"algorithm must be 'ComBat' or 'limma', got {self.algorithm!r}")
        if self.combat_mode not in (1, 2, 3, 4):
            raise ValueError(f"combat_mode must be 1-4, got {self.combat_mode}")
        if self.needed_values is not None and self.needed_values < 1:
            raise ValueError(f"needed_values must be >= 1 or None, got {self.needed_values}")
        if self.sort_strategy not in _VALID_SORT_STRATEGIES and self.sort_strategy is not None:
            raise ValueError(
                f"sort_strategy must be one of "
                f"{sorted(_VALID_SORT_STRATEGIES)!r} or None, "
                f"got {self.sort_strategy!r}"
            )
        if self.block_size is not None and self.block_size < 2:
            raise ValueError(f"block_size must be >= 2 or None, got {self.block_size}")
        if not isinstance(self.unique_removal, bool):
            raise TypeError(
                f"unique_removal must be bool, got {type(self.unique_removal).__name__!r}"
            )
