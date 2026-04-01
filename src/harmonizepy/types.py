"""Shared data structures for the HarmonizePy pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


_VALID_SORT_STRATEGIES: frozenset[str] = frozenset({"sparsity", "jaccard", "seriation"})


@dataclass(frozen=True, slots=True)
class HarmonizeConfig:
    """Configuration for a single harmonization run.

    Parameters
    ----------
    algorithm : str
        ``"ComBat"`` or ``"limma"``.
    combat_mode : int
        ComBat mode 1-4 (ignored when *algorithm* is ``"limma"``).
    needed_values : int
        Minimum non-missing values per batch for a feature to be included.
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
    needed_values: int = 2
    sort_strategy: str | None = None
    block_size: int | None = None
    unique_removal: bool = True

    def __post_init__(self) -> None:
        if self.algorithm not in ("ComBat", "limma"):
            raise ValueError(
                f"algorithm must be 'ComBat' or 'limma', got {self.algorithm!r}"
            )
        if self.combat_mode not in (1, 2, 3, 4):
            raise ValueError(
                f"combat_mode must be 1-4, got {self.combat_mode}"
            )
        if self.needed_values < 1:
            raise ValueError(
                f"needed_values must be >= 1, got {self.needed_values}"
            )
        if self.sort_strategy not in _VALID_SORT_STRATEGIES and self.sort_strategy is not None:
            raise ValueError(
                f"sort_strategy must be one of "
                f"{sorted(_VALID_SORT_STRATEGIES)!r} or None, "
                f"got {self.sort_strategy!r}"
            )
        if self.block_size is not None and self.block_size < 2:
            raise ValueError(
                f"block_size must be >= 2 or None, got {self.block_size}"
            )
        if not isinstance(self.unique_removal, bool):
            raise TypeError(
                f"unique_removal must be bool, got "
                f"{type(self.unique_removal).__name__!r}"
            )


class AffiliationEntry(NamedTuple):
    """Per-feature affiliation: the block IDs with sufficient data.

    Wraps the raw tuple produced by spotting to give it a name and
    make pipeline code self-documenting.

    Parameters
    ----------
    blocks : tuple[int, ...]
        Sorted block IDs where the feature has sufficient observations.
        An empty tuple means the feature is dropped from adjustment.

    Raises
    ------
    TypeError
        If *blocks* is not a tuple of integers.

    Examples
    --------
    >>> from harmonizepy.types import AffiliationEntry
    >>> entry = AffiliationEntry(blocks=(1, 2))
    >>> entry.blocks
    (1, 2)
    """

    blocks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BatchDescription:
    """Validated sample-to-batch mapping.

    Parameters
    ----------
    sample_ids : tuple[str, ...]
        Sample identifiers matching data column names.
    batch_labels : tuple[int, ...]
        Integer batch label per sample.

    Raises
    ------
    ValueError
        If *sample_ids* and *batch_labels* differ in length, or if
        *batch_labels* contains fewer than 2 unique values.

    Examples
    --------
    >>> from harmonizepy.types import BatchDescription
    >>> bd = BatchDescription(("s1", "s2", "s3"), (1, 1, 2))
    >>> len(bd.batch_labels)
    3
    """

    sample_ids: tuple[str, ...]
    batch_labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.sample_ids) != len(self.batch_labels):
            raise ValueError(
                f"sample_ids length ({len(self.sample_ids)}) != "
                f"batch_labels length ({len(self.batch_labels)})"
            )
        if len(set(self.batch_labels)) < 2:
            raise ValueError("batch_labels must contain at least 2 unique values")
