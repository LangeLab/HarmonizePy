"""The run configuration and the array type shared by the engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from .validation import validate_core_args

FloatArray = npt.NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class HarmonizeConfig:
    """Settings for one run of ``harmonize()``, checked when created.

    The fields have the meaning and defaults of the ``harmonize()`` arguments
    of similar name.

    Parameters
    ----------
    algorithm : {"ComBat", "limma"}
        Correction engine.
    combat_mode : {1, 2, 3, 4}
        ComBat mode; ignored for limma.
    needed_values : int or None
        Minimum observed values per batch; ``None`` chooses 2 for limma and
        modes 1 and 3, and 1 for modes 2 and 4.
    sort_strategy : {"sparsity", "jaccard", "seriation"} or None
        Batch order used to form blocks (``harmonize()``'s ``sort``).
    block_size : int or None
        Batches per block (``harmonize()``'s ``block``). Whether it is below
        the number of batches is checked when the data are read.
    unique_removal : bool
        Whether a feature with an affiliation of its own is moved into a
        shared one.

    Raises
    ------
    ValueError
        On an unknown algorithm, ComBat mode, or sort strategy, a
        ``needed_values`` below 1, or a block size that is not an integer of at
        least 2.
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

    algorithm: Literal["ComBat", "limma"] = "ComBat"
    combat_mode: int = 1
    needed_values: int | None = None
    sort_strategy: str | None = None
    block_size: int | None = None
    unique_removal: bool = True

    def __post_init__(self) -> None:
        validate_core_args(
            self.algorithm,
            self.combat_mode,
            self.needed_values,
            self.sort_strategy,
            self.block_size,
            self.unique_removal,
        )


class HarmonizeResult(NamedTuple):
    """The corrected matrix and the per-feature correction report of one run.

    Returned by ``harmonize(..., return_report=True)``; unpacks as
    ``data, report``.
    """

    data: pd.DataFrame
    report: pd.DataFrame
