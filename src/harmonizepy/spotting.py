"""Spotting logic: classify features by missingness pattern.

For each feature (row), determines which batches/blocks have sufficient
non-missing data to participate in batch-effect adjustment.  Returns an
*affiliation list* - a list of tuples, one per feature, containing the
sorted set of block IDs where the feature has enough observations.

This mirrors R ``HarmonizR:::spotting_missing_values``.

The heavy lifting is done by :func:`affiliation.build_affiliation_list`;
this module is a thin entry point that preserves the original call site.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .affiliation import build_affiliation_list


def spotting_missing_values(
    data: pd.DataFrame,
    batch_list: np.ndarray,
    block_list: np.ndarray,
    needed_values: int,
) -> list[tuple[int, ...]]:
    """Classify features by their missingness-aware batch affiliation.

    Parameters
    ----------
    data : DataFrame
        Features x samples.  May contain NaN.
    batch_list : ndarray, shape (n_samples,)
        Integer batch label per sample (1-indexed to match R convention).
    block_list : ndarray, shape (n_samples,)
        Integer block label per sample (equals *batch_list* when no
        blocking is applied).
    needed_values : int
        Minimum non-NA values required per *original batch* within a block
        for the feature to be considered present.  ``2`` for modes 1, 3 and
        limma; ``1`` for modes 2, 4.

    Returns
    -------
    list[tuple[int, ...]]
        One tuple per feature.  Each tuple contains the sorted block IDs
        where the feature has sufficient data in every original batch
        contained in that block.  An empty tuple means the feature will
        be dropped (insufficient data everywhere).
    """
    return build_affiliation_list(data, batch_list, block_list, needed_values)
