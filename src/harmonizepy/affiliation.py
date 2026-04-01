"""Affiliation logic: build and reduce per-feature batch affiliations.

The affiliation list records, for each feature (row), which blocks have
sufficient non-missing data.  ``reduce_to_unique_groups`` then collapses
features sharing identical affiliations into groups so that
``splitting.py`` can extract one sub-matrix per group.

In the R HarmonizR source, ``unique()`` on the affiliation list *is* the
binary reduction step - there is no separate operation.  This module
combines both responsibilities.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def build_affiliation_list(
    data: pd.DataFrame,
    batch_list: npt.NDArray[np.integer],
    block_list: npt.NDArray[np.integer],
    needed_values: int,
) -> list[tuple[int, ...]]:
    """Determine per-feature batch/block affiliation.

    For each feature (row), identifies the blocks where the feature has
    at least *needed_values* non-missing observations in every original
    batch contained within that block.

    Parameters
    ----------
    data : DataFrame
        Features x samples matrix (may contain NaN).
    batch_list : ndarray, shape (n_samples,)
        Integer batch label per sample.
    block_list : ndarray, shape (n_samples,)
        Integer block label per sample (equals *batch_list* when no
        blocking is applied).
    needed_values : int
        Minimum non-NA values per batch within a block.

    Returns
    -------
    list[tuple[int, ...]]
        One tuple per feature.  Each tuple contains the sorted block IDs
        where the feature has sufficient data.  An empty tuple means
        the feature has insufficient data everywhere.
    """
    mat = data.values
    n_features = mat.shape[0]

    batch_arr = np.asarray(batch_list)
    block_arr = np.asarray(block_list)

    unique_blocks = np.unique(block_arr)

    # Pre-compute: for each block, the sample indices per original batch
    block_info: list[list[np.ndarray]] = []
    for blk in unique_blocks:
        blk_mask = block_arr == blk
        batches_in_block = np.unique(batch_arr[blk_mask])
        batch_indices = [
            np.where((batch_arr == b) & blk_mask)[0]
            for b in batches_in_block
        ]
        block_info.append(batch_indices)

    notna = ~np.isnan(mat)

    affiliation_list: list[tuple[int, ...]] = []

    for i in range(n_features):
        row_notna = notna[i]
        blocks_present: list[int] = []

        for blk_idx, batch_indices in enumerate(block_info):
            all_ok = True
            for idx in batch_indices:
                if row_notna[idx].sum() < needed_values:
                    all_ok = False
                    break
            if all_ok:
                blocks_present.append(int(unique_blocks[blk_idx]))

        affiliation_list.append(tuple(sorted(blocks_present)))

    return affiliation_list


def reduce_to_unique_groups(
    affiliation_list: list[tuple[int, ...]],
) -> dict[tuple[int, ...], list[int]]:
    """Group feature indices by their shared affiliation pattern.

    Parameters
    ----------
    affiliation_list : list[tuple[int, ...]]
        One tuple per feature from :func:`build_affiliation_list`.

    Returns
    -------
    dict[tuple[int, ...], list[int]]
        Mapping from each unique affiliation to the list of feature
        row indices that share it.  Iteration order matches first
        appearance.
    """
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, affil in enumerate(affiliation_list):
        groups.setdefault(affil, []).append(i)
    return groups
