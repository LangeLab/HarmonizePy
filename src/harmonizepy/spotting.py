"""Spotting logic: classify features by missingness pattern.

For each feature (row), determines which batches/blocks have sufficient
non-missing data to participate in batch-effect adjustment.  Returns an
*affiliation list* — a list of tuples, one per feature, containing the
sorted set of block IDs where the feature has enough observations.

This mirrors R ``HarmonizR:::spotting_missing_values``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
        Features × samples.  May contain NaN.
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
    mat = data.values  # (n_features, n_samples)
    n_features, n_samples = mat.shape

    batch_arr = np.asarray(batch_list)
    block_arr = np.asarray(block_list)

    # Pre-compute batch and block boundaries
    unique_blocks = np.unique(block_arr)
    # For each block, find the original batches it contains and sample indices
    block_info: list[list[tuple[np.ndarray, ...]]] = []
    for blk in unique_blocks:
        blk_mask = block_arr == blk
        batches_in_block = np.unique(batch_arr[blk_mask])
        batch_indices = []
        for b in batches_in_block:
            idx = np.where((batch_arr == b) & blk_mask)[0]
            batch_indices.append(idx)
        block_info.append(batch_indices)

    # Not-NaN mask
    notna = ~np.isnan(mat)  # (n_features, n_samples)

    affiliation_list: list[tuple[int, ...]] = []

    for i in range(n_features):
        row_notna = notna[i]
        blocks_present: list[int] = []

        for blk_idx, batch_indices in enumerate(block_info):
            # Feature must have >= needed_values in EVERY batch within block
            all_ok = True
            for idx in batch_indices:
                if row_notna[idx].sum() < needed_values:
                    all_ok = False
                    break
            if all_ok:
                blocks_present.append(int(unique_blocks[blk_idx]))

        affiliation_list.append(tuple(sorted(blocks_present)))

    return affiliation_list
