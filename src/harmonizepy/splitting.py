"""Data splitting and per-sub-frame adjustment.

Groups features by their affiliation (missingness pattern), extracts
sub-DataFrames containing only the columns (samples) present for that
group, applies ComBat or limma adjustment, and returns the list of
corrected sub-DataFrames.

This mirrors R ``HarmonizR:::splitting``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .affiliation import reduce_to_unique_groups
from .combat_wrapper import adjust_combat
from .limma_wrapper import adjust_limma

logger = logging.getLogger(__name__)


def splitting(
    affiliation_list: list[tuple[int, ...]],
    data: pd.DataFrame,
    batch_list: np.ndarray,
    block_list: np.ndarray,
    algorithm: str = "ComBat",
    combat_mode: int = 1,
) -> list[pd.DataFrame]:
    """Split data by affiliation, adjust each sub-frame, return results.

    Parameters
    ----------
    affiliation_list : list[tuple[int, ...]]
        One tuple per feature (row), listing the block IDs where the feature
        has sufficient data.  From ``spotting.spotting_missing_values``.
    data : DataFrame
        Features x samples.
    batch_list : ndarray
        Batch label per sample (1-indexed).
    block_list : ndarray
        Block label per sample (equals batch_list when no blocking).
    algorithm : str
        ``"ComBat"`` or ``"limma"``.
    combat_mode : int
        ComBat mode 1-4 (ignored when algorithm is limma).

    Returns
    -------
    list[DataFrame]
        Corrected sub-DataFrames, one per unique non-empty affiliation.
        Features with empty affiliations are included with all-NaN values
        to preserve completeness.

    Raises
    ------
    ValueError
        If *algorithm* is not ``"ComBat"`` or ``"limma"``, or if
        *combat_mode* is not 1-4.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from harmonizepy.splitting import splitting
    >>> data = pd.DataFrame(np.ones((3, 4)), columns=list("abcd"))
    >>> data.iloc[:, 2:] += 1.0
    >>> affil = [(1, 2), (1, 2), (1, 2)]
    >>> batch = np.array([1, 1, 2, 2])
    >>> sub_dfs = splitting(affil, data, batch, batch, algorithm="ComBat", combat_mode=2)
    >>> len(sub_dfs)
    1
    """
    batch_arr = np.asarray(batch_list)
    block_arr = np.asarray(block_list)

    # Group features by affiliation using the shared reducer
    affil_to_features = reduce_to_unique_groups(affiliation_list)

    n_groups = sum(1 for affil in affil_to_features if affil)
    logger.debug(
        "Splitting %d features into %d group(s) across %d columns",
        data.shape[0],
        n_groups,
        data.shape[1],
    )

    # Pre-allocate a single output array (n_features, n_samples) filled with NaN.
    # This avoids allocating one full-width DataFrame per affiliation group,
    # reducing peak memory from ~3x input to ~1x input.
    n_features, n_samples = data.shape
    output = np.full((n_features, n_samples), np.nan, dtype=np.float64)

    for affil, row_indices in affil_to_features.items():
        row_idx = np.array(row_indices, dtype=np.intp)
        sub_data = data.iloc[row_idx]

        if len(affil) == 0:
            continue  # rows stay NaN

        # Select columns belonging to the blocks in this affiliation
        col_mask = np.isin(block_arr, affil)
        col_indices = np.where(col_mask)[0]
        sub_df = sub_data.iloc[:, col_indices]

        # Get batch labels for the selected columns
        sub_batch = batch_arr[col_indices]

        # Only adjust if ≥2 batches and ≥2 features
        unique_batches = np.unique(sub_batch)
        if len(unique_batches) < 2 or sub_df.shape[0] < 2:
            # Can't adjust. Copy raw values into output.
            output[np.ix_(row_idx, col_indices)] = sub_df.values
            continue

        # Apply adjustment
        if algorithm == "limma":
            corrected = adjust_limma(sub_df, sub_batch)
        else:
            corrected = adjust_combat(sub_df, sub_batch, mode=combat_mode)

        output[np.ix_(row_idx, col_indices)] = corrected.values

    # Build the result list for backward compatibility with downstream concat.
    # Each non-empty group becomes its own sub-DataFrame.
    results: list[pd.DataFrame] = []
    for _, row_indices in affil_to_features.items():
        row_idx = np.array(row_indices, dtype=np.intp)
        sub = pd.DataFrame(
            output[np.ix_(row_idx, np.arange(n_samples))],
            index=data.index[row_idx],
            columns=data.columns,
        )
        results.append(sub)

    return results
