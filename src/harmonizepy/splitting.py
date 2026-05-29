"""Data splitting and per-sub-frame adjustment.

Groups features by their affiliation (missingness pattern), extracts
sub-matrices containing only the columns (samples) present for that
group, applies ComBat or limma adjustment, and returns the fully
assembled corrected DataFrame.

This mirrors the split-and-rebuild effect of R ``HarmonizR:::splitting``
without materializing one full-width DataFrame per affiliation group.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

from .affiliation import reduce_to_unique_groups
from .combat import combat
from .combat_wrapper import _MODE_MAP
from .limma_wrapper import remove_batch_effect

logger = logging.getLogger(__name__)


def splitting(
    affiliation_list: list[tuple[int, ...]],
    data: pd.DataFrame,
    batch_list: np.ndarray,
    block_list: np.ndarray,
    algorithm: str = "ComBat",
    combat_mode: int = 1,
    *,
    output_col_order: npt.NDArray[np.intp] | None = None,
    output_columns: pd.Index | None = None,
    data_np: npt.NDArray[np.float64] | None = None,
) -> pd.DataFrame:
    """Split data by affiliation, adjust each sub-frame, return result.

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
    data_np : ndarray or None, keyword-only
        Optional precomputed float64 view or copy of *data* with shape
        ``(n_features, n_samples)``. When provided, reused instead of
        converting *data* again inside this function.

    Returns
    -------
    DataFrame
        Fully assembled corrected matrix. Features with empty affiliations
        remain all-NaN to preserve completeness.

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
    >>> result = splitting(affil, data, batch, batch, algorithm="ComBat", combat_mode=2)
    >>> result.shape
    (3, 4)
    """
    batch_arr = np.asarray(batch_list)
    block_arr = np.asarray(block_list)

    # Convert to numpy once to avoid DataFrame overhead in the hot loop.
    if data_np is None:
        data_np = data.to_numpy(dtype=np.float64)

    # Group features by affiliation using the shared reducer
    affil_to_features = reduce_to_unique_groups(affiliation_list)

    # Cache sample indices once per block and affiliation to avoid
    # rescanning block_arr for every unique group.
    block_to_cols: dict[int, list[int]] = {}
    for col_index, block_id in enumerate(block_arr.tolist()):
        block_to_cols.setdefault(int(block_id), []).append(col_index)
    block_to_cols_arr = {
        block_id: np.asarray(col_indices, dtype=np.intp)
        for block_id, col_indices in block_to_cols.items()
    }
    affil_to_cols = {
        affil: np.concatenate([block_to_cols_arr[block_id] for block_id in affil])
        for affil in affil_to_features
        if affil
    }

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

    n_single_batch = 0
    n_single_feature = 0

    for affil, row_indices in affil_to_features.items():
        if len(affil) == 0:
            continue  # rows stay NaN

        # Select cached columns for the blocks in this affiliation.
        col_indices = affil_to_cols[affil]
        target_col_indices = (
            col_indices if output_col_order is None else output_col_order[col_indices]
        )

        # Fast path: single feature (common case with high NaN).
        # Avoids np.array + np.ix_ overhead for the trivial 1-row case.
        if len(row_indices) == 1:
            ri = row_indices[0]
            sub_data = data_np[ri, col_indices]
            sub_batch = batch_arr[col_indices]
            if len(np.unique(sub_batch)) < 2:
                n_single_batch += 1
                output[ri, target_col_indices] = sub_data
            else:
                corrected = (
                    remove_batch_effect(sub_data.reshape(1, -1), sub_batch)
                    if algorithm == "limma"
                    else combat(sub_data.reshape(1, -1), sub_batch, **_MODE_MAP[combat_mode])
                )
                output[ri, target_col_indices] = corrected.ravel()
            continue

        # General path: multiple features.
        row_idx = np.array(row_indices, dtype=np.intp)
        sub_data = data_np[np.ix_(row_idx, col_indices)]
        sub_batch = batch_arr[col_indices]

        unique_batches = np.unique(sub_batch)
        if len(unique_batches) < 2:
            n_single_batch += len(row_idx)
            output[np.ix_(row_idx, target_col_indices)] = sub_data
            continue

        if algorithm == "limma":
            corrected = remove_batch_effect(sub_data, sub_batch)
        else:
            corrected = combat(sub_data, sub_batch, **_MODE_MAP[combat_mode])

        output[np.ix_(row_idx, target_col_indices)] = corrected

    n_uncorrected = n_single_batch + n_single_feature
    if n_uncorrected > 0:
        logger.info(
            "%d feature(s) passed through without correction "
            "(%d single-batch groups, %d single-feature groups)",
            n_uncorrected,
            n_single_batch,
            n_single_feature,
        )

    return pd.DataFrame(
        output,
        index=data.index,
        columns=data.columns if output_columns is None else output_columns,
        copy=False,
    )
