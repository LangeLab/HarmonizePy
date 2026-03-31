"""Data splitting and per-sub-frame adjustment.

Groups features by their affiliation (missingness pattern), extracts
sub-DataFrames containing only the columns (samples) present for that
group, applies ComBat or limma adjustment, and returns the list of
corrected sub-DataFrames.

This mirrors R ``HarmonizR:::splitting``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .combat_wrapper import adjust_combat
from .limma_wrapper import adjust_limma


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
        One tuple per feature (row) — the block IDs where the feature
        has sufficient data.  From :func:`spotting.spotting_missing_values`.
    data : DataFrame
        Features × samples.
    batch_list : ndarray
        Batch label per sample (1-indexed).
    block_list : ndarray
        Block label per sample (equals batch_list when no blocking).
    algorithm : str
        ``"ComBat"`` or ``"limma"``.
    combat_mode : int
        ComBat mode 1–4 (ignored when algorithm is limma).

    Returns
    -------
    list[DataFrame]
        Corrected sub-DataFrames, one per unique non-empty affiliation.
        Features with empty affiliations are included with all-NaN values
        to preserve completeness.
    """
    batch_arr = np.asarray(batch_list)
    block_arr = np.asarray(block_list)

    # Group features by affiliation
    affil_to_features: dict[tuple[int, ...], list[int]] = {}
    for i, affil in enumerate(affiliation_list):
        affil_to_features.setdefault(affil, []).append(i)

    results: list[pd.DataFrame] = []

    for affil, row_indices in affil_to_features.items():
        sub_data = data.iloc[row_indices]

        if len(affil) == 0:
            # Feature has insufficient data — keep as all-NaN row(s)
            nan_df = pd.DataFrame(
                np.nan,
                index=sub_data.index,
                columns=data.columns,
            )
            results.append(nan_df)
            continue

        # Select columns belonging to the blocks in this affiliation
        col_mask = np.isin(block_arr, affil)
        col_indices = np.where(col_mask)[0]
        sub_df = sub_data.iloc[:, col_indices]

        # Get batch labels for the selected columns
        sub_batch = batch_arr[col_indices]

        # Only adjust if ≥2 batches and ≥2 features
        unique_batches = np.unique(sub_batch)
        if len(unique_batches) < 2 or sub_df.shape[0] < 2:
            # Can't adjust — return as-is within the full column space
            full_df = pd.DataFrame(
                np.nan, index=sub_data.index, columns=data.columns,
            )
            full_df.iloc[:, col_indices] = sub_df.values
            results.append(full_df)
            continue

        # Apply adjustment
        if algorithm == "limma":
            corrected = adjust_limma(sub_df, sub_batch)
        else:
            corrected = adjust_combat(sub_df, sub_batch, mode=combat_mode)

        # Place corrected values back into full-width frame (NaN elsewhere)
        full_df = pd.DataFrame(
            np.nan, index=sub_data.index, columns=data.columns,
        )
        full_df.iloc[:, col_indices] = corrected.values
        results.append(full_df)

    return results
