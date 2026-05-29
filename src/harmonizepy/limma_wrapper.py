"""limma-style batch correction (removeBatchEffect), pure NumPy.

Reimplements the algorithm from R limma::removeBatchEffect:

1. Encode batch as sum-to-zero contrasts (``contr.sum``).
2. Build design ``[intercept | batch_contrasts]``.
3. OLS fit: ``beta = (X'X)^{-1} X' Y'``.
4. Subtract batch component: ``Y - beta_batch @ X_batch'``.

Reference
---------
Ritchie ME et al. "limma powers differential expression analyses for
RNA-sequencing and microarray studies." *Nucleic Acids Research*
43(7):e47, 2015.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .validation import validate_limma_input

logger = logging.getLogger(__name__)

_Array = npt.NDArray[np.floating[Any]]


def _group_valid_rows(data: _Array) -> list[tuple[npt.NDArray[np.bool_], npt.NDArray[np.intp]]]:
    """Group row indices by identical non-NaN masks."""
    grouped: dict[bytes, tuple[npt.NDArray[np.bool_], list[int]]] = {}
    valid_masks = ~np.isnan(data)
    for row_index, valid in enumerate(valid_masks):
        key = valid.tobytes()
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = (valid.copy(), [row_index])
        else:
            existing[1].append(row_index)
    return [
        (valid, np.asarray(row_indices, dtype=np.intp))
        for valid, row_indices in grouped.values()
    ]


def remove_batch_effect(
    data: _Array,
    batch: _Array,
) -> _Array:
    """Remove batch effects using a linear model (limma-style).

    Per-cell NaN is handled per-feature by omitting NaN observations from
    the OLS fit (matching R ``limma::removeBatchEffect`` behavior).
    NaN stays in the same positions in the output.

    Parameters
    ----------
    data : ndarray, shape (n_features, n_samples)
        Expression / abundance matrix.  Per-cell NaN is allowed.
    batch : ndarray, shape (n_samples,)
        Integer batch labels.

    Returns
    -------
    ndarray, shape (n_features, n_samples)
        Batch-corrected data.  NaN positions from input are preserved.

    Raises
    ------
    ValueError
        On wrong dimensionality or batch length mismatch.

    Examples
    --------
    >>> import numpy as np
    >>> from harmonizepy import remove_batch_effect
    >>> data = np.random.default_rng(0).normal(10, 2, (20, 8))
    >>> batch = np.array([0]*4 + [1]*4)
    >>> corrected = remove_batch_effect(data, batch)
    >>> corrected.shape
    (20, 8)
    """
    data = np.asarray(data, dtype=np.float64)
    batch = np.asarray(batch).ravel()

    validate_limma_input(data, batch)

    has_nan = np.isnan(data).any()
    if not has_nan:
        return _remove_batch_effect_dense(data, batch)

    if np.isnan(data).all():
        logger.debug("All-NaN input, returning copy")
        return data.copy()

    return _remove_batch_effect_nan(data, batch)


def _remove_batch_effect_nan(data: _Array, batch: _Array) -> _Array:
    """limma batch correction with per-feature NaN handling."""
    _, n_samples = data.shape

    unique_batches = np.unique(batch)
    n_batch = len(unique_batches)
    if n_batch < 2:
        logger.debug("Single batch input, returning copy")
        return data.copy()

    # Build design matrix
    label_map = {b: i for i, b in enumerate(unique_batches)}
    batch_idx = np.array([label_map[b] for b in batch], dtype=np.intp)

    X_batch = np.zeros((n_samples, n_batch - 1), dtype=np.float64)  # noqa: N806
    for j in range(n_batch - 1):
        X_batch[batch_idx == j, j] = 1.0
    X_batch[batch_idx == n_batch - 1, :] = -1.0

    intercept = np.ones((n_samples, 1), dtype=np.float64)
    design = np.hstack([intercept, X_batch])

    corrected = data.copy()
    for valid, row_indices in _group_valid_rows(data):
        if valid.sum() < n_batch:
            continue  # not enough observations, keep NaN

        valid_idx = np.flatnonzero(valid)
        des = design[valid, :]
        group_data = data[np.ix_(row_indices, valid_idx)]
        beta = np.linalg.lstsq(des, group_data.T, rcond=None)[0].T
        beta_batch = np.nan_to_num(beta[:, 1:], nan=0.0)
        corrected[np.ix_(row_indices, valid_idx)] = group_data - beta_batch @ X_batch[valid, :].T

    return corrected


def _remove_batch_effect_dense(data: _Array, batch: _Array) -> _Array:
    """Dense (NaN-free) limma-style batch correction.  See ``remove_batch_effect``."""
    _, n_samples = data.shape

    unique_batches = np.unique(batch)
    n_batch = len(unique_batches)
    if n_batch < 2:
        logger.debug("Single batch input, returning copy")
        return data.copy()

    # --- Sum-to-zero contrasts (R's contr.sum) ---
    label_map = {b: i for i, b in enumerate(unique_batches)}
    batch_idx = np.array([label_map[b] for b in batch], dtype=np.intp)

    X_batch = np.zeros((n_samples, n_batch - 1), dtype=np.float64)  # noqa: N806
    for j in range(n_batch - 1):
        X_batch[batch_idx == j, j] = 1.0
    X_batch[batch_idx == n_batch - 1, :] = -1.0

    intercept = np.ones((n_samples, 1), dtype=np.float64)
    design = np.hstack([intercept, X_batch])

    beta = np.linalg.lstsq(design, data.T, rcond=None)[0].T

    beta_batch = beta[:, 1:]
    beta_batch = np.nan_to_num(beta_batch, nan=0.0)

    corrected = data - beta_batch @ X_batch.T

    return corrected  # type: ignore[no-any-return]


def adjust_limma(
    sub_df: pd.DataFrame,
    batch_labels: _Array,
) -> pd.DataFrame:
    """DataFrame wrapper around ``remove_batch_effect``.

    Parameters
    ----------
    sub_df : DataFrame
        Features x samples.  Must not contain NaN.
    batch_labels : array-like
        Integer batch label per sample (column).

    Returns
    -------
    DataFrame
        Batch-corrected matrix with original index/columns preserved.

    Raises
    ------
    ValueError
        On NaN in *sub_df* or fewer than 2 batches.

    Examples
    --------
    >>> import pandas as pd
    >>> from harmonizepy import adjust_limma
    >>> df = pd.DataFrame({"s1": [1.0, 2.0], "s2": [3.0, 4.0],
    ...                     "s3": [5.0, 6.0], "s4": [7.0, 8.0]})
    >>> corrected = adjust_limma(df, [0, 0, 1, 1])
    """
    logger.debug(
        "Adjusting sub-matrix (%d x %d) with limma",
        sub_df.shape[0],
        sub_df.shape[1],
    )
    result = remove_batch_effect(sub_df.values, np.asarray(batch_labels))
    return pd.DataFrame(result, index=sub_df.index, columns=sub_df.columns)
