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

from .validation import batch_codes, validate_limma_input

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
        (valid, np.asarray(row_indices, dtype=np.intp)) for valid, row_indices in grouped.values()
    ]


def remove_batch_effect(
    data: _Array,
    batch: npt.ArrayLike,
) -> _Array:
    """Remove batch effects using a linear model (limma-style).

    Per-cell NaN is handled per-feature by omitting NaN observations from
    the OLS fit (matching R ``limma::removeBatchEffect`` behavior).
    NaN stays in the same positions in the output.

    Parameters
    ----------
    data : ndarray, shape (n_features, n_samples)
        Expression / abundance matrix.  Per-cell NaN is allowed.
    batch : array_like, shape (n_samples,)
        Batch label per sample: whole numbers, one batch per distinct value.

    Returns
    -------
    ndarray, shape (n_features, n_samples)
        Batch-corrected data.  NaN positions from input are preserved.  A
        feature with fewer observed values than batches cannot be fitted and
        keeps its raw values; the run log counts such features.

    Raises
    ------
    ValueError
        On wrong dimensionality, batch length mismatch, batch labels that are
        not whole numbers, or infinite values.

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
    batch_int = batch_codes(batch)

    validate_limma_input(data, batch_int)

    has_nan = np.isnan(data).any()
    if not has_nan:
        return _remove_batch_effect_dense(data, batch_int)

    if np.isnan(data).all():
        logger.debug("All-NaN input, returning copy")
        return data.copy()

    return _remove_batch_effect_nan(data, batch_int)


def _remove_batch_effect_nan(data: _Array, batch: npt.NDArray[np.intp]) -> _Array:
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
    n_unfitted = 0
    for valid, row_indices in _group_valid_rows(data):
        if valid.sum() < n_batch:
            n_unfitted += len(row_indices)
            continue

        valid_idx = np.flatnonzero(valid)
        des = design[valid, :]
        group_data = data[np.ix_(row_indices, valid_idx)]
        beta = np.linalg.lstsq(des, group_data.T, rcond=None)[0].T
        beta_batch = np.nan_to_num(beta[:, 1:], nan=0.0)
        corrected[np.ix_(row_indices, valid_idx)] = group_data - beta_batch @ X_batch[valid, :].T

    if n_unfitted:
        logger.info(
            "%d feature(s) with fewer observed values than batches left uncorrected",
            n_unfitted,
        )
    return corrected


def _remove_batch_effect_dense(data: _Array, batch: npt.NDArray[np.intp]) -> _Array:
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

    return corrected


def adjust_limma(
    sub_df: pd.DataFrame,
    batch_labels: npt.ArrayLike,
) -> pd.DataFrame:
    """DataFrame wrapper around ``remove_batch_effect``.

    Parameters
    ----------
    sub_df : DataFrame
        Features x samples.  Missing cells are allowed and stay missing.
    batch_labels : array-like
        Batch label per sample (column), whole numbers.

    Returns
    -------
    DataFrame
        Batch-corrected matrix with original index/columns preserved.

    Raises
    ------
    ValueError
        If the batch labels are not whole numbers or do not match the number
        of columns, or *sub_df* holds infinite values.

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
