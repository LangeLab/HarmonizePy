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


def remove_batch_effect(
    data: _Array,
    batch: _Array,
) -> _Array:
    """Remove batch effects using a linear model (limma-style).

    Per-cell NaN is handled by dropping affected feature rows before
    computation.  NaN rows stay NaN in the output.

    Parameters
    ----------
    data : ndarray, shape (n_features, n_samples)
        Expression / abundance matrix.  Per-cell NaN is allowed;
        rows with any NaN are omitted from the OLS fit.
    batch : ndarray, shape (n_samples,)
        Integer batch labels.

    Returns
    -------
    ndarray, shape (n_features, n_samples)
        Batch-corrected data.  Rows with any NaN in the input remain
        all-NaN in the output.  All other rows are adjusted.

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

    # ---- Handle per-cell NaN: drop rows with any NaN -----------------------
    # Match R's implicit row-omission behavior for missing data.
    nan_rows = np.isnan(data).any(axis=1)
    if nan_rows.any():
        clean_data = data[~nan_rows]
        logger.debug(
            "Removed %d feature row(s) with NaN before adjustment; "
            "%d clean feature(s) remain",
            int(nan_rows.sum()),
            int(clean_data.shape[0]),
        )
        result = np.full_like(data, np.nan)
        if clean_data.shape[0] >= 1:
            corrected_clean = _remove_batch_effect_dense(clean_data, batch)
            result[~nan_rows] = corrected_clean
        return result

    return _remove_batch_effect_dense(data, batch)


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
