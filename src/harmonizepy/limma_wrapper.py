"""limma-style batch correction (removeBatchEffect) — pure NumPy.

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

import numpy as np
import numpy.typing as npt
import pandas as pd

_Array = npt.NDArray[np.floating]


def remove_batch_effect(
    data: _Array,
    batch: _Array,
) -> _Array:
    """Remove batch effects using a linear model (limma-style).

    Parameters
    ----------
    data : ndarray, shape (n_features, n_samples)
        Expression / abundance matrix.  **Must not contain NaN.**
    batch : ndarray, shape (n_samples,)
        Integer batch labels.

    Returns
    -------
    ndarray, shape (n_features, n_samples)
        Batch-corrected data.

    Raises
    ------
    ValueError
        On NaN in *data* or fewer than 2 batches.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got {data.ndim}-D")
    if np.isnan(data).any():
        raise ValueError("data must not contain NaN")

    batch = np.asarray(batch).ravel()
    n_features, n_samples = data.shape
    if batch.shape[0] != n_samples:
        raise ValueError(
            f"batch length ({batch.shape[0]}) != number of samples ({n_samples})"
        )

    unique_batches = np.unique(batch)
    n_batch = len(unique_batches)
    if n_batch < 2:
        return data.copy()

    # --- Sum-to-zero contrasts (R's contr.sum) ---
    # For k levels, produces (n_samples, k-1) matrix.
    # Level i (i < k): column j = 1 if i==j, else 0
    # Level k (last):  all columns = -1
    label_map = {b: i for i, b in enumerate(unique_batches)}
    batch_idx = np.array([label_map[b] for b in batch], dtype=np.intp)

    X_batch = np.zeros((n_samples, n_batch - 1), dtype=np.float64)
    for j in range(n_batch - 1):
        X_batch[batch_idx == j, j] = 1.0
    X_batch[batch_idx == n_batch - 1, :] = -1.0

    # --- Design: intercept + batch contrasts ---
    intercept = np.ones((n_samples, 1), dtype=np.float64)
    design = np.hstack([intercept, X_batch])  # (n_samples, 1 + n_batch-1)

    # --- OLS fit: beta = (X'X)^{-1} X' Y'  →  (n_coefs, n_features) ---
    # Then beta.T → (n_features, n_coefs)
    beta = np.linalg.lstsq(design, data.T, rcond=None)[0].T  # (n_features, n_coefs)

    # --- Subtract batch effect (columns after intercept) ---
    beta_batch = beta[:, 1:]  # (n_features, n_batch-1)
    beta_batch = np.nan_to_num(beta_batch, nan=0.0)

    corrected = data - beta_batch @ X_batch.T

    return corrected


def adjust_limma(
    sub_df: pd.DataFrame,
    batch_labels: _Array,
) -> pd.DataFrame:
    """DataFrame wrapper around :func:`remove_batch_effect`.

    Parameters
    ----------
    sub_df : DataFrame
        Features × samples.  Must not contain NaN.
    batch_labels : array-like
        Integer batch label per sample (column).

    Returns
    -------
    DataFrame
        Batch-corrected matrix with original index/columns preserved.
    """
    result = remove_batch_effect(sub_df.values, np.asarray(batch_labels))
    return pd.DataFrame(result, index=sub_df.index, columns=sub_df.columns)
