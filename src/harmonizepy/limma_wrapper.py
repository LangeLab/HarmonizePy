"""limma-style batch correction (removeBatchEffect)  -  pure NumPy.

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

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .validation import validate_limma_input

_Array = npt.NDArray[np.floating[Any]]


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

    _, n_samples = data.shape

    unique_batches = np.unique(batch)
    n_batch = len(unique_batches)
    if n_batch < 2:
        return data.copy()  # type: ignore[no-any-return]

    # --- Sum-to-zero contrasts (R's contr.sum) ---
    # For k levels, produces (n_samples, k-1) matrix.
    # Level i (i < k): column j = 1 if i==j, else 0
    # Level k (last):  all columns = -1
    label_map = {b: i for i, b in enumerate(unique_batches)}
    batch_idx = np.array([label_map[b] for b in batch], dtype=np.intp)

    X_batch = np.zeros((n_samples, n_batch - 1), dtype=np.float64)  # noqa: N806
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

    return corrected  # type: ignore[no-any-return]


def adjust_limma(
    sub_df: pd.DataFrame,
    batch_labels: _Array,
) -> pd.DataFrame:
    """DataFrame wrapper around :func:`remove_batch_effect`.

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
    result = remove_batch_effect(sub_df.values, np.asarray(batch_labels))
    return pd.DataFrame(result, index=sub_df.index, columns=sub_df.columns)
