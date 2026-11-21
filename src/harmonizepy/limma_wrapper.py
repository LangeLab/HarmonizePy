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

import numpy as np
import numpy.typing as npt
import pandas as pd

from .combat import group_valid_rows
from .types import FloatArray
from .validation import batch_codes, validate_engine_input

logger = logging.getLogger(__name__)


def _contrast_design(batch: npt.NDArray[np.intp]) -> tuple[FloatArray, FloatArray]:
    """Return the intercept-plus-batch design and its batch columns, with sum-to-zero contrasts.

    The batch columns follow R's ``contr.sum``: batch ``j`` is 1 in column ``j``,
    and the last batch is -1 in every column, so the batch coefficients sum to
    zero and removing them keeps each feature's mean over batches.
    """
    labels, batch_idx = np.unique(batch, return_inverse=True)
    n_batch = len(labels)
    x_batch = np.zeros((batch.shape[0], n_batch - 1), dtype=np.float64)
    for j in range(n_batch - 1):
        x_batch[batch_idx == j, j] = 1.0
    x_batch[batch_idx == n_batch - 1, :] = -1.0
    design = np.hstack([np.ones((batch.shape[0], 1), dtype=np.float64), x_batch])
    return design, x_batch


def remove_batch_effect(
    data: FloatArray,
    batch: npt.ArrayLike,
) -> FloatArray:
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
        feature is fitted on the batches it has values in, with sum-to-zero
        contrasts over those batches, so a batch it lacks does not shift its
        level. A feature with values in one batch only, or missing a batch with
        no more values than observed batches, keeps its raw values; the log
        counts both.

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

    validate_engine_input(data, batch_int)

    has_nan = np.isnan(data).any()
    if not has_nan:
        return _remove_batch_effect_dense(data, batch_int)

    if np.isnan(data).all():
        logger.debug("All-NaN input, returning copy")
        return data.copy()

    return _remove_batch_effect_nan(data, batch_int)


def _remove_batch_effect_nan(data: FloatArray, batch: npt.NDArray[np.intp]) -> FloatArray:
    """limma correction fitted on each feature's observed cells.

    Features that share a pattern of observed cells are fitted together, with
    contrasts over the batches they are observed in. A feature with values in
    one batch only keeps its raw values, as does a feature missing a batch whose
    values do not outnumber its observed batches, since its fit would leave no
    residual and collapse its values to one level.
    """
    n_batch = len(np.unique(batch))
    if n_batch < 2:
        logger.debug("Single batch input, returning copy")
        return data.copy()
    design, x_batch = _contrast_design(batch)

    corrected = data.copy()
    n_single_batch = n_no_residual = 0
    groups = group_valid_rows(data)
    for row_indices in groups.row_groups:
        valid = groups.valid_masks[row_indices[0]]
        observed_batches = np.unique(batch[valid])
        if observed_batches.size == n_batch:
            group_design, group_x = design[valid, :], x_batch[valid, :]
        elif observed_batches.size < 2:
            n_single_batch += len(row_indices)
            continue
        elif valid.sum() <= observed_batches.size:
            n_no_residual += len(row_indices)
            continue
        else:
            group_design, group_x = _contrast_design(batch[valid])
        valid_idx = np.flatnonzero(valid)
        group_data = data[np.ix_(row_indices, valid_idx)]
        beta = np.linalg.lstsq(group_design, group_data.T, rcond=None)[0].T
        # Coefficients overflow to NaN only for values near the float64 limit;
        # they are then treated as no batch effect.
        beta_batch = np.nan_to_num(beta[:, 1:], nan=0.0)
        corrected[np.ix_(row_indices, valid_idx)] = group_data - beta_batch @ group_x.T

    if n_single_batch or n_no_residual:
        logger.info(
            "%d feature(s) left uncorrected: %d with values in a single batch, %d missing a "
            "batch with no more values than observed batches",
            n_single_batch + n_no_residual,
            n_single_batch,
            n_no_residual,
        )
    return corrected


def _remove_batch_effect_dense(data: FloatArray, batch: npt.NDArray[np.intp]) -> FloatArray:
    """limma correction of a matrix without missing cells. See ``remove_batch_effect``."""
    if len(np.unique(batch)) < 2:
        logger.debug("Single batch input, returning copy")
        return data.copy()
    design, x_batch = _contrast_design(batch)
    beta = np.linalg.lstsq(design, data.T, rcond=None)[0].T
    # Coefficients overflow to NaN only for values near the float64 limit; they
    # are then treated as no batch effect.
    corrected: FloatArray = data - np.nan_to_num(beta[:, 1:], nan=0.0) @ x_batch.T
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
