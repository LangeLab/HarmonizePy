"""Batch correction by a linear model, as R ``limma::removeBatchEffect`` does.

Batch is encoded as sum-to-zero contrasts (R's ``contr.sum``), the design is
an intercept plus those contrasts, each feature is fitted by least squares,
and the fitted batch component is subtracted. Features that share a pattern
of observed cells are fitted together.

References
----------
Ritchie ME et al. limma powers differential expression analyses for
RNA-sequencing and microarray studies. Nucleic Acids Research 43(7):e47,
2015. doi:10.1093/nar/gkv007
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

    Missing cells are left out of each feature's least-squares fit, as R
    ``limma::removeBatchEffect`` does, and stay missing in the output. A
    feature missing from a whole batch is handled differently from R; see
    Returns.

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
        level. A feature with values in one batch only, or with no more values
        than batches it is observed in, keeps its raw values: its fit would
        leave no residual and collapse it to one level. The log counts both.
        Each feature is fitted alone, so a single feature is corrected, and a
        batch of one sample takes that sample's whole deviation into its batch
        term, so the sample lands on the feature's mean of batch means.

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
    corrected, _ = _remove_batch_effect_with_reasons(data, batch)
    return corrected


def _remove_batch_effect_with_reasons(
    data: FloatArray, batch: npt.ArrayLike
) -> tuple[FloatArray, npt.NDArray[np.str_]]:
    """Validate and correct as ``remove_batch_effect()`` does, and return why each feature was left raw.

    The reason is an empty string for a corrected feature, otherwise
    ``"no_values"``, ``"single_batch"``, or ``"too_few_values"`` (no more values
    than batches it is observed in).
    """
    data = np.asarray(data, dtype=np.float64)
    batch_int = batch_codes(batch)
    validate_engine_input(data, batch_int)

    if not np.isnan(data).any():
        return _remove_batch_effect_dense(data, batch_int)
    if np.isnan(data).all():
        logger.debug("All-NaN input, returning copy")
        return data.copy(), np.full(data.shape[0], "no_values", dtype="<U16")
    return _remove_batch_effect_nan(data, batch_int)


def _remove_batch_effect_nan(
    data: FloatArray, batch: npt.NDArray[np.intp]
) -> tuple[FloatArray, npt.NDArray[np.str_]]:
    """limma correction fitted on each feature's observed cells.

    Features that share a pattern of observed cells are fitted together, with
    contrasts over the batches they are observed in. A feature with values in
    one batch only keeps its raw values, as does a feature whose values do not
    outnumber its observed batches, since its fit would leave no residual and
    collapse its values to one level. Returns the matrix and the reasons of
    ``_remove_batch_effect_with_reasons``.
    """
    reasons = np.full(data.shape[0], "", dtype="<U16")
    n_batch = len(np.unique(batch))
    if n_batch < 2:
        logger.debug("Single batch input, returning copy")
        reasons[:] = "single_batch"
        return data.copy(), reasons
    design, x_batch = _contrast_design(batch)

    corrected = data.copy()
    n_single_batch = n_no_residual = n_no_values = 0
    groups = group_valid_rows(data)
    for row_indices in groups.row_groups:
        valid = groups.valid_masks[row_indices[0]]
        observed_batches = np.unique(batch[valid])
        if observed_batches.size == 0:
            n_no_values += len(row_indices)
            reasons[row_indices] = "no_values"
            continue
        if observed_batches.size < 2:
            n_single_batch += len(row_indices)
            reasons[row_indices] = "single_batch"
            continue
        if valid.sum() <= observed_batches.size:
            n_no_residual += len(row_indices)
            reasons[row_indices] = "too_few_values"
            continue
        if observed_batches.size == n_batch:
            group_design, group_x = design[valid, :], x_batch[valid, :]
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
            "%d feature(s) left uncorrected: %d with values in a single batch, %d with no "
            "more values than observed batches",
            n_single_batch + n_no_residual,
            n_single_batch,
            n_no_residual,
        )
    if n_no_values:
        logger.debug("%d feature(s) with no value stay missing", n_no_values)
    return corrected, reasons


def _remove_batch_effect_dense(
    data: FloatArray, batch: npt.NDArray[np.intp]
) -> tuple[FloatArray, npt.NDArray[np.str_]]:
    """limma correction of a matrix without missing cells. See ``remove_batch_effect``."""
    n_batch = len(np.unique(batch))
    if n_batch < 2:
        logger.debug("Single batch input, returning copy")
        return data.copy(), np.full(data.shape[0], "single_batch", dtype="<U16")
    if data.shape[1] <= n_batch:
        logger.info("%d feature(s) left uncorrected: no more values than batches", data.shape[0])
        return data.copy(), np.full(data.shape[0], "too_few_values", dtype="<U16")
    design, x_batch = _contrast_design(batch)
    beta = np.linalg.lstsq(design, data.T, rcond=None)[0].T
    # Coefficients overflow to NaN only for values near the float64 limit; they
    # are then treated as no batch effect.
    corrected: FloatArray = data - np.nan_to_num(beta[:, 1:], nan=0.0) @ x_batch.T
    return corrected, np.full(data.shape[0], "", dtype="<U16")


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
