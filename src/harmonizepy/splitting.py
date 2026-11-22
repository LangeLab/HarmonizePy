"""Correction of each group of features that share an affiliation.

Each group's sub-matrix holds its features and the columns of its blocks; it
is corrected by ComBat or limma and written into one output matrix. This has
the effect of R ``HarmonizR:::splitting`` and its rebuild step.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

from .affiliation import reduce_to_unique_groups
from .combat import COMBAT_MODES, _combat_with_reasons
from .limma_wrapper import _remove_batch_effect_with_reasons
from .validation import validate_combat_mode

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
    """Correct each group of features that share an affiliation, and assemble the result.

    Parameters
    ----------
    affiliation_list : list of tuple of int
        One affiliation per feature, as ``build_affiliation_list`` returns.
    data : DataFrame
        Features x samples matrix; missing cells are allowed. Not modified.
    batch_list : ndarray, shape (n_samples,)
        Batch label of each column.
    block_list : ndarray, shape (n_samples,)
        Block label of each column; equal to *batch_list* without blocking.
    algorithm : {"ComBat", "limma"}
        Correction engine.
    combat_mode : {1, 2, 3, 4}
        ComBat mode; ignored for limma.
    output_col_order : ndarray of intp, shape (n_samples,), optional
        For each column of *data*, as sorted, the output column it is written
        to, so that the result is in the order before sorting.
    output_columns : Index, optional
        Column labels of the output; defaults to ``data.columns``.
    data_np : ndarray, shape (n_features, n_samples), optional
        *data* as float64, reused instead of converting it again.

    Returns
    -------
    DataFrame
        The corrected matrix with the index of *data*. A feature with an empty
        affiliation is all missing. A feature alone in its group, or in a group
        that spans one batch, keeps its raw values with either engine: one
        batch has no batch effect to remove, ComBat cannot form a prior from
        one feature, and limma would remove the feature's own batch means with
        any true difference between batches. The log counts both kinds.

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

    Notes
    -----
    One output array is allocated and filled group by group; sub-matrices are
    extracted one group at a time. A group of one feature takes a shortcut
    without matrix indexing, and a group that spans the whole matrix is read
    without a copy.
    """
    corrected, _, _ = _correct_groups(
        affiliation_list,
        data,
        batch_list,
        block_list,
        algorithm,
        combat_mode,
        output_col_order=output_col_order,
        output_columns=output_columns,
        data_np=data_np,
    )
    return corrected


def _correct_groups(
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
) -> tuple[pd.DataFrame, npt.NDArray[np.str_], npt.NDArray[np.intp]]:
    """Do the work of ``splitting()`` and also return each feature's outcome.

    Returns the corrected matrix, a reason per feature (empty when corrected;
    ``"no_usable_batch"``, ``"single_batch"``, ``"alone_in_pattern"``, or a
    ComBat reason from ``_combat_validated``), and the size of each feature's
    group (0 for an empty affiliation).
    """
    if algorithm not in ("ComBat", "limma"):
        raise ValueError(f"algorithm must be 'ComBat' or 'limma', got {algorithm!r}.")
    if algorithm == "ComBat":
        validate_combat_mode(combat_mode)

    batch_arr = np.asarray(batch_list)
    block_arr = np.asarray(block_list)

    if data_np is None:
        data_np = data.to_numpy(dtype=np.float64)

    affil_to_features = reduce_to_unique_groups(affiliation_list)

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

    n_features, n_samples = data.shape
    output = np.full((n_features, n_samples), np.nan, dtype=np.float64)

    n_single_batch = 0
    n_single_feature = 0
    reasons = np.full(n_features, "", dtype="<U16")
    group_sizes = np.zeros(n_features, dtype=np.intp)

    for affil, row_indices in affil_to_features.items():
        if len(affil) == 0:
            reasons[row_indices] = "no_usable_batch"
            continue
        group_sizes[row_indices] = len(row_indices)

        col_indices = affil_to_cols[affil]
        target_col_indices = (
            col_indices if output_col_order is None else output_col_order[col_indices]
        )

        if len(row_indices) == 1:
            ri = row_indices[0]
            sub_data = data_np[ri, col_indices]
            sub_batch = batch_arr[col_indices]
            if len(np.unique(sub_batch)) < 2:
                n_single_batch += 1
                reasons[ri] = "single_batch"
            else:
                n_single_feature += 1
                reasons[ri] = "alone_in_pattern"
            output[ri, target_col_indices] = sub_data
            continue

        row_idx = np.array(row_indices, dtype=np.intp)
        spans_matrix = row_idx.size == n_features and np.array_equal(
            col_indices, np.arange(n_samples)
        )
        sub_data = data_np if spans_matrix else data_np[np.ix_(row_idx, col_indices)]
        sub_batch = batch_arr[col_indices]

        unique_batches = np.unique(sub_batch)
        if len(unique_batches) < 2:
            n_single_batch += len(row_idx)
            reasons[row_idx] = "single_batch"
            output[np.ix_(row_idx, target_col_indices)] = sub_data
            continue

        if algorithm == "limma":
            corrected, reasons[row_idx] = _remove_batch_effect_with_reasons(sub_data, sub_batch)
        else:
            corrected, reasons[row_idx] = _combat_with_reasons(
                sub_data, sub_batch, **COMBAT_MODES[combat_mode]
            )

        output[np.ix_(row_idx, target_col_indices)] = corrected

    n_uncorrected = n_single_batch + n_single_feature
    if n_uncorrected > 0:
        logger.info(
            "%d feature(s) passed through without correction "
            "(%d in single-batch groups, %d alone in their pattern)",
            n_uncorrected,
            n_single_batch,
            n_single_feature,
        )

    corrected_frame = pd.DataFrame(
        output,
        index=data.index,
        columns=data.columns if output_columns is None else output_columns,
        copy=False,
    )
    return corrected_frame, reasons, group_sizes
