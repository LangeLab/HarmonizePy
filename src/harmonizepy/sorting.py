"""Batch orders that place similar batches next to each other before blocking.

Blocking groups neighbouring batches, so the order decides which batches share
a block, and how many values are lost where a feature is missing from one
batch of a block. Without blocking the order has no effect on corrected
values. The strategies follow R ``HarmonizR:::sorting``; how each of the three
orders differs from R's is listed in ``assets/FEATURE_PARITY.md``.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

from .blocking import _unique_ordered
from .validation import _VALID_SORT_STRATEGIES

logger = logging.getLogger(__name__)


def sort_batches(
    data: pd.DataFrame,
    batch_list: npt.NDArray[np.integer],
    strategy: str,
    needed_values: int,
) -> tuple[pd.DataFrame, npt.NDArray[np.integer], npt.NDArray[np.intp]]:
    """Reorder the columns so that similar batches are adjacent.

    A feature counts as present in a batch when it has at least
    *needed_values* observed values there. ``"sparsity"`` puts batches with
    more present features first; ``"jaccard"`` chains batches by the overlap of
    their present features; ``"seriation"`` orders them by the first principal
    component of their presence vectors. Samples keep their relative order
    within a batch.

    Parameters
    ----------
    data : DataFrame
        Features x samples matrix; missing cells are allowed. Not modified.
    batch_list : ndarray, shape (n_samples,)
        Batch label of each column.
    strategy : {"sparsity", "jaccard", "seriation"}
        Ordering strategy.
    needed_values : int
        Observed values a feature needs in a batch to count as present.

    Returns
    -------
    sorted_data : DataFrame
        A new frame with the columns in the new order.
    sorted_batch_list : ndarray, shape (n_samples,)
        *batch_list* in the new order.
    col_order : ndarray of intp, shape (n_samples,)
        The input column index at each position of *sorted_data*;
        ``np.argsort(col_order)`` restores the input order.

    Raises
    ------
    ValueError
        If *strategy* is not one of the three names.

    Examples
    --------
    Batch 1 has values for 3 of 10 features, so ``"sparsity"`` moves it last:

    >>> import numpy as np, pandas as pd
    >>> from harmonizepy.sorting import sort_batches
    >>> data = pd.DataFrame(
    ...     np.arange(60.0).reshape(10, 6), columns=["s1", "s2", "s3", "s4", "s5", "s6"]
    ... )
    >>> data.iloc[3:, 0:2] = np.nan
    >>> sorted_data, sorted_batches, col_order = sort_batches(
    ...     data, np.array([1, 1, 2, 2, 3, 3]), "sparsity", needed_values=2
    ... )
    >>> list(sorted_data.columns)
    ['s3', 's4', 's5', 's6', 's1', 's2']
    >>> col_order.tolist()
    [2, 3, 4, 5, 0, 1]
    """
    if strategy not in _VALID_SORT_STRATEGIES:
        raise ValueError(
            f"sort strategy must be one of {sorted(_VALID_SORT_STRATEGIES)!r}, got {strategy!r}. "
            f"Use sort=None to disable sorting."
        )

    unique_batches = np.array(_unique_ordered(batch_list), dtype=batch_list.dtype)
    presence = _build_presence_matrix(data, batch_list, unique_batches, needed_values)
    logger.debug(
        "Sorting %d batches by '%s' (presence matrix: %d features x %d batches)",
        len(unique_batches),
        strategy,
        presence.shape[0],
        presence.shape[1],
    )

    if strategy == "sparsity":
        batch_order = _sparsity_order(presence)
    elif strategy == "jaccard":
        batch_order = _jaccard_order(presence)
    else:
        batch_order = _seriation_order(presence)

    ordered_batches = unique_batches[batch_order]
    col_order = _column_order(batch_list, ordered_batches)

    cols = data.columns.tolist()
    sorted_data = data[[cols[i] for i in col_order]]
    sorted_batch_list = batch_list[col_order]

    return sorted_data, sorted_batch_list, col_order


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_presence_matrix(
    data: pd.DataFrame,
    batch_list: npt.NDArray[np.integer],
    unique_batches: npt.NDArray[np.integer],
    needed_values: int,
) -> npt.NDArray[np.bool_]:
    """Return the (n_features, n_batches) matrix of features present in each batch.

    Entry (i, j) is true when feature *i* has at least *needed_values*
    observed values in batch ``unique_batches[j]``.
    """
    values = data.to_numpy(dtype=np.float64)
    notna = ~np.isnan(values)
    n_features = values.shape[0]
    n_batches = len(unique_batches)
    presence = np.empty((n_features, n_batches), dtype=np.bool_)
    for j, bid in enumerate(unique_batches):
        mask = batch_list == bid
        valid_counts = notna[:, mask].sum(axis=1)
        presence[:, j] = valid_counts >= needed_values
    return presence


def _column_order(
    batch_list: npt.NDArray[np.integer],
    ordered_batches: npt.NDArray[np.integer],
) -> npt.NDArray[np.intp]:
    """Map the new batch order to a full column permutation.

    For each batch ID in *ordered_batches* (in order), appends the original
    column indices belonging to that batch.  Samples within a batch retain
    their original relative order.
    """
    order: list[int] = []
    for bid in ordered_batches:
        indices = np.where(batch_list == bid)[0]
        order.extend(indices.tolist())
    return np.array(order, dtype=np.intp)


def _sparsity_order(presence: npt.NDArray[np.bool_]) -> npt.NDArray[np.intp]:
    """Sort batches descending by feature completeness count.

    Batches with the most present features come first; batches with the
    fewest come last.  This mirrors R HarmonizR's ``find_na()``-based
    sort which orders by ascending NA count (= descending completeness),
    ensuring similar-completeness batches become neighbours under blocking.
    """
    completeness = presence.sum(axis=0).astype(np.int64)
    return np.argsort(-completeness, kind="stable")


def _jaccard_order(presence: npt.NDArray[np.bool_]) -> npt.NDArray[np.intp]:
    """Order batches by a greedy nearest-neighbour chain on Jaccard similarity.

    The similarity of two batches is the number of features present in both
    divided by the number present in either, and 0 when neither has any. The
    chain starts at the batch with the largest total similarity to the others
    (the diagonal left out, since an empty batch has a self-similarity of 0, not
    1) and moves each time to the most similar batch not yet visited.
    """
    counts = presence.astype(np.int32)
    intersection = (counts.T @ counts).astype(np.float64)
    present_per_batch = presence.sum(axis=0).astype(np.float64)
    union = present_per_batch[:, None] + present_per_batch[None, :] - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        jaccard = np.where(union > 0, intersection / union, 0.0)

    n_batches = jaccard.shape[0]
    if n_batches == 1:
        return np.array([0], dtype=np.intp)

    similarity_to_others = jaccard.sum(axis=1) - np.diag(jaccard)
    start = int(np.argmax(similarity_to_others))
    visited = np.zeros(n_batches, dtype=np.bool_)
    order: list[int] = [start]
    visited[start] = True
    for _ in range(n_batches - 1):
        candidates = jaccard[order[-1]].copy()
        candidates[visited] = -1.0
        nearest = int(np.argmax(candidates))
        order.append(nearest)
        visited[nearest] = True
    return np.array(order, dtype=np.intp)


def _seriation_order(presence: npt.NDArray[np.bool_]) -> npt.NDArray[np.intp]:
    """Order batches by their score on the first principal component of presence.

    Batches are the observations and features the variables; the presence
    matrix is centred per feature and not scaled, as ``prcomp(scale = FALSE)``
    does. The component is the leading eigenvector of the batches' Gram matrix,
    which is computed exactly in integers, so the order does not depend on the
    order of the features. Scores are rounded to 12 decimals of the unit
    eigenvector before ordering, so batches whose scores differ only by
    rounding tie, and ties keep the input order. The sign is fixed so that the
    first batch with the largest absolute score is positive. When the two
    largest eigenvalues are equal the component is not unique, and the one the
    eigensolver returns for that exact matrix is used. With two batches or
    fewer the input order is returned. R HarmonizR's seriation uses
    ``seriation::seriate``, which gives a different order in general.
    """
    n_batches = presence.shape[1]
    if n_batches <= 2:
        return np.arange(n_batches, dtype=np.intp)
    counts = presence.astype(np.int64)
    # n_batches times the centred presence, kept in integers so the Gram
    # matrix below is exact and identical for any order of the features.
    scaled = n_batches * counts - counts.sum(axis=1, keepdims=True)
    gram = (scaled.T @ scaled).astype(np.float64)
    _, vectors = np.linalg.eigh(gram)
    scores = np.round(vectors[:, -1], 12)
    largest = int(np.argmax(np.abs(scores)))
    if scores[largest] < 0:
        scores = -scores
    return np.argsort(scores, kind="stable")
