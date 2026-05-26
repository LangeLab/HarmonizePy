"""Batch column sorting strategies for HarmonizePy.

Sorting reorders **batch columns** (and their samples) in the data matrix
so that similar batches become neighbors.  This only affects output when
combined with blocking. Adjacent batches are grouped into blocks, so
the ordering determines which batches share a block and therefore how
much data is lost when features are absent in one block member.

Calling ``sort_batches`` without subsequently applying blocking has
no effect on adjusted values.

This mirrors R ``HarmonizR:::sorting``.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

from .validation import _VALID_SORT_STRATEGIES

logger = logging.getLogger(__name__)


def sort_batches(
    data: pd.DataFrame,
    batch_list: npt.NDArray[np.integer],
    strategy: str,
    needed_values: int,
) -> tuple[pd.DataFrame, npt.NDArray[np.integer], npt.NDArray[np.intp]]:
    """Reorder data columns so that similar batches are adjacent.

    Computes a permutation of batches using *strategy*, then reorders the
    sample columns (and the accompanying *batch_list*) to match.  Call
    this before blocking; call the returned ``col_order`` inverse after
    ``rebuild`` to restore the original column ordering.

    .. note::
        Sorting without subsequent blocking leaves adjusted values
        unchanged.  The function is a no-op on output when ``block=None``.

    Parameters
    ----------
    data : DataFrame
        Features x samples matrix.  Columns are sample IDs; rows are
        features.  May contain NaN.
    batch_list : ndarray, shape (n_samples,)
        Integer batch label per sample.
    strategy : str
        ``"sparsity"``, ``"jaccard"``, or ``"seriation"``.
    needed_values : int
        Minimum non-NaN values per batch per feature for the feature to
        count as "present" in that batch when building the binary
        presence matrix used by all strategies.

    Returns
    -------
    sorted_data : DataFrame
        Copy of *data* with columns reordered by the new batch order.
    sorted_batch_list : ndarray
        *batch_list* reordered to match the new column order.
    col_order : ndarray of int, shape (n_samples,)
        Original column indices for each position in *sorted_data*.
        To restore the original column order after rebuild, use
        ``result.iloc[:, np.argsort(col_order)]``.

    Raises
    ------
    ValueError
        If *strategy* is not one of the three accepted values.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from harmonizepy.sorting import sort_batches
    >>> rng = np.random.default_rng(0)
    >>> data = pd.DataFrame(
    ...     rng.standard_normal((10, 6)),
    ...     columns=["s1", "s2", "s3", "s4", "s5", "s6"],
    ... )
    >>> batch = np.array([1, 1, 2, 2, 3, 3])
    >>> data.iloc[5:, 4:] = float("nan")  # batch 3 has fewer present features
    >>> sd, sb, co = sort_batches(data, batch, "sparsity", needed_values=2)
    >>> sd.shape
    (10, 6)
    >>> list(sd.columns[:2])  # batch 3 (sparsest) is now first
    ['s4', 's5']
    """
    if strategy not in _VALID_SORT_STRATEGIES:
        raise ValueError(
            f"sort strategy must be one of {sorted(_VALID_SORT_STRATEGIES)!r}, got {strategy!r}. "
            f"Use sort=None to disable sorting."
        )

    unique_batches = _unique_batches_ordered(batch_list)
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
    else:  # seriation
        batch_order = _seriation_order(presence)

    ordered_batches = unique_batches[batch_order]
    col_order = _column_order(batch_list, ordered_batches)

    cols = data.columns.tolist()
    sorted_data = data[[cols[i] for i in col_order]].copy()
    sorted_batch_list = batch_list[col_order]

    return sorted_data, sorted_batch_list, col_order


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _unique_batches_ordered(batch_list: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
    """Return unique batch IDs in first-appearance order."""
    seen: set[int] = set()
    result: list[int] = []
    for b in batch_list:
        if b not in seen:
            seen.add(b)
            result.append(int(b))
    return np.array(result, dtype=batch_list.dtype)


def _build_presence_matrix(
    data: pd.DataFrame,
    batch_list: npt.NDArray[np.integer],
    unique_batches: npt.NDArray[np.integer],
    needed_values: int,
) -> npt.NDArray[np.bool_]:
    """Binary (n_features x n_batches) presence matrix.

    ``presence[i, j] = True`` iff batch ``unique_batches[j]`` has at least
    *needed_values* non-NaN observations for feature *i*.
    """
    values = data.to_numpy(dtype=float, na_value=np.nan)
    n_features = values.shape[0]
    n_batches = len(unique_batches)
    presence = np.empty((n_features, n_batches), dtype=np.bool_)
    for j, bid in enumerate(unique_batches):
        mask = batch_list == bid
        valid_counts = np.sum(~np.isnan(values[:, mask]), axis=1)
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
    """Greedy nearest-neighbour Jaccard ordering.

    Computes the pairwise Jaccard similarity matrix between batches via
    vectorised integer matrix multiplication, then traverses batches
    starting from the one with the highest total similarity to all others,
    always picking the most similar unvisited batch next.
    """
    p = presence.astype(np.int32)  # (n_features, n_batches)
    # intersection[i, j] = number of features present in both batch i and j
    intersection = (p.T @ p).astype(np.float64)  # (n_batches, n_batches)
    col_sums = presence.sum(axis=0).astype(np.float64)  # (n_batches,)
    # union[i, j] = |A| + |B| - |A ∩ B|
    union = col_sums[:, None] + col_sums[None, :] - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        jaccard = np.where(union > 0, intersection / union, 0.0)

    n = jaccard.shape[0]
    if n == 1:
        return np.array([0], dtype=np.intp)

    # Start from the batch most similar (on average) to all others.
    # Subtract the diagonal (self-similarity) rather than a constant 1: empty
    # batches have jaccard[i,i]=0 due to the 0/0 guard above, not 1.
    sim_sums = jaccard.sum(axis=1) - np.diag(jaccard)
    start = int(np.argmax(sim_sums))

    visited = np.zeros(n, dtype=np.bool_)
    order: list[int] = [start]
    visited[start] = True

    for _ in range(n - 1):
        current = order[-1]
        sims = jaccard[current].copy()
        sims[visited] = -1.0
        nxt = int(np.argmax(sims))
        order.append(nxt)
        visited[nxt] = True

    return np.array(order, dtype=np.intp)


def _seriation_order(presence: npt.NDArray[np.bool_]) -> npt.NDArray[np.intp]:
    """PCA-based seriation ordering (matches R ``seriation::seriate`` PCA method).

    Performs PCA on the batch presence vectors (batches as observations,
    features as variables) and orders batches ascending by their first-PC
    score.  This replicates R's ``seriation::seriate(binary_df, margin=2)``
    default behaviour which uses ``prcomp(t(x), scale=FALSE)`` internally.
    Sign is anchored so the batch with the largest absolute PC1 score is
    positive (matching R's ``prcomp`` sign convention).
    """
    p = presence.T.astype(np.float64)  # (n_batches, n_features)
    n = p.shape[0]

    if n <= 2:
        return np.arange(n, dtype=np.intp)

    # Centre columns (features) to mirror R's prcomp(t(binary_df), scale=FALSE)
    p -= p.mean(axis=0)

    # SVD into PC1 scores for each batch
    U, s, _Vt = np.linalg.svd(p, full_matrices=False)  # noqa: N806 (standard SVD notation)
    pc1 = U[:, 0] * s[0]

    # Sign convention: make the PC1 direction point "toward" the batch with
    # the largest absolute score being positive (mirrors R prcomp convention).
    if pc1[np.argmax(np.abs(pc1))] < 0:
        pc1 = -pc1

    return np.argsort(pc1, kind="stable")
