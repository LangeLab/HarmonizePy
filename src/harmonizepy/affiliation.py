"""Affiliation logic: build and reduce per-feature batch affiliations.

The affiliation list records, for each feature (row), which blocks have
sufficient non-missing data.  ``reduce_to_unique_groups`` then collapses
features sharing identical affiliations into groups so that
``splitting.py`` can extract one sub-matrix per group.

In the R HarmonizR source, ``unique()`` on the affiliation list *is* the
binary reduction step - there is no separate operation.  This module
combines both responsibilities.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)


def _assemble_affiliations(
    unique_blocks: npt.NDArray[np.integer],
    block_oks: list[npt.NDArray[np.bool_]],
    n_features: int,
) -> list[tuple[int, ...]]:
    """Assemble per-feature affiliation tuples from per-block masks.

    Repeated block-membership patterns are common on fragmented datasets, so
    cache the tuple reconstruction by mask pattern instead of rebuilding the
    same Python tuple for every matching feature.
    """
    if not block_oks:
        return [()] * n_features

    n_blocks = len(unique_blocks)

    if n_blocks <= 64:
        codes = np.zeros(n_features, dtype=np.uint64)
        for bit_index, mask in enumerate(block_oks):
            codes |= mask.astype(np.uint64) << np.uint64(bit_index)

        cache: dict[int, tuple[int, ...]] = {0: ()}
        affiliation_list: list[tuple[int, ...]] = []
        for code in codes.tolist():
            affil = cache.get(code)
            if affil is None:
                bits = int(code)
                affil = tuple(
                    int(unique_blocks[block_index])
                    for block_index in range(n_blocks)
                    if bits & (1 << block_index)
                )
                cache[code] = affil
            affiliation_list.append(affil)
        return affiliation_list

    packed = np.packbits(np.column_stack(block_oks), axis=1)
    cache_bytes: dict[bytes, tuple[int, ...]] = {}
    affiliation_list = []
    for row_index, packed_row in enumerate(packed):
        key = packed_row.tobytes()
        affil = cache_bytes.get(key)
        if affil is None:
            affil = tuple(
                int(unique_blocks[block_index])
                for block_index, mask in enumerate(block_oks)
                if mask[row_index]
            )
            cache_bytes[key] = affil
        affiliation_list.append(affil)
    return affiliation_list


def build_affiliation_list(
    data: pd.DataFrame,
    batch_list: npt.NDArray[np.integer],
    block_list: npt.NDArray[np.integer],
    needed_values: int,
    *,
    data_np: npt.NDArray[np.float64] | None = None,
) -> list[tuple[int, ...]]:
    """Determine per-feature batch/block affiliation.

    For each feature (row), identifies the blocks where the feature has
    at least *needed_values* non-missing observations in every original
    batch contained within that block.

    Parameters
    ----------
    data : DataFrame
        Features x samples matrix (may contain NaN).
    batch_list : ndarray, shape (n_samples,)
        Integer batch label per sample.
    block_list : ndarray, shape (n_samples,)
        Integer block label per sample (equals *batch_list* when no
        blocking is applied).
    needed_values : int
        Minimum non-NA values per batch within a block.
    data_np : ndarray or None, keyword-only
        Optional precomputed float64 view or copy of *data* with shape
        ``(n_features, n_samples)``. When provided, reused instead of
        converting *data* again inside this function.

    Returns
    -------
    list[tuple[int, ...]]
        One tuple per feature.  Each tuple contains the sorted block IDs
        where the feature has sufficient data.  An empty tuple means
        the feature has insufficient data everywhere.

    Raises
    ------
    ValueError
        Propagated from NumPy if *batch_list* or *block_list* cannot be
        broadcast against the columns of *data*.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from harmonizepy.affiliation import build_affiliation_list
    >>> data = pd.DataFrame({"s1": [1.0, np.nan], "s2": [2.0, 3.0], "s3": [4.0, 5.0]})
    >>> batch = np.array([1, 1, 2])
    >>> build_affiliation_list(data, batch, batch, needed_values=1)
    [(1, 2), (2,)]
    """
    batch_arr = np.asarray(batch_list)
    block_arr = np.asarray(block_list)

    unique_blocks = np.unique(block_arr)

    # Pre-compute: for each block, the sample indices per original batch
    block_info: list[list[np.ndarray]] = []
    for blk in unique_blocks:
        blk_mask = block_arr == blk
        batches_in_block = np.unique(batch_arr[blk_mask])
        batch_indices = [np.where((batch_arr == b) & blk_mask)[0] for b in batches_in_block]
        block_info.append(batch_indices)

    if data_np is None:
        data_np = data.to_numpy(dtype=np.float64)

    notna = ~np.isnan(data_np)

    # Vectorised per-block check across all features at once.
    # For each block, compute a boolean array indicating whether each
    # feature has >= needed_values non-NA in every batch within that block.
    block_oks: list[np.ndarray] = []
    for batch_indices in block_info:
        batch_oks = [notna[:, idx].sum(axis=1) >= needed_values for idx in batch_indices]
        block_oks.append(np.all(batch_oks, axis=0))

    n_features = data.shape[0]
    affiliation_list = _assemble_affiliations(unique_blocks, block_oks, n_features)

    n_empty = sum(1 for a in affiliation_list if len(a) == 0)
    logger.debug(
        "Affiliation: %d features across %d blocks; %d with insufficient data",
        n_features,
        len(unique_blocks),
        n_empty,
    )
    return affiliation_list


def reduce_to_unique_groups(
    affiliation_list: list[tuple[int, ...]],
) -> dict[tuple[int, ...], list[int]]:
    """Group feature indices by their shared affiliation pattern.

    Parameters
    ----------
    affiliation_list : list[tuple[int, ...]]
        One tuple per feature from ``build_affiliation_list``.

    Returns
    -------
    dict[tuple[int, ...], list[int]]
        Mapping from each unique affiliation to the list of feature
        row indices that share it.  Iteration order matches first
        appearance.

    Raises
    ------
    TypeError
        If *affiliation_list* contains non-hashable elements.

    Examples
    --------
    >>> from harmonizepy.affiliation import reduce_to_unique_groups
    >>> groups = reduce_to_unique_groups([(1, 2), (1,), (1, 2)])
    >>> groups == {(1, 2): [0, 2], (1,): [1]}
    True
    """
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, affil in enumerate(affiliation_list):
        groups.setdefault(affil, []).append(i)
    return groups


def remove_unique_combinations(
    affiliation_list: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """Rescue singleton features by cropping their affiliation to a shared pattern.

    A feature whose batch-presence combination is unique (no other feature
    shares it) would end up alone in a sub-matrix.  ComBat and limma cannot
    adjust a single feature, so it would otherwise be dropped.

    For each such singleton, this function finds the closest non-unique
    affiliation (the one reachable by removing the fewest batches) and
    replaces the singleton's affiliation with that pattern.  Empty
    affiliations (features with no data anywhere) are left unchanged.

    This mirrors R ``HarmonizR:::unique_removal`` (``ur`` parameter,
    default ``TRUE`` in R HarmonizR v1.8.0).

    Parameters
    ----------
    affiliation_list : list[tuple[int, ...]]
        One tuple per feature from ``build_affiliation_list``.
        Each tuple contains sorted block IDs where the feature has
        sufficient data.

    Returns
    -------
    list[tuple[int, ...]]
        New affiliation list of the same length.  Singleton tuples are
        replaced with a subset of their original blocks that at least one
        other feature also shares.  Non-singleton and empty tuples are
        unchanged.

    Raises
    ------
    ValueError
        If all non-empty affiliations are unique (no shared pattern exists
        to crop to).  This indicates a dataset too fragmented for unique
        removal to help. In practice this cannot happen when n_features > 1.

    Examples
    --------
    >>> from harmonizepy.affiliation import remove_unique_combinations
    >>> # Feature 0 is a singleton; features 1 and 2 share (1, 2)
    >>> result = remove_unique_combinations([(1, 2, 3), (1, 2), (1, 2)])
    >>> result[0]  # cropped to the nearest shared pattern
    (1, 2)
    >>> result[1:]  # unchanged
    [(1, 2), (1, 2)]
    """
    result = list(affiliation_list)

    # Count how many features share each non-empty affiliation
    counts: dict[tuple[int, ...], int] = {}
    for affil in result:
        if affil:
            counts[affil] = counts.get(affil, 0) + 1

    # Collect the non-unique (shared) patterns used as rescue targets
    non_unique = {affil for affil, cnt in counts.items() if cnt > 1}

    if not non_unique:
        logger.debug("No shared patterns available for rescue")
        return result

    rescued = 0
    for i, affil in enumerate(result):
        if not affil or affil in non_unique:
            continue

        best = _find_best_crop(frozenset(affil), non_unique)
        if best is not None:
            result[i] = best
            rescued += 1

    if rescued > 0:
        logger.debug("Unique removal rescued %d feature(s)", rescued)
    return result


def _find_best_crop(
    affil_set: frozenset[int],
    non_unique: set[tuple[int, ...]],
) -> tuple[int, ...] | None:
    """Return the largest subset of *affil_set* that is in *non_unique*.

    Iterates over the non-unique patterns directly rather than generating
    all subsets of *affil_set*. For datasets with few non-unique patterns
    and long affiliations this is orders of magnitude faster.

    Returns ``None`` if no non-empty subset matches.
    """
    best = None
    for pattern in non_unique:
        if set(pattern).issubset(affil_set) and len(pattern) < len(affil_set):
            if best is None or len(pattern) > len(best):
                best = pattern
    return best
