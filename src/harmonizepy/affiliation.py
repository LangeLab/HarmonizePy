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

import itertools

import numpy as np
import numpy.typing as npt
import pandas as pd


def build_affiliation_list(
    data: pd.DataFrame,
    batch_list: npt.NDArray[np.integer],
    block_list: npt.NDArray[np.integer],
    needed_values: int,
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
    mat = data.values
    n_features = mat.shape[0]

    batch_arr = np.asarray(batch_list)
    block_arr = np.asarray(block_list)

    unique_blocks = np.unique(block_arr)

    # Pre-compute: for each block, the sample indices per original batch
    block_info: list[list[np.ndarray]] = []
    for blk in unique_blocks:
        blk_mask = block_arr == blk
        batches_in_block = np.unique(batch_arr[blk_mask])
        batch_indices = [
            np.where((batch_arr == b) & blk_mask)[0]
            for b in batches_in_block
        ]
        block_info.append(batch_indices)

    notna = ~np.isnan(mat)

    affiliation_list: list[tuple[int, ...]] = []

    for i in range(n_features):
        row_notna = notna[i]
        blocks_present: list[int] = []

        for blk_idx, batch_indices in enumerate(block_info):
            all_ok = True
            for idx in batch_indices:
                if row_notna[idx].sum() < needed_values:
                    all_ok = False
                    break
            if all_ok:
                blocks_present.append(int(unique_blocks[blk_idx]))

        affiliation_list.append(tuple(sorted(blocks_present)))

    return affiliation_list


def reduce_to_unique_groups(
    affiliation_list: list[tuple[int, ...]],
) -> dict[tuple[int, ...], list[int]]:
    """Group feature indices by their shared affiliation pattern.

    Parameters
    ----------
    affiliation_list : list[tuple[int, ...]]
        One tuple per feature from :func:`build_affiliation_list`.

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
    affiliation — the one reachable by removing the fewest batches — and
    replaces the singleton's affiliation with that pattern.  Empty
    affiliations (features with no data anywhere) are left unchanged.

    This mirrors R ``HarmonizR:::unique_removal`` (``ur`` parameter,
    default ``TRUE`` in R HarmonizR v1.8.0).

    Parameters
    ----------
    affiliation_list : list[tuple[int, ...]]
        One tuple per feature from :func:`build_affiliation_list`.
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
        removal to help — in practice this cannot happen when n_features > 1.

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

    # Collect the non-unique (shared) patterns — used as rescue targets
    non_unique = {affil for affil, cnt in counts.items() if cnt > 1}

    if not non_unique:
        # Nothing to rescue to; leave list unchanged
        return result

    for i, affil in enumerate(result):
        if not affil or affil in non_unique:
            continue  # empty or already non-unique — nothing to do

        # Find the reachable non-unique pattern with fewest blocks removed.
        # Search all non-empty subsets of affil (in descending size order)
        # until we hit one that is non-unique.
        best = _find_best_crop(frozenset(affil), non_unique)
        if best is not None:
            result[i] = best

    return result


def _find_best_crop(
    affil_set: frozenset[int],
    non_unique: set[tuple[int, ...]],
) -> tuple[int, ...] | None:
    """Return the largest subset of *affil_set* that is in *non_unique*.

    Searches by decreasing subset size so that the first match found
    removes the minimum number of batches (greedy, matching R's approach).
    Returns ``None`` if no non-empty subset matches.
    """
    affil_sorted = tuple(sorted(affil_set))
    n = len(affil_sorted)

    # Try subsets from size n-1 down to 1
    for size in range(n - 1, 0, -1):
        for combo in itertools.combinations(affil_sorted, size):
            if combo in non_unique:
                return combo
    return None
