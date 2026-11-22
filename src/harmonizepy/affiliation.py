"""Affiliations: the blocks in which each feature has enough values.

A feature's affiliation is the sorted tuple of block labels in which every
batch holds at least ``needed_values`` observed values of the feature.
Features with the same affiliation form a group, and each group is corrected
as one sub-matrix. Unique removal moves a feature whose affiliation no other
feature shares into the largest shared affiliation within its own, as R
HarmonizR's ``ur = TRUE`` does.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)


def _assemble_affiliations(
    unique_blocks: npt.NDArray[np.integer],
    block_oks: list[npt.NDArray[np.bool_]],
    n_features: int,
) -> list[tuple[int, ...]]:
    """Return one affiliation tuple per feature from the per-block masks.

    Features often share a pattern, so each distinct pattern is turned into a
    tuple once: by a bit code for up to 64 blocks, by packed bytes beyond.
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
    """Return each feature's affiliation: the blocks where it has enough values.

    A block counts for a feature when every batch inside the block holds at
    least *needed_values* observed values of it.

    Parameters
    ----------
    data : DataFrame
        Features x samples matrix; missing cells are allowed.
    batch_list : ndarray, shape (n_samples,)
        Batch label of each column.
    block_list : ndarray, shape (n_samples,)
        Block label of each column; equal to *batch_list* without blocking.
    needed_values : int
        Minimum observed values per batch.
    data_np : ndarray, shape (n_features, n_samples), optional
        *data* as float64, reused instead of converting *data* again.

    Returns
    -------
    list of tuple of int
        One sorted tuple of block labels per feature, in row order. An empty
        tuple means no block qualifies.

    Raises
    ------
    ValueError
        If *batch_list* or *block_list* does not have one label per column.

    Examples
    --------
    The second feature has one value in batch 2, below ``needed_values=2``:

    >>> import numpy as np, pandas as pd
    >>> from harmonizepy.affiliation import build_affiliation_list
    >>> data = pd.DataFrame(
    ...     {"s1": [1.0, 2.0], "s2": [1.5, 2.5], "s3": [3.0, np.nan], "s4": [3.5, 4.0]}
    ... )
    >>> batch = np.array([1, 1, 2, 2])
    >>> build_affiliation_list(data, batch, batch, needed_values=2)
    [(1, 2), (1,)]
    """
    batch_arr = np.asarray(batch_list)
    block_arr = np.asarray(block_list)
    n_samples = data.shape[1]
    if batch_arr.shape != (n_samples,) or block_arr.shape != (n_samples,):
        raise ValueError(
            f"batch_list and block_list need one label per column ({n_samples}); "
            f"got {batch_arr.shape} and {block_arr.shape}."
        )

    unique_blocks = np.unique(block_arr)

    block_info: list[list[np.ndarray]] = []
    for blk in unique_blocks:
        blk_mask = block_arr == blk
        batches_in_block = np.unique(batch_arr[blk_mask])
        batch_indices = [np.where((batch_arr == b) & blk_mask)[0] for b in batches_in_block]
        block_info.append(batch_indices)

    if data_np is None:
        data_np = data.to_numpy(dtype=np.float64)

    notna = ~np.isnan(data_np)
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
    """Group feature row indices by affiliation.

    Parameters
    ----------
    affiliation_list : list of tuple of int
        One affiliation per feature, as ``build_affiliation_list`` returns.

    Returns
    -------
    dict
        Each distinct affiliation, in order of first appearance, mapped to the
        row indices of the features that have it, in row order.

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
    batches_per_block: Mapping[int, int] | None = None,
) -> list[tuple[int, ...]]:
    """Move each feature with an affiliation of its own into another feature's.

    A feature whose affiliation no other feature shares would form a group of
    one, which is returned with its raw values. This step moves such a feature
    into the longest affiliation of another feature that is a proper subset of
    its own; the feature then loses the values in the dropped blocks but is
    corrected with others. As in HarmonizR's ``ur`` option, the target may
    itself be held by a single feature, which then keeps its affiliation and
    shares it.

    Features are taken longest affiliation first, so the order of the rows
    does not matter. Among equally long targets, the one more features hold
    wins, then the smallest tuple. An affiliation that spans a single batch is
    never a target: its group is left raw, so the move would only discard the
    feature's other values. A feature with no target keeps its affiliation and
    its values; HarmonizR empties it instead.

    Parameters
    ----------
    affiliation_list : list of tuple of int
        One affiliation per feature, as ``build_affiliation_list`` returns.
    batches_per_block : mapping of int to int or None
        The number of batches in each block, to tell which affiliations span a
        single batch. ``None`` counts each block as one batch, which is exact
        without blocking.

    Returns
    -------
    list of tuple of int
        A new list of the same length; the input is not modified.

    Examples
    --------
    The first two features each have an affiliation of their own. The first
    moves into the second's, which keeps it:

    >>> from harmonizepy.affiliation import remove_unique_combinations
    >>> remove_unique_combinations([(1, 2, 3), (1, 2), (4, 5), (4, 5)])
    [(1, 2), (1, 2), (4, 5), (4, 5)]
    """
    result = list(affiliation_list)
    counts = Counter(affil for affil in result if affil)

    def n_batches(pattern: tuple[int, ...]) -> int:
        if batches_per_block is None:
            return len(pattern)
        return sum(batches_per_block.get(block, 1) for block in pattern)

    sets = {pattern: frozenset(pattern) for pattern in counts}
    by_length: dict[int, list[tuple[int, ...]]] = {}
    for pattern in counts:
        if n_batches(pattern) >= 2:
            by_length.setdefault(len(pattern), []).append(pattern)
    lengths = sorted(by_length, reverse=True)

    unique = sorted(
        ((affil, i) for i, affil in enumerate(result) if affil and counts[affil] == 1),
        key=lambda item: (-len(item[0]), item[0]),
    )
    holder = dict(unique)
    kept: set[int] = set()
    rescued = 0
    for own, i in unique:
        if i in kept:
            continue
        own_set = sets[own]
        target = None
        for length in lengths:
            if length >= len(own):
                continue
            candidates = [p for p in by_length[length] if counts[p] > 0 and sets[p] < own_set]
            if candidates:
                target = max(candidates, key=lambda p: (counts[p], tuple(-b for b in p)))
                break
        if target is None:
            continue
        if counts[target] == 1:
            kept.add(holder[target])
        counts[own] -= 1
        counts[target] += 1
        result[i] = target
        rescued += 1

    if rescued:
        logger.debug("Unique removal moved %d feature(s)", rescued)
    return result
