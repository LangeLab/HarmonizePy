"""Batch blocking: group neighboring batches into superblocks.

Blocking collapses *N* consecutive batches into a single block ID for
the purpose of dissection (deciding which samples enter which sub-matrix
during splitting). It is purely a indexing operation. The original
``batch_list`` is kept unchanged and still drives the ComBat/limma
adjustment step.

When no blocking is requested, ``block_list == batch_list`` and no code
in this module is needed.

This mirrors R ``HarmonizR:::blocking``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def build_block_list(
    batch_list: npt.NDArray[np.integer],
    block_size: int,
) -> npt.NDArray[np.integer]:
    """Map each sample to a block ID by grouping neighbouring batches.

    Batches appear in the order they are first encountered in
    *batch_list*.  Every *block_size* consecutive unique batches are
    assigned the same block ID.  If the total number of unique batches is
    not evenly divisible by *block_size*, the trailing remainder batches
    form a smaller final block.

    The returned array has the same length and dtype as *batch_list* and
    is 1-indexed (block IDs start at 1) to match R's convention.

    Parameters
    ----------
    batch_list : ndarray, shape (n_samples,)
        Integer batch label per sample.  The unique values, taken in
        first-appearance order, define the batch sequence that is
        partitioned into blocks.
    block_size : int
        Number of consecutive unique batches per block.  Must be >= 2
        and < the number of unique batches (a block size >= n_batches
        would merge all batches into one block, making adjustments
        impossible).

    Returns
    -------
    ndarray, shape (n_samples,), dtype same as *batch_list*
        Block label per sample.

    Raises
    ------
    ValueError
        If *block_size* < 2 or *block_size* >= number of unique batches.

    Examples
    --------
    >>> import numpy as np
    >>> from harmonizepy.blocking import build_block_list
    >>> batch = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    >>> build_block_list(batch, block_size=2)
    array([1, 1, 1, 1, 2, 2, 2, 2])
    >>> build_block_list(batch, block_size=3)  # remainder: batch 4 alone
    array([1, 1, 1, 1, 1, 1, 2, 2])
    """
    batch_arr = np.asarray(batch_list)
    unique_batches = _unique_ordered(batch_arr)
    n_batches = len(unique_batches)

    if block_size < 2:
        raise ValueError(
            f"block_size must be >= 2, got {block_size}. Use block=None to disable blocking."
        )
    if block_size >= n_batches:
        raise ValueError(
            f"block_size ({block_size}) must be < number of unique batches "
            f"({n_batches}). A block spanning all batches makes adjustment impossible "
            f"because every feature would be in a single group."
        )

    # Assign each unique batch to a block index (0-based), then +1 for 1-indexing
    batch_to_block = {}
    for i, bid in enumerate(unique_batches):
        batch_to_block[int(bid)] = i // block_size + 1

    block_arr = np.empty_like(batch_arr)
    for j, bid in enumerate(batch_arr):
        block_arr[j] = batch_to_block[int(bid)]

    return block_arr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _unique_ordered(arr: npt.NDArray[np.integer]) -> list[int]:
    """Return unique values from *arr* in first-appearance order."""
    seen: set[int] = set()
    result: list[int] = []
    for v in arr:
        iv = int(v)
        if iv not in seen:
            seen.add(iv)
            result.append(iv)
    return result
