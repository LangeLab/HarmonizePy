"""Blocks: groups of neighbouring batches used to form the sub-matrices.

A block label replaces the batch label only when affiliations and
sub-matrices are formed; the engines still correct each original batch.
Without blocking, the block list is the batch list. This follows R
``HarmonizR:::blocking``.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def build_block_list(
    batch_list: npt.NDArray[np.integer],
    block_size: int,
) -> npt.NDArray[np.integer]:
    """Return the block label of each sample, grouping neighbouring batches.

    Batches are taken in order of first appearance, and every *block_size*
    of them share a block. When the number of batches is not a multiple of
    *block_size*, the last block is smaller. Block labels start at 1, as in R.

    Parameters
    ----------
    batch_list : ndarray, shape (n_samples,)
        Batch label of each sample.
    block_size : int
        Batches per block: at least 2 and fewer than the number of batches,
        since one block holding every batch would leave nothing to correct.

    Returns
    -------
    ndarray, shape (n_samples,)
        Block label of each sample, with the dtype of *batch_list*.

    Raises
    ------
    ValueError
        If *block_size* is below 2 or not below the number of batches.

    Examples
    --------
    With four batches and blocks of three, the fourth batch forms a block alone:

    >>> import numpy as np
    >>> from harmonizepy.blocking import build_block_list
    >>> batch = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    >>> build_block_list(batch, block_size=2)
    array([1, 1, 1, 1, 2, 2, 2, 2])
    >>> build_block_list(batch, block_size=3)
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

    _, first_index, inverse = np.unique(batch_arr, return_index=True, return_inverse=True)
    rank_by_first_appearance = np.argsort(np.argsort(first_index, kind="stable"), kind="stable")
    block_of_label = rank_by_first_appearance // block_size + 1
    block_arr: npt.NDArray[np.integer] = block_of_label[inverse].astype(batch_arr.dtype)

    n_blocks = len(np.unique(block_arr))
    logger.debug(
        "Built %d blocks of size %d from %d batches",
        n_blocks,
        block_size,
        n_batches,
    )
    return block_arr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _unique_ordered(arr: npt.NDArray[np.generic]) -> list[object]:
    """Return the distinct values of *arr* in order of first appearance."""
    return list(dict.fromkeys(arr.tolist()))
