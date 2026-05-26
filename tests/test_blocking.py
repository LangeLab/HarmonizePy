"""Unit tests for harmonizepy.blocking.

Sections
--------
1. TestBuildBlockList: grouping, remainder, dtype, 1-indexing
2. TestBlockListValues: exact output for hand-crafted inputs
3. TestEdgeCases: minimum block_size=2, non-contiguous batch IDs
4. TestValidation: ValueError on bad block_size
5. TestHelperFunctions: _unique_ordered
"""

from __future__ import annotations

import numpy as np
import pytest

from harmonizepy.blocking import _unique_ordered, build_block_list

# ---------------------------------------------------------------------------
# 1. TestBuildBlockList
# ---------------------------------------------------------------------------


class TestBuildBlockList:
    def test_output_length_matches_input(self):
        """Output array length must match input batch length.

        Failure condition: samples are dropped or duplicated in the
        block-mapping step.
        """
        batch = np.array([1, 1, 2, 2, 3, 3])
        result = build_block_list(batch, block_size=2)
        assert len(result) == len(batch)

    def test_dtype_preserved(self):
        """Output dtype must match the input batch dtype.

        Failure condition: dtype is silently cast, e.g. int64 to int32.
        """
        batch = np.array([1, 1, 2, 2, 3, 3], dtype=np.int32)
        result = build_block_list(batch, block_size=2)
        assert result.dtype == batch.dtype

    def test_output_is_one_indexed(self):
        """Block IDs start at 1, not 0.

        Failure condition: the minimum block ID is 0 (0-indexed).
        """
        batch = np.array([1, 2, 3, 4])
        result = build_block_list(batch, block_size=2)
        assert result.min() == 1

    def test_number_of_unique_blocks_even_division(self):
        """6 batches with block_size=2 produce exactly 3 blocks.

        Failure condition: too many or too few blocks are created
        when the division is exact.
        """
        batch = np.repeat(np.arange(1, 7), 3)
        result = build_block_list(batch, block_size=2)
        assert np.unique(result).tolist() == [1, 2, 3]

    def test_number_of_unique_blocks_with_remainder(self):
        """5 batches with block_size=2 produce 3 blocks (remainder in last).

        Failure condition: the trailing remainder batch is dropped or
        incorrectly grouped.
        """
        batch = np.repeat(np.arange(1, 6), 2)
        result = build_block_list(batch, block_size=2)
        assert np.unique(result).tolist() == [1, 2, 3]

    def test_samples_in_same_batch_get_same_block(self):
        """All samples from the same original batch share one block ID.

        Failure condition: a batch is split across multiple blocks.
        """
        batch = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        result = build_block_list(batch, block_size=2)
        for bid in np.unique(batch):
            block_ids_for_batch = result[batch == bid]
            assert len(np.unique(block_ids_for_batch)) == 1

    def test_no_mutation_of_input(self):
        """Input batch array must not be modified in place.

        Failure condition: the function alters the caller's array.
        """
        batch = np.array([1, 1, 2, 2, 3, 3])
        original = batch.copy()
        build_block_list(batch, block_size=2)
        np.testing.assert_array_equal(batch, original)


# ---------------------------------------------------------------------------
# 2. TestBlockListValues
# ---------------------------------------------------------------------------


class TestBlockListValues:
    def test_block_size_2_four_batches(self):
        """4 batches, block_size=2 group into 2 blocks of 2 batches each.

        Failure condition: batches are not grouped as {1,2} and {3,4}.
        """
        batch = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        result = build_block_list(batch, block_size=2)
        expected = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        np.testing.assert_array_equal(result, expected)

    def test_block_size_3_five_batches(self):
        """5 batches, block_size=3 group into {1,2,3} and {4,5}.

        Failure condition: the first block has fewer than 3 batches
        or the remainder is not a separate block.
        """
        batch = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        result = build_block_list(batch, block_size=3)
        expected = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        np.testing.assert_array_equal(result, expected)

    def test_block_size_2_five_batches_remainder(self):
        """5 unit-size batches with block_size=2 produce blocks {1,2}, {3,4}, {5}.

        Failure condition: the trailing singleton batch is merged into
        the previous block or dropped.
        """
        batch = np.array([1, 2, 3, 4, 5])
        result = build_block_list(batch, block_size=2)
        expected = np.array([1, 1, 2, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_block_list_equals_batch_list_when_block_size_equals_n_minus_1(self):
        """block_size = n_batches-1 puts the last batch alone in the final block.

        Failure condition: the final singleton batch is not assigned
        its own block.
        """
        batch = np.array([1, 2, 3, 4])
        result = build_block_list(batch, block_size=3)
        expected = np.array([1, 1, 1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_non_unit_sized_batches(self):
        """Batches with multiple samples per batch group correctly.

        Failure condition: samples from the same batch are assigned
        different block IDs.
        """
        batch = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        result = build_block_list(batch, block_size=2)
        expected = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# 3. TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_minimum_valid_block_size(self):
        """block_size=2 with 3 batches is the smallest valid call.

        Failure condition: the minimum configuration crashes or returns
        wrong-length output.
        """
        batch = np.array([1, 1, 2, 2, 3, 3])
        result = build_block_list(batch, block_size=2)
        assert result is not None
        assert len(result) == 6

    def test_non_contiguous_batch_ids(self):
        """Non-consecutive batch IDs like 10, 20, 30 group correctly.

        Failure condition: blocking is based on numeric value rather
        than first-appearance order.
        """
        batch = np.array([10, 10, 20, 20, 30, 30])
        result = build_block_list(batch, block_size=2)
        expected = np.array([1, 1, 1, 1, 2, 2])
        np.testing.assert_array_equal(result, expected)

    def test_batches_in_non_sorted_order(self):
        """First-appearance order determines blocks, not numeric order.

        Failure condition: batches arriving as 3, 1, 2 are grouped
        by sorted value instead of arrival order.
        """
        batch = np.array([3, 3, 1, 1, 2, 2])
        result = build_block_list(batch, block_size=2)
        block_for_3 = result[0]
        block_for_1 = result[2]
        block_for_2 = result[4]
        assert block_for_3 == block_for_1  # same block
        assert block_for_2 != block_for_1  # different block

    def test_unequal_batch_sizes(self):
        """Batches with different sample counts group correctly.

        Failure condition: variable-size batches cause misalignment
        in block assignment.
        """
        batch = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        result = build_block_list(batch, block_size=2)
        assert result[0] == result[1] == result[2] == 1  # batch 1 and 2
        assert result[3] == result[6] == 2  # batch 3 and 4


# ---------------------------------------------------------------------------
# 4. TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_block_size_one_raises(self):
        """block_size=1 raises ValueError.

        Failure condition: a block_size below the minimum of 2 is
        accepted instead of rejected.
        """
        batch = np.array([1, 1, 2, 2, 3, 3])
        with pytest.raises(ValueError, match="block_size must be >= 2"):
            build_block_list(batch, block_size=1)

    def test_block_size_zero_raises(self):
        """block_size=0 raises ValueError.

        Failure condition: zero is accepted as a valid block size.
        """
        batch = np.array([1, 1, 2, 2, 3, 3])
        with pytest.raises(ValueError, match="block_size must be >= 2"):
            build_block_list(batch, block_size=0)

    def test_block_size_equals_n_batches_raises(self):
        """block_size equal to n_batches raises ValueError.

        Failure condition: merging all batches into one block is accepted.
        """
        batch = np.array([1, 1, 2, 2, 3, 3])
        with pytest.raises(ValueError, match="must be < number of unique batches"):
            build_block_list(batch, block_size=3)

    def test_block_size_exceeds_n_batches_raises(self):
        """block_size larger than n_batches raises ValueError.

        Failure condition: too-large block size is accepted.
        """
        batch = np.array([1, 1, 2, 2, 3, 3])
        with pytest.raises(ValueError, match="must be < number of unique batches"):
            build_block_list(batch, block_size=10)


# ---------------------------------------------------------------------------
# 5. TestHelperFunctions
# ---------------------------------------------------------------------------


class TestUniqueOrdered:
    def test_first_appearance_order(self):
        """Unique values are returned in first-appearance order.

        Failure condition: values are sorted or reordered.
        """
        arr = np.array([3, 3, 1, 1, 2, 2])
        assert _unique_ordered(arr) == [3, 1, 2]

    def test_already_sorted(self):
        """Already-sorted input preserves order.

        Failure condition: the function reorders an already-sorted input.
        """
        arr = np.array([1, 2, 3])
        assert _unique_ordered(arr) == [1, 2, 3]

    def test_single_element(self):
        """Repeated single value returns just that value once.

        Failure condition: duplicates are not deduplicated.
        """
        arr = np.array([5, 5, 5])
        assert _unique_ordered(arr) == [5]
