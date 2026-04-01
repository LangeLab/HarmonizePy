"""Unit tests for harmonizepy.blocking.

Sections
--------
1. TestBuildBlockList        — grouping, remainder, dtype, 1-indexing
2. TestBlockListValues       — exact output for hand-crafted inputs
3. TestEdgeCases             — minimum block_size=2, non-contiguous batch IDs
4. TestValidation            — ValueError on bad block_size
5. TestHelperFunctions       — _unique_ordered
"""

from __future__ import annotations

import numpy as np
import pytest

from harmonizepy.blocking import build_block_list, _unique_ordered


# ---------------------------------------------------------------------------
# 1. TestBuildBlockList
# ---------------------------------------------------------------------------


class TestBuildBlockList:
    def test_output_length_matches_input(self):
        batch = np.array([1, 1, 2, 2, 3, 3])
        result = build_block_list(batch, block_size=2)
        assert len(result) == len(batch)

    def test_dtype_preserved(self):
        batch = np.array([1, 1, 2, 2, 3, 3], dtype=np.int32)
        result = build_block_list(batch, block_size=2)
        assert result.dtype == batch.dtype

    def test_output_is_one_indexed(self):
        """Block IDs start at 1, not 0."""
        batch = np.array([1, 2, 3, 4])
        result = build_block_list(batch, block_size=2)
        assert result.min() == 1

    def test_number_of_unique_blocks_even_division(self):
        """6 batches / block_size=2 → 3 blocks."""
        batch = np.repeat(np.arange(1, 7), 3)
        result = build_block_list(batch, block_size=2)
        assert np.unique(result).tolist() == [1, 2, 3]

    def test_number_of_unique_blocks_with_remainder(self):
        """5 batches / block_size=2 → 3 blocks (remainder batch in block 3)."""
        batch = np.repeat(np.arange(1, 6), 2)
        result = build_block_list(batch, block_size=2)
        assert np.unique(result).tolist() == [1, 2, 3]

    def test_samples_in_same_batch_get_same_block(self):
        """All samples from the same batch must share one block ID."""
        batch = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        result = build_block_list(batch, block_size=2)
        for bid in np.unique(batch):
            block_ids_for_batch = result[batch == bid]
            assert len(np.unique(block_ids_for_batch)) == 1

    def test_no_mutation_of_input(self):
        batch = np.array([1, 1, 2, 2, 3, 3])
        original = batch.copy()
        build_block_list(batch, block_size=2)
        np.testing.assert_array_equal(batch, original)


# ---------------------------------------------------------------------------
# 2. TestBlockListValues
# ---------------------------------------------------------------------------


class TestBlockListValues:
    def test_block_size_2_four_batches(self):
        """4 batches, block_size=2 → batches {1,2}→block 1, {3,4}→block 2."""
        batch = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        result = build_block_list(batch, block_size=2)
        expected = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        np.testing.assert_array_equal(result, expected)

    def test_block_size_3_five_batches(self):
        """5 batches, block_size=3 → {1,2,3}→block 1, {4,5}→block 2."""
        batch = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        result = build_block_list(batch, block_size=3)
        expected = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        np.testing.assert_array_equal(result, expected)

    def test_block_size_2_five_batches_remainder(self):
        """5 batches, block_size=2 → {1,2}→1, {3,4}→2, {5}→3."""
        batch = np.array([1, 2, 3, 4, 5])
        result = build_block_list(batch, block_size=2)
        expected = np.array([1, 1, 2, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_block_list_equals_batch_list_when_block_size_equals_n_minus_1(self):
        """block_size = n_batches-1: first block has n-1 batches, last block is singleton."""
        # 4 batches: {1,2,3}→block 1, {4}→block 2
        batch = np.array([1, 2, 3, 4])
        result = build_block_list(batch, block_size=3)
        expected = np.array([1, 1, 1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_non_unit_sized_batches(self):
        """3 samples per batch, standard case."""
        batch = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        result = build_block_list(batch, block_size=2)
        expected = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# 3. TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_minimum_valid_block_size(self):
        """block_size=2 with 3 batches is the smallest valid call."""
        batch = np.array([1, 1, 2, 2, 3, 3])
        result = build_block_list(batch, block_size=2)
        assert result is not None
        assert len(result) == 6

    def test_non_contiguous_batch_ids(self):
        """Batch IDs need not be consecutive integers."""
        batch = np.array([10, 10, 20, 20, 30, 30])
        result = build_block_list(batch, block_size=2)
        # Batches 10+20 → block 1; batch 30 → block 2
        expected = np.array([1, 1, 1, 1, 2, 2])
        np.testing.assert_array_equal(result, expected)

    def test_batches_in_non_sorted_order(self):
        """Blocks are based on first-appearance order, not numeric batch ID order."""
        # batches appear as 3, 1, 2 in the data
        batch = np.array([3, 3, 1, 1, 2, 2])
        result = build_block_list(batch, block_size=2)
        # batch 3 (first seen) + batch 1 (second seen) → block 1
        # batch 2 (third seen) → block 2
        block_for_3 = result[0]
        block_for_1 = result[2]
        block_for_2 = result[4]
        assert block_for_3 == block_for_1  # same block
        assert block_for_2 != block_for_1  # different block

    def test_unequal_batch_sizes(self):
        """Batches with different numbers of samples still group correctly."""
        batch = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        result = build_block_list(batch, block_size=2)
        # batches {1,2}→block 1, {3,4}→block 2
        assert result[0] == result[1] == result[2] == 1    # batch 1 and 2
        assert result[3] == result[6] == 2                 # batch 3 and 4


# ---------------------------------------------------------------------------
# 4. TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_block_size_one_raises(self):
        batch = np.array([1, 1, 2, 2, 3, 3])
        with pytest.raises(ValueError, match="block_size must be >= 2"):
            build_block_list(batch, block_size=1)

    def test_block_size_zero_raises(self):
        batch = np.array([1, 1, 2, 2, 3, 3])
        with pytest.raises(ValueError, match="block_size must be >= 2"):
            build_block_list(batch, block_size=0)

    def test_block_size_equals_n_batches_raises(self):
        """block_size == n_batches would merge everything into one block."""
        batch = np.array([1, 1, 2, 2, 3, 3])
        with pytest.raises(ValueError, match="must be < number of unique batches"):
            build_block_list(batch, block_size=3)

    def test_block_size_exceeds_n_batches_raises(self):
        batch = np.array([1, 1, 2, 2, 3, 3])
        with pytest.raises(ValueError, match="must be < number of unique batches"):
            build_block_list(batch, block_size=10)


# ---------------------------------------------------------------------------
# 5. TestHelperFunctions
# ---------------------------------------------------------------------------


class TestUniqueOrdered:
    def test_first_appearance_order(self):
        arr = np.array([3, 3, 1, 1, 2, 2])
        assert _unique_ordered(arr) == [3, 1, 2]

    def test_already_sorted(self):
        arr = np.array([1, 2, 3])
        assert _unique_ordered(arr) == [1, 2, 3]

    def test_single_element(self):
        arr = np.array([5, 5, 5])
        assert _unique_ordered(arr) == [5]
