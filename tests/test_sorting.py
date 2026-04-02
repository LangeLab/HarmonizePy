"""Unit tests for harmonizepy.sorting.

Tests cover all three sort strategies (sparsity, jaccard, seriation),
the inverse-permutation property (restoring original column order after
sort + imaginary rebuild), no-mutation of the input data, and error
handling for unknown strategies.

Sections
--------
1. TestSparsitySort    — ordering by completeness count
2. TestJaccardSort     — greedy nearest-neighbour ordering
3. TestSeriationSort   — PCA seriation ordering
4. TestInversePermutation  — col_order restores original columns
5. TestNoDataMutation  — input DataFrame unchanged
6. TestReturnTypes     — shapes and dtypes of return values
7. TestEdgeCases       — two batches, all-NaN features, intra-batch order
8. TestValidation      — ValueError on unknown strategy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harmonizepy.sorting import (
    _build_presence_matrix,
    _column_order,
    _jaccard_order,
    _seriation_order,
    _sparsity_order,
    _unique_batches_ordered,
    sort_batches,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_dataset(
    n_features: int = 20,
    n_batches: int = 3,
    n_per_batch: int = 4,
    seed: int = 0,
    missing_frac: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (data, batch_list) with optional structural missingness."""
    rng = np.random.default_rng(seed)
    n_samples = n_batches * n_per_batch
    values = rng.standard_normal((n_features, n_samples))
    if missing_frac > 0:
        mask = rng.random((n_features, n_samples)) < missing_frac
        values[mask] = np.nan
    cols = [f"s{i}" for i in range(n_samples)]
    df = pd.DataFrame(values, index=[f"f{i}" for i in range(n_features)], columns=cols)
    batch = np.repeat(np.arange(1, n_batches + 1), n_per_batch)
    return df, batch


def _make_skewed_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    """Three batches with very different completeness profiles.

    Batch 1: all 20 features present (0 NaN).
    Batch 2: 10 features present (all NaN in the other 10).
    Batch 3: 5 features present (all NaN in the other 15).

    With needed_values=2 and n_per_batch=3, presence counts are 20, 10, 5.
    Sparsity sort descending → [batch1, batch2, batch3] (most complete first).
    """
    rng = np.random.default_rng(42)
    n_features = 20
    _n_per_batch = 3
    n_samples = 9
    values = rng.standard_normal((n_features, n_samples))
    # Batch2: features 10-19 all NaN
    values[10:, 3:6] = np.nan
    # Batch3: features 5-19 all NaN
    values[5:, 6:] = np.nan
    cols = [f"s{i}" for i in range(n_samples)]
    df = pd.DataFrame(values, index=[f"f{i}" for i in range(n_features)], columns=cols)
    batch = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    return df, batch


# ---------------------------------------------------------------------------
# 1. TestSparsitySort
# ---------------------------------------------------------------------------


class TestSparsitySort:
    def test_ordering_is_descending_completeness(self):
        """Batches ordered by descending feature count (most complete first)."""
        df, batch = _make_skewed_dataset()
        _, sb, _ = sort_batches(df, batch, "sparsity", needed_values=2)
        # _make_skewed_dataset: batch 3 → 5 features, batch 2 → 10, batch 1 → 20
        # Sparsity descending → [1, 2, 3] (most complete first, matching R)
        assert sb[0] == 1  # most complete first
        assert sb[3] == 2  # mid completeness in the middle
        assert sb[6] == 3  # sparsest last

    def test_all_samples_preserved(self):
        df, batch = _make_skewed_dataset()
        sd, sb, co = sort_batches(df, batch, "sparsity", needed_values=2)
        assert sd.shape == df.shape
        assert len(sb) == len(batch)
        assert len(co) == len(batch)

    def test_column_values_unchanged(self):
        """Each column in sorted_data must equal the same column in original."""
        df, batch = _make_skewed_dataset()
        sd, _, co = sort_batches(df, batch, "sparsity", needed_values=2)
        for new_pos, orig_pos in enumerate(co):
            orig_col = df.columns[orig_pos]
            pd.testing.assert_series_equal(sd.iloc[:, new_pos], df[orig_col], check_names=False)

    def test_uniform_dataset_stable(self):
        """When all batches have equal completeness, order is stable (unchanged)."""
        df, batch = _make_dataset(n_features=10, n_batches=3, missing_frac=0.0)
        _, _, co = sort_batches(df, batch, "sparsity", needed_values=2)
        # All completeness equal → argsort stable → original order preserved
        np.testing.assert_array_equal(co, np.arange(len(batch)))


# ---------------------------------------------------------------------------
# 2. TestJaccardSort
# ---------------------------------------------------------------------------


class TestJaccardSort:
    def test_returns_valid_permutation(self):
        """All batch IDs appear exactly once."""
        df, batch = _make_dataset(n_batches=4, missing_frac=0.3, seed=1)
        _, sb, _ = sort_batches(df, batch, "jaccard", needed_values=2)
        unique_in_result = np.unique(sb)
        unique_in_input = np.unique(batch)
        np.testing.assert_array_equal(unique_in_result, unique_in_input)

    def test_all_samples_preserved(self):
        df, batch = _make_dataset(n_batches=4, missing_frac=0.3, seed=1)
        sd, _, _ = sort_batches(df, batch, "jaccard", needed_values=2)
        assert sd.shape == df.shape

    def test_column_values_unchanged(self):
        df, batch = _make_dataset(n_batches=4, missing_frac=0.3, seed=1)
        sd, _, co = sort_batches(df, batch, "jaccard", needed_values=2)
        for new_pos, orig_pos in enumerate(co):
            orig_col = df.columns[orig_pos]
            pd.testing.assert_series_equal(sd.iloc[:, new_pos], df[orig_col], check_names=False)

    def test_similar_batches_are_adjacent(self):
        """The two most similar batches (by Jaccard) end up neighbors."""
        # Batch 1 and 2 share all features; batch 3 shares nothing with them.
        rng = np.random.default_rng(7)
        vals = rng.standard_normal((10, 9))
        # Batch 3: all NaN
        vals[:, 6:] = np.nan
        df = pd.DataFrame(vals, columns=[f"s{i}" for i in range(9)])
        # Batches 1 and 2 ↔ cols 0-2 and 3-5; batch 3 ↔ cols 6-8
        batch = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        _, sb, _ = sort_batches(df, batch, "jaccard", needed_values=2)
        # Batch 3 (all NaN → zero features present) is dissimilar from 1 and 2.
        # Batches 1 and 2 should be adjacent somewhere in the result.
        positions = {b: i // 3 for i, b in enumerate(sb)}  # batch → position index
        assert abs(positions[1] - positions[2]) == 1


# ---------------------------------------------------------------------------
# 3. TestSeriationSort
# ---------------------------------------------------------------------------


class TestSeriationSort:
    def test_returns_valid_permutation(self):
        df, batch = _make_dataset(n_batches=4, missing_frac=0.3, seed=2)
        _, sb, _ = sort_batches(df, batch, "seriation", needed_values=2)
        np.testing.assert_array_equal(np.sort(sb), np.sort(batch))

    def test_all_samples_preserved(self):
        df, batch = _make_dataset(n_batches=4, missing_frac=0.3, seed=2)
        sd, _, _ = sort_batches(df, batch, "seriation", needed_values=2)
        assert sd.shape == df.shape

    def test_column_values_unchanged(self):
        df, batch = _make_dataset(n_batches=4, missing_frac=0.3, seed=2)
        sd, _, co = sort_batches(df, batch, "seriation", needed_values=2)
        for new_pos, orig_pos in enumerate(co):
            orig_col = df.columns[orig_pos]
            pd.testing.assert_series_equal(sd.iloc[:, new_pos], df[orig_col], check_names=False)

    def test_five_batches(self):
        """Seriation with >3 batches exercises PCA on the presence matrix."""
        df, batch = _make_dataset(n_batches=5, missing_frac=0.2, seed=3)
        sd, sb, _ = sort_batches(df, batch, "seriation", needed_values=2)
        assert sd.shape == df.shape
        assert set(sb.tolist()) == set(range(1, 6))


# ---------------------------------------------------------------------------
# 4. TestInversePermutation
# ---------------------------------------------------------------------------


class TestInversePermutation:
    """Verify that np.argsort(col_order) restores the original column order."""

    @pytest.mark.parametrize("strategy", ["sparsity", "jaccard", "seriation"])
    def test_restore_original_order(self, strategy):
        df, batch = _make_dataset(n_batches=3, missing_frac=0.2, seed=5)
        sd, _, co = sort_batches(df, batch, strategy, needed_values=2)
        # Simulate a rebuild that preserves sorted column order (values unchanged)
        restored = sd.iloc[:, np.argsort(co)]
        # Column names should match original
        assert list(restored.columns) == list(df.columns)
        # Values should be identical
        pd.testing.assert_frame_equal(restored.reset_index(drop=True), df.reset_index(drop=True))

    @pytest.mark.parametrize("strategy", ["sparsity", "jaccard", "seriation"])
    def test_col_order_is_permutation(self, strategy):
        df, batch = _make_dataset(n_batches=4, missing_frac=0.15, seed=6)
        _, _, co = sort_batches(df, batch, strategy, needed_values=2)
        assert len(co) == df.shape[1]
        np.testing.assert_array_equal(np.sort(co), np.arange(df.shape[1]))


# ---------------------------------------------------------------------------
# 5. TestNoDataMutation
# ---------------------------------------------------------------------------


class TestNoDataMutation:
    @pytest.mark.parametrize("strategy", ["sparsity", "jaccard", "seriation"])
    def test_input_data_unchanged(self, strategy):
        df, batch = _make_dataset(n_batches=3, missing_frac=0.1, seed=10)
        original_values = df.values.copy()
        original_batch = batch.copy()
        sort_batches(df, batch, strategy, needed_values=2)
        np.testing.assert_array_equal(df.values, original_values)
        np.testing.assert_array_equal(batch, original_batch)


# ---------------------------------------------------------------------------
# 6. TestReturnTypes
# ---------------------------------------------------------------------------


class TestReturnTypes:
    @pytest.mark.parametrize("strategy", ["sparsity", "jaccard", "seriation"])
    def test_return_types(self, strategy):
        df, batch = _make_dataset(n_batches=3, seed=11)
        sd, sb, co = sort_batches(df, batch, strategy, needed_values=2)
        assert isinstance(sd, pd.DataFrame)
        assert isinstance(sb, np.ndarray)
        assert isinstance(co, np.ndarray)
        assert co.dtype == np.intp

    @pytest.mark.parametrize("strategy", ["sparsity", "jaccard", "seriation"])
    def test_batch_list_dtype_preserved(self, strategy):
        df, batch = _make_dataset(n_batches=3, seed=12)
        _, sb, _ = sort_batches(df, batch, strategy, needed_values=2)
        assert sb.dtype == batch.dtype


# ---------------------------------------------------------------------------
# 7. TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.parametrize("strategy", ["sparsity", "jaccard", "seriation"])
    def test_two_batches(self, strategy):
        """Works correctly with only two batches."""
        df, batch = _make_dataset(n_batches=2, n_per_batch=5, seed=20)
        sd, sb, _ = sort_batches(df, batch, strategy, needed_values=2)
        assert sd.shape == df.shape
        assert set(sb.tolist()) == {1, 2}

    @pytest.mark.parametrize("strategy", ["sparsity", "jaccard", "seriation"])
    def test_all_nan_feature_present_in_presence_matrix(self, strategy):
        """All-NaN features yield False in presence matrix; sort still runs."""
        df, batch = _make_dataset(n_features=15, n_batches=3, seed=21)
        # Make the first 3 features all-NaN
        df.iloc[:3, :] = np.nan
        sd, _, _ = sort_batches(df, batch, strategy, needed_values=2)
        assert sd.shape == df.shape

    @pytest.mark.parametrize("strategy", ["sparsity", "jaccard", "seriation"])
    def test_samples_within_batch_keep_relative_order(self, strategy):
        """Samples within the same batch retain their original relative order."""
        df, batch = _make_skewed_dataset()
        sd, _, co = sort_batches(df, batch, strategy, needed_values=2)
        for bid in np.unique(batch):
            orig_positions = np.where(batch == bid)[0]
            new_positions_in_co = [i for i, v in enumerate(co) if v in orig_positions]
            # Values at new positions should match original order
            orig_cols = [df.columns[p] for p in orig_positions]
            new_cols = [sd.columns[i] for i in new_positions_in_co]
            assert orig_cols == new_cols


# ---------------------------------------------------------------------------
# 8. TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_strategy_raises(self):
        df, batch = _make_dataset(n_batches=2)
        with pytest.raises(ValueError, match="sort strategy"):
            sort_batches(df, batch, "unknown", needed_values=2)

    def test_typo_raises(self):
        df, batch = _make_dataset(n_batches=2)
        with pytest.raises(ValueError):
            sort_batches(df, batch, "Sparsity", needed_values=2)


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Direct unit tests for internal helper functions."""

    def test_sparsity_order_descending(self):
        """_sparsity_order returns batches in descending completeness order (matches R)."""
        # 3 batches: completeness 2, 5, 2 → expected descending order: [1, 0, 2]
        # (stable: index 1 has 5, then 0 before 2 because they tie at 2)
        presence = np.array(
            [
                [True, True, True],
                [True, True, False],
                [False, True, True],
                [False, True, False],
                [False, True, False],
            ],
            dtype=np.bool_,
        )
        # Completeness: col0=2, col1=5, col2=2 → descending order: [1, 0, 2]
        order = _sparsity_order(presence)
        np.testing.assert_array_equal(order, [1, 0, 2])

    def test_jaccard_order_length(self):
        """_jaccard_order returns a permutation of length n_batches."""
        presence = np.array(
            [
                [True, True, False],
                [True, False, True],
                [True, True, True],
            ],
            dtype=np.bool_,
        )
        order = _jaccard_order(presence)
        assert len(order) == 3
        np.testing.assert_array_equal(np.sort(order), [0, 1, 2])

    def test_jaccard_order_single_batch(self):
        """_jaccard_order handles the n==1 fast path."""
        presence = np.array([[True], [False]], dtype=np.bool_)
        order = _jaccard_order(presence)
        np.testing.assert_array_equal(order, [0])

    def test_seriation_order_two_batches_returns_identity(self):
        """_seriation_order fast-path for n<=2 returns [0, 1]."""
        presence = np.array([[True, False], [False, True]], dtype=np.bool_)
        order = _seriation_order(presence)
        np.testing.assert_array_equal(order, [0, 1])

    def test_seriation_order_permutation(self):
        """_seriation_order returns a valid permutation for n>2."""
        df, batch = _make_dataset(n_batches=4, missing_frac=0.2, seed=99)
        unique = _unique_batches_ordered(batch)
        presence = _build_presence_matrix(df, batch, unique, needed_values=2)
        order = _seriation_order(presence)
        assert len(order) == 4
        np.testing.assert_array_equal(np.sort(order), [0, 1, 2, 3])

    def test_column_order_groups_by_batch(self):
        """_column_order appends all indices for each batch in requested order."""
        batch = np.array([1, 1, 2, 2, 3, 3])
        # Request order: batch 3, batch 1, batch 2
        ordered = np.array([3, 1, 2])
        co = _column_order(batch, ordered)
        np.testing.assert_array_equal(co, [4, 5, 0, 1, 2, 3])

    def test_column_order_preserves_intra_batch_order(self):
        """Samples within each batch keep their original relative order."""
        batch = np.array([2, 2, 1, 1, 1])
        ordered = np.array([1, 2])
        co = _column_order(batch, ordered)
        # batch 1 → original indices 2,3,4; batch 2 → 0,1
        np.testing.assert_array_equal(co, [2, 3, 4, 0, 1])


class TestPresenceMatrix:
    def test_shape(self):
        df, batch = _make_dataset(n_features=10, n_batches=3, n_per_batch=4)
        unique = _unique_batches_ordered(batch)
        p = _build_presence_matrix(df, batch, unique, needed_values=2)
        assert p.shape == (10, 3)
        assert p.dtype == np.bool_

    def test_all_present_when_no_nan(self):
        df, batch = _make_dataset(n_features=5, n_batches=2, n_per_batch=3, missing_frac=0.0)
        unique = _unique_batches_ordered(batch)
        p = _build_presence_matrix(df, batch, unique, needed_values=2)
        assert p.all()

    def test_absent_when_all_nan(self):
        df, batch = _make_dataset(n_features=5, n_batches=2, n_per_batch=3)
        # Blank out all of batch 2
        df.iloc[:, 3:] = np.nan
        unique = _unique_batches_ordered(batch)
        p = _build_presence_matrix(df, batch, unique, needed_values=2)
        assert p[:, 0].all()  # batch 1 present
        assert not p[:, 1].any()  # batch 2 absent

    def test_needed_values_one_threshold(self):
        """needed_values=1: a feature with exactly one non-NaN per batch is present."""
        df, batch = _make_dataset(n_features=4, n_batches=2, n_per_batch=3)
        # Leave only one non-NaN per batch for features 0 and 1
        df.iloc[0, 1:3] = np.nan  # batch 1: 1 valid, 2 NaN
        df.iloc[0, 4:6] = np.nan  # batch 2: 1 valid, 2 NaN
        unique = _unique_batches_ordered(batch)
        p2 = _build_presence_matrix(df, batch, unique, needed_values=2)
        p1 = _build_presence_matrix(df, batch, unique, needed_values=1)
        # With nv=2, feature 0 is absent from both batches
        assert not p2[0, 0] and not p2[0, 1]
        # With nv=1, feature 0 is present in both batches
        assert p1[0, 0] and p1[0, 1]


class TestUniqueOrderedBatches:
    def test_first_appearance_order(self):
        batch = np.array([3, 3, 1, 1, 2, 2])
        result = _unique_batches_ordered(batch)
        np.testing.assert_array_equal(result, [3, 1, 2])

    def test_already_sorted(self):
        batch = np.array([1, 1, 2, 2, 3, 3])
        result = _unique_batches_ordered(batch)
        np.testing.assert_array_equal(result, [1, 2, 3])


class TestJaccardDegenerateCases:
    def test_all_batches_identical_presence(self):
        """All batches have the same features present — Jaccard sim = 1 everywhere.

        Greedy traversal must still produce a valid full permutation.
        """
        df, batch = _make_dataset(n_features=10, n_batches=4, missing_frac=0.0)
        _, sb, co = sort_batches(df, batch, "jaccard", needed_values=2)
        # All unique batch IDs appear exactly once
        np.testing.assert_array_equal(np.sort(np.unique(sb)), [1, 2, 3, 4])
        # col_order is a complete permutation
        np.testing.assert_array_equal(np.sort(co), np.arange(df.shape[1]))

    def test_all_batches_empty_presence(self):
        """All features NaN in every batch — all Jaccard similarities are 0.

        Greedy traversal must still produce a valid full permutation.
        """
        df, batch = _make_dataset(n_features=10, n_batches=3, n_per_batch=4)
        df.iloc[:, :] = np.nan
        _, _, co = sort_batches(df, batch, "jaccard", needed_values=2)
        np.testing.assert_array_equal(np.sort(co), np.arange(df.shape[1]))

    def test_empty_batch_not_chosen_as_start_when_others_are_not_empty(self):
        """A batch with zero present features should not be the Jaccard start node.

        The start node is the one with highest total similarity to all others.
        An empty batch has similarity 0 to everything, so it must not start.
        """
        df, batch = _make_dataset(n_features=10, n_batches=3, n_per_batch=3)
        # Make batch 2 (indices 3-5) all NaN
        df.iloc[:, 3:6] = np.nan
        _, sb, _ = sort_batches(df, batch, "jaccard", needed_values=2)
        # Batch 2 (the empty one) must not be in first position
        assert sb[0] != 2
