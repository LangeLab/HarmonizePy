"""Unit tests for harmonizepy.sorting.

Tests cover all three sort strategies (sparsity, jaccard, seriation),
the inverse-permutation property (restoring original column order after
sort + imaginary rebuild), no-mutation of the input data, and error
handling for unknown strategies.

Sections
--------
1. TestSparsitySort: ordering by completeness count
2. TestJaccardSort: greedy nearest-neighbour ordering
3. TestSeriationSort: PCA seriation ordering
4. TestInversePermutation: col_order restores original columns
5. TestNoDataMutation: input DataFrame unchanged
6. TestReturnTypes: shapes and dtypes of return values
7. TestEdgeCases: two batches, all-NaN features, intra-batch order
8. TestValidation: ValueError on unknown strategy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harmonizepy.blocking import _unique_ordered
from harmonizepy.sorting import (
    _build_presence_matrix,
    _column_order,
    _jaccard_order,
    _seriation_order,
    _sparsity_order,
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
        """Sparsity sort orders batches by descending feature count.

        Failure condition: the sparsest batch appears first or the
        most complete batch appears last.
        """
        df, batch = _make_skewed_dataset()
        _, sb, _ = sort_batches(df, batch, "sparsity", needed_values=2)
        assert sb[0] == 1  # batch 1 (20 features) first
        assert sb[3] == 2  # batch 2 (10 features) middle
        assert sb[6] == 3  # batch 3 (5 features) last

    def test_all_samples_preserved(self):
        """Sample count must be preserved after sorting.

        Failure condition: samples are dropped or duplicated.
        """
        df, batch = _make_skewed_dataset()
        sd, sb, co = sort_batches(df, batch, "sparsity", needed_values=2)
        assert sd.shape == df.shape
        assert len(sb) == len(batch)
        assert len(co) == len(batch)

    def test_column_values_unchanged(self):
        """Each column in sorted_data matches one original column.

        Failure condition: column values are transformed or corrupted.
        """
        df, batch = _make_skewed_dataset()
        sd, _, co = sort_batches(df, batch, "sparsity", needed_values=2)
        for new_pos, orig_pos in enumerate(co):
            orig_col = df.columns[orig_pos]
            pd.testing.assert_series_equal(sd.iloc[:, new_pos], df[orig_col], check_names=False)

    def test_uniform_dataset_stable(self):
        """Equal-completeness batches preserve original order.

        Failure condition: stable sort property is violated and
        batches are arbitrarily reordered.
        """
        df, batch = _make_dataset(n_features=10, n_batches=3, missing_frac=0.0)
        _, _, co = sort_batches(df, batch, "sparsity", needed_values=2)
        np.testing.assert_array_equal(co, np.arange(len(batch)))


# ---------------------------------------------------------------------------
# 2. TestJaccardSort
# ---------------------------------------------------------------------------


class TestJaccardSort:
    def test_returns_valid_permutation(self):
        """Jaccard sort preserves all unique batch IDs.

        Failure condition: a batch is dropped or duplicated in the sort.
        """
        df, batch = _make_dataset(n_batches=4, missing_frac=0.3, seed=1)
        _, sb, _ = sort_batches(df, batch, "jaccard", needed_values=2)
        unique_in_result = np.unique(sb)
        unique_in_input = np.unique(batch)
        np.testing.assert_array_equal(unique_in_result, unique_in_input)

    def test_all_samples_preserved(self):
        """Sample count is preserved after Jaccard sort.

        Failure condition: samples are dropped.
        """
        df, batch = _make_dataset(n_batches=4, missing_frac=0.3, seed=1)
        sd, _, _ = sort_batches(df, batch, "jaccard", needed_values=2)
        assert sd.shape == df.shape

    def test_column_values_unchanged(self):
        """Each sorted column matches one original column after Jaccard sort.

        Failure condition: values are corrupted by reordering.
        """
        df, batch = _make_dataset(n_batches=4, missing_frac=0.3, seed=1)
        sd, _, co = sort_batches(df, batch, "jaccard", needed_values=2)
        for new_pos, orig_pos in enumerate(co):
            orig_col = df.columns[orig_pos]
            pd.testing.assert_series_equal(sd.iloc[:, new_pos], df[orig_col], check_names=False)

    def test_similar_batches_are_adjacent(self):
        """The two most similar batches end up neighbours after Jaccard sort.

        Failure condition: the Jaccard similarity ordering does not
        place the most similar batches adjacent.
        """
        rng = np.random.default_rng(7)
        vals = rng.standard_normal((10, 9))
        vals[:, 6:] = np.nan
        df = pd.DataFrame(vals, columns=[f"s{i}" for i in range(9)])
        batch = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        _, sb, _ = sort_batches(df, batch, "jaccard", needed_values=2)
        positions = {b: i // 3 for i, b in enumerate(sb)}
        assert abs(positions[1] - positions[2]) == 1


# ---------------------------------------------------------------------------
# 3. TestSeriationSort
# ---------------------------------------------------------------------------


class TestSeriationSort:
    def test_returns_valid_permutation(self):
        """Seriation preserves all unique batch IDs.

        Failure condition: a batch is dropped or duplicated.
        """
        df, batch = _make_dataset(n_batches=4, missing_frac=0.3, seed=2)
        _, sb, _ = sort_batches(df, batch, "seriation", needed_values=2)
        np.testing.assert_array_equal(np.sort(sb), np.sort(batch))

    def test_all_samples_preserved(self):
        """Sample count is preserved after seriation sort.

        Failure condition: samples are dropped.
        """
        df, batch = _make_dataset(n_batches=4, missing_frac=0.3, seed=2)
        sd, _, _ = sort_batches(df, batch, "seriation", needed_values=2)
        assert sd.shape == df.shape

    def test_column_values_unchanged(self):
        """Each sorted column matches one original column after seriation.

        Failure condition: values are corrupted by reordering.
        """
        df, batch = _make_dataset(n_batches=4, missing_frac=0.3, seed=2)
        sd, _, co = sort_batches(df, batch, "seriation", needed_values=2)
        for new_pos, orig_pos in enumerate(co):
            orig_col = df.columns[orig_pos]
            pd.testing.assert_series_equal(sd.iloc[:, new_pos], df[orig_col], check_names=False)

    def test_five_batches(self):
        """Seriation works with more than 3 batches (exercises PCA path).

        Failure condition: the PCA-based ordering fails or crashes
        when the batch count exceeds the default fast path.
        """
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
        """Applying argsort(col_order) must restore the original column order.

        Failure condition: the inverse permutation does not reconstruct
        the original column arrangement.
        """
        df, batch = _make_dataset(n_batches=3, missing_frac=0.2, seed=5)
        sd, _, co = sort_batches(df, batch, strategy, needed_values=2)
        restored = sd.iloc[:, np.argsort(co)]
        assert list(restored.columns) == list(df.columns)
        pd.testing.assert_frame_equal(restored.reset_index(drop=True), df.reset_index(drop=True))

    @pytest.mark.parametrize("strategy", ["sparsity", "jaccard", "seriation"])
    def test_col_order_is_permutation(self, strategy):
        """col_order must be a complete permutation of column indices.

        Failure condition: indices are duplicated or out of range.
        """
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
        """sort_batches must not modify the input DataFrame or batch array.

        Failure condition: the function mutates the caller's data in place.
        """
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
        """Return types must be (DataFrame, ndarray, ndarray) with correct dtypes.

        Failure condition: a return value has the wrong container type
        or ``col_order`` is not ``np.intp``.
        """
        df, batch = _make_dataset(n_batches=3, seed=11)
        sd, sb, co = sort_batches(df, batch, strategy, needed_values=2)
        assert isinstance(sd, pd.DataFrame)
        assert isinstance(sb, np.ndarray)
        assert isinstance(co, np.ndarray)
        assert co.dtype == np.intp

    @pytest.mark.parametrize("strategy", ["sparsity", "jaccard", "seriation"])
    def test_batch_list_dtype_preserved(self, strategy):
        """Sorted batch list must preserve the input dtype.

        Failure condition: dtype is cast, e.g. int32 to int64.
        """
        df, batch = _make_dataset(n_batches=3, seed=12)
        _, sb, _ = sort_batches(df, batch, strategy, needed_values=2)
        assert sb.dtype == batch.dtype


# ---------------------------------------------------------------------------
# 7. TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.parametrize("strategy", ["sparsity", "jaccard", "seriation"])
    def test_two_batches(self, strategy):
        """Sorting works with only two batches.

        Failure condition: the function crashes or drops data when
        the minimum number of batches is provided.
        """
        df, batch = _make_dataset(n_batches=2, n_per_batch=5, seed=20)
        sd, sb, _ = sort_batches(df, batch, strategy, needed_values=2)
        assert sd.shape == df.shape
        assert set(sb.tolist()) == {1, 2}

    @pytest.mark.parametrize("strategy", ["sparsity", "jaccard", "seriation"])
    def test_all_nan_feature_present_in_presence_matrix(self, strategy):
        """All-NaN features are handled without crashing.

        Failure condition: a feature with no valid data causes a
        division-by-zero or index error in the presence matrix.
        """
        df, batch = _make_dataset(n_features=15, n_batches=3, seed=21)
        df.iloc[:3, :] = np.nan
        sd, _, _ = sort_batches(df, batch, strategy, needed_values=2)
        assert sd.shape == df.shape

    @pytest.mark.parametrize("strategy", ["sparsity", "jaccard", "seriation"])
    def test_samples_within_batch_keep_relative_order(self, strategy):
        """Samples within the same batch retain their original relative order.

        Failure condition: columns are reordered within a batch.
        """
        df, batch = _make_skewed_dataset()
        sd, _, co = sort_batches(df, batch, strategy, needed_values=2)
        for bid in np.unique(batch):
            orig_positions = np.where(batch == bid)[0]
            new_positions_in_co = [i for i, v in enumerate(co) if v in orig_positions]
            orig_cols = [df.columns[p] for p in orig_positions]
            new_cols = [sd.columns[i] for i in new_positions_in_co]
            assert orig_cols == new_cols


# ---------------------------------------------------------------------------
# 8. TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_strategy_raises(self):
        """Unknown strategy string raises ValueError.

        Failure condition: a nonsense strategy name is accepted.
        """
        df, batch = _make_dataset(n_batches=2)
        with pytest.raises(ValueError, match="sort strategy"):
            sort_batches(df, batch, "unknown", needed_values=2)

    def test_typo_raises(self):
        """Case-mismatched strategy name raises ValueError.

        Failure condition: ``"sparsity"`` vs ``"Sparsity"`` is not
        distinguished.
        """
        df, batch = _make_dataset(n_batches=2)
        with pytest.raises(ValueError):
            sort_batches(df, batch, "Sparsity", needed_values=2)


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Direct unit tests for internal helper functions."""

    def test_sparsity_order_descending(self):
        """_sparsity_order returns batches in descending completeness.

        Failure condition: the ordering is ascending instead of
        descending, or stable sort is not used for ties.
        """
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
        order = _sparsity_order(presence)
        np.testing.assert_array_equal(order, [1, 0, 2])

    def test_jaccard_order_length(self):
        """_jaccard_order returns a permutation of the correct length.

        Failure condition: the output length does not match n_batches.
        """
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
        """_jaccard_order handles n=1 without crashing.

        Failure condition: the single-batch fast path is missing or
        returns an incorrect result.
        """
        presence = np.array([[True], [False]], dtype=np.bool_)
        order = _jaccard_order(presence)
        np.testing.assert_array_equal(order, [0])

    def test_seriation_order_two_batches_returns_identity(self):
        """_seriation_order fast-path for n<=2 returns [0, 1].

        Failure condition: fewer than 3 batches triggers PCA and
        returns a non-identity order.
        """
        presence = np.array([[True, False], [False, True]], dtype=np.bool_)
        order = _seriation_order(presence)
        np.testing.assert_array_equal(order, [0, 1])

    def test_seriation_order_permutation(self):
        """_seriation_order returns a valid permutation for n>2.

        Failure condition: the PCA-based ordering produces duplicate
        or out-of-range indices.
        """
        df, batch = _make_dataset(n_batches=4, missing_frac=0.2, seed=99)
        unique = _unique_ordered(batch)
        presence = _build_presence_matrix(df, batch, unique, needed_values=2)
        order = _seriation_order(presence)
        assert len(order) == 4
        np.testing.assert_array_equal(np.sort(order), [0, 1, 2, 3])

    def test_column_order_groups_by_batch(self):
        """_column_order appends indices in the requested batch order.

        Failure condition: the column order does not reflect the
        requested batch sequence.
        """
        batch = np.array([1, 1, 2, 2, 3, 3])
        ordered = np.array([3, 1, 2])
        co = _column_order(batch, ordered)
        np.testing.assert_array_equal(co, [4, 5, 0, 1, 2, 3])

    def test_column_order_preserves_intra_batch_order(self):
        """Samples within each batch retain their original relative order.

        Failure condition: columns within a batch are reordered.
        """
        batch = np.array([2, 2, 1, 1, 1])
        ordered = np.array([1, 2])
        co = _column_order(batch, ordered)
        np.testing.assert_array_equal(co, [2, 3, 4, 0, 1])


class TestPresenceMatrix:
    def test_shape(self):
        """Presence matrix has shape (n_features, n_batches) with bool dtype.

        Failure condition: dimensions are swapped or dtype is not bool.
        """
        df, batch = _make_dataset(n_features=10, n_batches=3, n_per_batch=4)
        unique = _unique_ordered(batch)
        p = _build_presence_matrix(df, batch, unique, needed_values=2)
        assert p.shape == (10, 3)
        assert p.dtype == np.bool_

    def test_all_present_when_no_nan(self):
        """All features present when data has no NaN.

        Failure condition: missing values are falsely detected.
        """
        df, batch = _make_dataset(n_features=5, n_batches=2, n_per_batch=3, missing_frac=0.0)
        unique = _unique_ordered(batch)
        p = _build_presence_matrix(df, batch, unique, needed_values=2)
        assert p.all()

    def test_absent_when_all_nan(self):
        """All-NaN batch marks all features absent for that batch.

        Failure condition: an all-NaN batch is still counted as present.
        """
        df, batch = _make_dataset(n_features=5, n_batches=2, n_per_batch=3)
        df.iloc[:, 3:] = np.nan
        unique = _unique_ordered(batch)
        p = _build_presence_matrix(df, batch, unique, needed_values=2)
        assert p[:, 0].all()  # batch 1 present
        assert not p[:, 1].any()  # batch 2 absent

    def test_needed_values_one_threshold(self):
        """needed_values=1 includes borderline batches; needed_values=2 excludes them.

        Failure condition: the threshold is applied incorrectly,
        e.g. nv=2 includes a batch with only 1 valid observation.
        """
        df, batch = _make_dataset(n_features=4, n_batches=2, n_per_batch=3)
        df.iloc[0, 1:3] = np.nan
        df.iloc[0, 4:6] = np.nan
        unique = _unique_ordered(batch)
        p2 = _build_presence_matrix(df, batch, unique, needed_values=2)
        p1 = _build_presence_matrix(df, batch, unique, needed_values=1)
        assert not p2[0, 0] and not p2[0, 1]
        assert p1[0, 0] and p1[0, 1]


class TestUniqueOrderedBatches:
    def test_first_appearance_order(self):
        """Unique values returned in first-appearance order, not sorted order.

        Failure condition: values are sorted numerically instead of
        by appearance order.
        """
        batch = np.array([3, 3, 1, 1, 2, 2])
        result = _unique_ordered(batch)
        np.testing.assert_array_equal(result, [3, 1, 2])

    def test_already_sorted(self):
        """Already sorted input preserves its order.

        Failure condition: the function reorders an already-sorted input.
        """
        batch = np.array([1, 1, 2, 2, 3, 3])
        result = _unique_ordered(batch)
        np.testing.assert_array_equal(result, [1, 2, 3])


class TestJaccardDegenerateCases:
    def test_all_batches_identical_presence(self):
        """When all batches have identical presence, Jaccard sort must still
        produce a valid full permutation.

        Failure condition: the greedy traversal fails to visit all batches
        when all similarities are equal.
        """
        df, batch = _make_dataset(n_features=10, n_batches=4, missing_frac=0.0)
        _, sb, co = sort_batches(df, batch, "jaccard", needed_values=2)
        np.testing.assert_array_equal(np.sort(np.unique(sb)), [1, 2, 3, 4])
        np.testing.assert_array_equal(np.sort(co), np.arange(df.shape[1]))

    def test_all_batches_empty_presence(self):
        """When all features are NaN in every batch, Jaccard sort must
        produce a valid permutation.

        Failure condition: zero similarities cause a division-by-zero
        or an incomplete traversal.
        """
        df, batch = _make_dataset(n_features=10, n_batches=3, n_per_batch=4)
        df.iloc[:, :] = np.nan
        _, _, co = sort_batches(df, batch, "jaccard", needed_values=2)
        np.testing.assert_array_equal(np.sort(co), np.arange(df.shape[1]))

    def test_empty_batch_not_chosen_as_start_when_others_are_not_empty(self):
        """A batch with zero present features must not be the Jaccard start node.

        Failure condition: an empty batch has highest total similarity
        (incorrectly) and is chosen as the traversal start.
        """
        df, batch = _make_dataset(n_features=10, n_batches=3, n_per_batch=3)
        df.iloc[:, 3:6] = np.nan
        _, sb, _ = sort_batches(df, batch, "jaccard", needed_values=2)
        assert sb[0] != 2
