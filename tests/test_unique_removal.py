"""Unit tests for harmonizepy.affiliation.remove_unique_combinations.

Sections
--------
1. TestSingletonRescue        — singletons are cropped to nearest shared pattern
2. TestNonSingletonUnchanged  — non-singleton and empty tuples are not modified
3. TestMinimalCropping        — fewest batches removed (greedy largest-first)
4. TestEdgeCases              — all unique, single feature, already shared
5. TestHelperFunctions        — _find_best_crop
"""

from __future__ import annotations

from harmonizepy.affiliation import _find_best_crop, remove_unique_combinations

# ---------------------------------------------------------------------------
# 1. TestSingletonRescue
# ---------------------------------------------------------------------------


class TestSingletonRescue:
    def test_basic_singleton_cropped_to_shared_pattern(self):
        """A singleton (1,2,3) with (1,2) shared by 2 others → cropped to (1,2)."""
        affil = [(1, 2, 3), (1, 2), (1, 2)]
        result = remove_unique_combinations(affil)
        assert result[0] == (1, 2)

    def test_non_singletons_unchanged(self):
        affil = [(1, 2, 3), (1, 2), (1, 2)]
        result = remove_unique_combinations(affil)
        assert result[1] == (1, 2)
        assert result[2] == (1, 2)

    def test_multiple_singletons_all_rescued(self):
        """Two singletons, both rescuable to the same shared pattern."""
        affil = [(1, 2, 3), (1, 2, 4), (1, 2), (1, 2)]
        result = remove_unique_combinations(affil)
        assert result[0] == (1, 2)
        assert result[1] == (1, 2)

    def test_singleton_with_one_batch_removed(self):
        """(1, 2, 3) → (1, 3) when (1, 3) is the only shared pattern."""
        affil = [(1, 2, 3), (1, 3), (1, 3)]
        result = remove_unique_combinations(affil)
        assert result[0] == (1, 3)

    def test_singleton_chooses_largest_shared_match(self):
        """Two possible crops: (1,2,3) and (1,2). Should choose (1,2,3) (larger)."""
        affil = [(1, 2, 3, 4), (1, 2, 3), (1, 2, 3), (1, 2), (1, 2)]
        result = remove_unique_combinations(affil)
        # Greedy: size 3 is tried before size 2 → (1,2,3)
        assert result[0] == (1, 2, 3)

    def test_result_length_unchanged(self):
        affil = [(1, 2, 3), (1, 2), (1, 2), (3,), (3,)]
        result = remove_unique_combinations(affil)
        assert len(result) == len(affil)

    def test_list_is_independent_copy(self):
        """The returned list must not be the same object as the input."""
        affil = [(1, 2), (1, 2), (1, 2, 3)]
        result = remove_unique_combinations(affil)
        assert result is not affil


# ---------------------------------------------------------------------------
# 2. TestNonSingletonUnchanged
# ---------------------------------------------------------------------------


class TestNonSingletonUnchanged:
    def test_no_singletons_list_unchanged(self):
        affil = [(1, 2), (1, 2), (3,), (3,)]
        result = remove_unique_combinations(affil)
        assert result == [(1, 2), (1, 2), (3,), (3,)]

    def test_empty_tuple_unchanged(self):
        """Features with empty affiliation (no data) are never touched."""
        affil = [(), (1, 2), (1, 2), ()]
        result = remove_unique_combinations(affil)
        assert result[0] == ()
        assert result[3] == ()

    def test_empty_tuple_not_counted_as_singleton(self):
        """An empty tuple is not a singleton — it represents dropped features."""
        affil = [(), (), (1, 2), (1, 2)]
        result = remove_unique_combinations(affil)
        # The shared pattern (1,2) should not be considered for rescuing ()
        assert result[0] == ()
        assert result[1] == ()


# ---------------------------------------------------------------------------
# 3. TestMinimalCropping
# ---------------------------------------------------------------------------


class TestMinimalCropping:
    def test_removes_minimum_batches(self):
        """(1,2,3,4) should be cropped to (1,2,3) not (1,2) if (1,2,3) is shared."""
        affil = [(1, 2, 3, 4), (1, 2, 3), (1, 2, 3), (1, 2), (1, 2)]
        result = remove_unique_combinations(affil)
        # (1,2,3) costs 1 removal; (1,2) costs 2 removals → choose (1,2,3)
        assert result[0] == (1, 2, 3)

    def test_ambiguous_same_size_crops(self):
        """When two same-size crops exist, any valid one is acceptable (just not larger)."""
        # (1,2,3) can crop to (1,2) or (1,3) or (2,3) in size-2 subsets
        # Both (1,2) and (2,3) are shared; any size-2 subset is fine
        affil = [(1, 2, 3), (1, 2), (1, 2), (2, 3), (2, 3)]
        result = remove_unique_combinations(affil)
        # result[0] must be one of the shared size-2 patterns
        assert result[0] in {(1, 2), (2, 3)}

    def test_crop_only_as_far_as_needed(self):
        """A crop to size n-2 is only used if no size n-1 match exists."""
        # No size-3 shared pattern for (1,2,3,4); only (1,2) is shared
        affil = [(1, 2, 3, 4), (1, 2), (1, 2)]
        result = remove_unique_combinations(affil)
        assert result[0] == (1, 2)


# ---------------------------------------------------------------------------
# 4. TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_non_empty_unique_no_change(self):
        """When all non-empty affiliations are unique, return list unchanged."""
        affil = [(1, 2), (1, 3), (2, 3)]
        result = remove_unique_combinations(affil)
        assert result == [(1, 2), (1, 3), (2, 3)]

    def test_single_non_empty_feature(self):
        """Only one non-empty feature — it is unique by definition, no rescue."""
        affil = [(1, 2)]
        result = remove_unique_combinations(affil)
        assert result == [(1, 2)]

    def test_singleton_unreachable_stays_as_is(self):
        """If no subset of a singleton's blocks is shared, it stays unchanged."""
        # (1, 4) has no subsets that are shared (only (2,3) is shared)
        affil = [(1, 4), (2, 3), (2, 3)]
        result = remove_unique_combinations(affil)
        # (1, 4) subsets are (1,) and (4,) — neither is shared → unchanged
        assert result[0] == (1, 4)

    def test_empty_list(self):
        result = remove_unique_combinations([])
        assert result == []

    def test_all_empty_affiliations(self):
        affil = [(), (), ()]
        result = remove_unique_combinations(affil)
        assert result == [(), (), ()]

    def test_large_singleton(self):
        """A singleton with many blocks; correct crop to largest shared subset."""
        shared = (1, 2, 3, 4, 5)
        affil = [(1, 2, 3, 4, 5, 6), shared, shared]
        result = remove_unique_combinations(affil)
        assert result[0] == shared


# ---------------------------------------------------------------------------
# 5. TestHelperFunctions
# ---------------------------------------------------------------------------


class TestFindBestCrop:
    def test_finds_largest_matching_subset(self):
        non_unique = {(1, 2, 3), (1, 2)}
        best = _find_best_crop(frozenset({1, 2, 3, 4}), non_unique)
        assert best == (1, 2, 3)

    def test_returns_none_when_no_match(self):
        non_unique = {(5, 6)}
        best = _find_best_crop(frozenset({1, 2}), non_unique)
        assert best is None

    def test_single_element_subset(self):
        non_unique = {(3,)}
        best = _find_best_crop(frozenset({1, 2, 3}), non_unique)
        assert best == (3,)

    def test_exact_match_not_searched(self):
        """_find_best_crop only searches strict subsets (size < n)."""
        non_unique = {(1, 2, 3), (1, 2)}
        # Input is already (1,2,3); best crop would be (1,2)
        best = _find_best_crop(frozenset({1, 2, 3}), non_unique)
        assert best == (1, 2)
