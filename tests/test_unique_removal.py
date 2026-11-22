"""Unit tests for harmonizepy.affiliation.remove_unique_combinations.

Sections
--------
1. TestSingletonRescue: singletons are cropped to nearest shared pattern
2. TestNonSingletonUnchanged: non-singleton and empty tuples are not modified
3. TestMinimalCropping: fewest batches removed (greedy largest-first)
4. TestEdgeCases: all unique, single feature, already shared
5. TestFindBestCrop: _find_best_crop
"""

from __future__ import annotations

from harmonizepy.affiliation import _find_best_crop, remove_unique_combinations

# ---------------------------------------------------------------------------
# 1. TestSingletonRescue
# ---------------------------------------------------------------------------


class TestSingletonRescue:
    def test_basic_singleton_cropped_to_shared_pattern(self):
        """Singleton is cropped to the nearest shared pattern.

        Failure condition: a singleton feature is not rescued or is
        cropped to an incorrect subset.
        """
        affil = [(1, 2, 3), (1, 2), (1, 2)]
        result = remove_unique_combinations(affil)
        assert result[0] == (1, 2)

    def test_non_singletons_unchanged(self):
        """Non-singleton features are not modified.

        Failure condition: already-shared patterns are altered.
        """
        affil = [(1, 2, 3), (1, 2), (1, 2)]
        result = remove_unique_combinations(affil)
        assert result[1] == (1, 2)
        assert result[2] == (1, 2)

    def test_multiple_singletons_all_rescued(self):
        """Multiple singletons are all rescuable to the same pattern.

        Failure condition: only one singleton is rescued and others
        remain untouched.
        """
        affil = [(1, 2, 3), (1, 2, 4), (1, 2), (1, 2)]
        result = remove_unique_combinations(affil)
        assert result[0] == (1, 2)
        assert result[1] == (1, 2)

    def test_singleton_with_one_batch_removed(self):
        """Singleton is cropped to the correct subset when one batch must be removed.

        Failure condition: a wrong subset is chosen.
        """
        affil = [(1, 2, 3), (1, 3), (1, 3)]
        result = remove_unique_combinations(affil)
        assert result[0] == (1, 3)

    def test_singleton_chooses_largest_shared_match(self):
        """Singleton is cropped to the largest possible shared subset.

        Failure condition: a smaller subset is chosen when a larger
        valid subset exists.
        """
        affil = [(1, 2, 3, 4), (1, 2, 3), (1, 2, 3), (1, 2), (1, 2)]
        result = remove_unique_combinations(affil)
        assert result[0] == (1, 2, 3)

    def test_result_length_unchanged(self):
        """Output length must equal input length.

        Failure condition: features are dropped or duplicated.
        """
        affil = [(1, 2, 3), (1, 2), (1, 2), (3,), (3,)]
        result = remove_unique_combinations(affil)
        assert len(result) == len(affil)

    def test_list_is_independent_copy(self):
        """Returned list must not be the same object as the input.

        Failure condition: the input list is mutated in place.
        """
        affil = [(1, 2), (1, 2), (1, 2, 3)]
        result = remove_unique_combinations(affil)
        assert result is not affil


# ---------------------------------------------------------------------------
# 2. TestNonSingletonUnchanged
# ---------------------------------------------------------------------------


class TestNonSingletonUnchanged:
    def test_no_singletons_list_unchanged(self):
        """When no singletons exist, the list is returned unchanged.

        Failure condition: non-singleton patterns are modified.
        """
        affil = [(1, 2), (1, 2), (3,), (3,)]
        result = remove_unique_combinations(affil)
        assert result == [(1, 2), (1, 2), (3,), (3,)]

    def test_empty_tuple_unchanged(self):
        """Empty affiliations (no data anywhere) are never touched.

        Failure condition: empty tuples are incorrectly rescued.
        """
        affil = [(), (1, 2), (1, 2), ()]
        result = remove_unique_combinations(affil)
        assert result[0] == ()
        assert result[3] == ()

    def test_empty_tuple_not_counted_as_singleton(self):
        """An empty tuple is not a singleton and is never rescued.

        Failure condition: empty tuples trigger the rescue path.
        """
        affil = [(), (), (1, 2), (1, 2)]
        result = remove_unique_combinations(affil)
        assert result[0] == ()
        assert result[1] == ()


# ---------------------------------------------------------------------------
# 3. TestMinimalCropping
# ---------------------------------------------------------------------------


class TestMinimalCropping:
    def test_removes_minimum_batches(self):
        """The crop removes the fewest possible batches.

        Failure condition: a deeper crop is chosen when a shallower
        valid crop exists.
        """
        affil = [(1, 2, 3, 4), (1, 2, 3), (1, 2, 3), (1, 2), (1, 2)]
        result = remove_unique_combinations(affil)
        assert result[0] == (1, 2, 3)

    def test_ambiguous_same_size_crops(self):
        """Any valid same-size crop is acceptable when multiple exist.

        Failure condition: the function produces a crop larger than
        the minimum required.
        """
        affil = [(1, 2, 3), (1, 2), (1, 2), (2, 3), (2, 3)]
        result = remove_unique_combinations(affil)
        assert result[0] in {(1, 2), (2, 3)}

    def test_crop_only_as_far_as_needed(self):
        """The crop is no deeper than necessary.

        Failure condition: a deeper crop is used when a shallower
        valid one exists.
        """
        affil = [(1, 2, 3, 4), (1, 2), (1, 2)]
        result = remove_unique_combinations(affil)
        assert result[0] == (1, 2)


# ---------------------------------------------------------------------------
# 4. TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_non_empty_unique_no_change(self):
        """When all non-empty affiliations are unique, list is unchanged.

        Failure condition: the function modifies the list when no
        rescue is possible.
        """
        affil: list[tuple[int, ...]] = [(1, 2), (1, 3), (2, 3)]
        result = remove_unique_combinations(affil)
        assert result == [(1, 2), (1, 3), (2, 3)]

    def test_single_non_empty_feature(self):
        """A single non-empty feature stays unchanged (unique by definition).

        Failure condition: the lone feature is incorrectly cropped.
        """
        affil: list[tuple[int, ...]] = [(1, 2)]
        result = remove_unique_combinations(affil)
        assert result == [(1, 2)]

    def test_singleton_unreachable_stays_as_is(self):
        """A singleton with no reachable shared subset stays unchanged.

        Failure condition: the feature is incorrectly cropped to a
        non-existent shared pattern.
        """
        affil: list[tuple[int, ...]] = [(1, 4), (2, 3), (2, 3)]
        result = remove_unique_combinations(affil)
        assert result[0] == (1, 4)

    def test_empty_list(self):
        """Empty input returns an empty list.

        Failure condition: the function crashes on an empty list.
        """
        result = remove_unique_combinations([])
        assert result == []

    def test_all_empty_affiliations(self):
        """All-empty affiliations are returned unchanged.

        Failure condition: empty tuples are processed or removed.
        """
        affil: list[tuple[int, ...]] = [(), (), ()]
        result = remove_unique_combinations(affil)
        assert result == [(), (), ()]

    def test_large_singleton(self):
        """A singleton with many blocks is cropped to the largest shared subset.

        Failure condition: a singleton with 6 blocks is cropped
        too aggressively.
        """
        shared = (1, 2, 3, 4, 5)
        affil = [(1, 2, 3, 4, 5, 6), shared, shared]
        result = remove_unique_combinations(affil)
        assert result[0] == shared


# ---------------------------------------------------------------------------
# 5. TestFindBestCrop
# ---------------------------------------------------------------------------


class TestFindBestCrop:
    def test_finds_largest_matching_subset(self):
        """Returns the largest matching subset of the input set.

        Failure condition: a non-maximal subset is returned.
        """
        non_unique: set[tuple[int, ...]] = {(1, 2, 3), (1, 2)}
        best = _find_best_crop(frozenset({1, 2, 3, 4}), [(p, frozenset(p)) for p in non_unique])
        assert best == (1, 2, 3)

    def test_returns_none_when_no_match(self):
        """Returns None when no subset matches.

        Failure condition: a non-existent match is returned.
        """
        non_unique: set[tuple[int, ...]] = {(5, 6)}
        best = _find_best_crop(frozenset({1, 2}), [(p, frozenset(p)) for p in non_unique])
        assert best is None

    def test_single_element_subset(self):
        """A single-element subset can be a valid match.

        Failure condition: single-element subsets are skipped.
        """
        non_unique: set[tuple[int, ...]] = {(3,)}
        best = _find_best_crop(frozenset({1, 2, 3}), [(p, frozenset(p)) for p in non_unique])
        assert best == (3,)

    def test_exact_match_not_searched(self):
        """Only strict subsets are searched, not the input itself.

        Failure condition: the input itself is returned when it
        is already a non-unique pattern.
        """
        non_unique: set[tuple[int, ...]] = {(1, 2, 3), (1, 2)}
        best = _find_best_crop(frozenset({1, 2, 3}), [(p, frozenset(p)) for p in non_unique])
        assert best == (1, 2)


class TestTieOrder:
    """Equally long candidates are chosen by a rule that ignores row order."""

    _AFFILIATIONS = (
        (2, 3),
        (2, 3),
        (1, 2),
        (1, 2),
        (3, 6),
        (3, 6),
        (2, 4, 5, 6),
        (2, 4, 5, 6),
        (1, 2, 3, 5),
        (2, 3, 5, 6),
        (1, 2, 4, 6),
    )

    def test_choice_is_the_same_in_every_row_order(self) -> None:
        """Every shuffle of the rows moves each feature to the same affiliation.

        Failure condition: the choice follows the iteration order of a set built
        from the rows, so it flips between ``(2, 3)`` and ``(1, 2)``.
        """
        import random

        expected = remove_unique_combinations(list(self._AFFILIATIONS))
        rng = random.Random(0)
        for _ in range(200):
            order = list(range(len(self._AFFILIATIONS)))
            rng.shuffle(order)
            result = remove_unique_combinations([self._AFFILIATIONS[i] for i in order])
            assert [result[order.index(i)] for i in range(len(order))] == expected

    def test_smallest_tuple_breaks_a_tie_of_equal_groups(self) -> None:
        """Between candidates shared by equally many features, the smaller tuple wins.

        Failure condition: the tie goes to ``(2, 3)`` in some runs.
        """
        assert remove_unique_combinations(list(self._AFFILIATIONS))[8] == (1, 2)

    def test_larger_group_wins_a_tie_of_length(self) -> None:
        """Among equally long candidates, the one more features share wins.

        Failure condition: the smaller tuple wins although its group is smaller.
        """
        affiliations = [(1, 2), (1, 2), (2, 3), (2, 3), (2, 3), (1, 2, 3)]
        assert remove_unique_combinations(affiliations)[5] == (2, 3)

    def test_harmonize_output_does_not_depend_on_row_order(self) -> None:
        """Shuffling the rows of the matrix gives the same corrected values per feature.

        Failure condition: a tie resolved by row order puts a feature in another
        group, and its corrected values change.
        """
        import numpy as np
        import pandas as pd

        from harmonizepy import harmonize

        rng = np.random.default_rng(3)
        batch = np.repeat(np.arange(1, 7), 2)
        samples = [f"s{i}" for i in range(12)]
        rows = {}
        for k, affiliation in enumerate(self._AFFILIATIONS):
            values = rng.normal(20, 1) + rng.normal(0, 0.3, 12) + 0.5 * batch
            values[~np.isin(batch, affiliation)] = np.nan
            rows[f"f{k}"] = values
        data = pd.DataFrame(rows, index=samples).T
        desc = pd.DataFrame({"ID": samples, "sample": range(1, 13), "batch": batch})
        expected = harmonize(data, desc)
        for seed in range(10):
            shuffled = harmonize(data.sample(frac=1.0, random_state=seed), desc)
            pd.testing.assert_frame_equal(shuffled.loc[expected.index], expected, rtol=1e-12)


class TestOneBatchTargets:
    """A crop never lands in an affiliation that spans a single batch."""

    def test_one_batch_target_is_refused(self) -> None:
        """A feature is not moved into a pattern of one batch.

        Failure condition: the feature loses its other batches and is still left raw.
        """
        assert remove_unique_combinations([(2,), (2,), (2, 3)]) == [(2,), (2,), (2, 3)]

    def test_one_block_of_two_batches_is_a_target(self) -> None:
        """With blocking, a single block that holds two batches is still a target.

        Failure condition: blocks are counted as batches and the crop is refused.
        """
        result = remove_unique_combinations([(2,), (2,), (2, 3)], {2: 2, 3: 1})
        assert result == [(2,), (2,), (2,)]

    def test_harmonize_keeps_the_feature_whole(self) -> None:
        """Proteins A and B measured in batch 2 only, C in batches 2 and 3: C keeps
        its batch 3 values and is reported alone in its pattern.

        Failure condition: C is cropped to batch 2, its batch 3 values are emptied,
        and it still comes out raw.
        """
        import numpy as np
        import pandas as pd

        from harmonizepy import harmonize

        samples = [f"s{i}" for i in range(9)]
        batch = np.repeat([1, 2, 3], 3)
        rng = np.random.default_rng(5)
        data = pd.DataFrame(np.nan, index=["A", "B", "C"], columns=samples)
        data.loc[["A", "B"], np.array(samples)[batch == 2]] = rng.normal(10, 1, (2, 3))
        data.loc["C", np.array(samples)[batch >= 2]] = rng.normal(12, 1, 6)
        desc = pd.DataFrame({"ID": samples, "sample": range(1, 10), "batch": batch})
        out, report = harmonize(data, desc, return_report=True)
        pd.testing.assert_series_equal(out.loc["C"], data.loc["C"])
        assert report.loc["C", "status"] == "raw_alone_in_pattern"
        assert not report.loc["C", "cropped"]
