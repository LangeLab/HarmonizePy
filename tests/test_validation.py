"""Direct unit tests for harmonizepy.validation.

Every public validation function is tested in isolation, not just through
downstream callers.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from harmonizepy.validation import (
    sample_batch_map,
    validate_combat_input,
    validate_data_matrix,
    validate_description,
    validate_harmonize_args,
    validate_limma_input,
)

# ---------------------------------------------------------------------------
# validate_data_matrix
# ---------------------------------------------------------------------------


class TestValidateDataMatrix:
    def test_valid(self) -> None:
        """Valid data matrix passes without error.

        Failure condition: validation rejects a well-formed input.
        """
        df = pd.DataFrame({"s1": [1.0, 2.0], "s2": [3.0, 4.0]})
        validate_data_matrix(df)  # no error

    def test_duplicate_features(self) -> None:
        """Duplicate feature names raise ValueError.

        Failure condition: duplicates are accepted silently.
        """
        df = pd.DataFrame({"s1": [1.0, 2.0]}, index=["dup", "dup"])
        with pytest.raises(ValueError, match="duplicate feature"):
            validate_data_matrix(df)

    def test_non_numeric_column(self) -> None:
        """Non-numeric columns raise ValueError.

        Failure condition: string or object columns are accepted.
        """
        df = pd.DataFrame({"s1": [1.0, 2.0], "s2": ["a", "b"]})
        with pytest.raises(ValueError, match="non-numeric"):
            validate_data_matrix(df)

    def test_too_few_columns(self) -> None:
        """Fewer than 2 sample columns raises ValueError.

        Failure condition: a single-column matrix is accepted.
        """
        df = pd.DataFrame({"s1": [1.0, 2.0]})
        with pytest.raises(ValueError, match="at least 2 sample columns"):
            validate_data_matrix(df)

    def test_infinite_value_raises_naming_the_feature(self) -> None:
        """An infinite value is rejected, and the message names its feature.

        Failure condition: the matrix passes and the engine turns every value missing.
        """
        df = pd.DataFrame({"s1": [1.0, -np.inf], "s2": [2.0, 3.0]}, index=["p1", "p2"])
        with pytest.raises(
            ValueError, match=r"infinite values in 1 feature\(s\) \(first: \['p2'\]\)"
        ):
            validate_data_matrix(df)


# ---------------------------------------------------------------------------
# validate_description
# ---------------------------------------------------------------------------


class TestValidateDescription:
    def test_valid(self) -> None:
        """Valid description passes without error.

        Failure condition: a correct description is rejected.
        """
        data = pd.DataFrame({"s1": [1.0], "s2": [2.0]})
        desc = pd.DataFrame({"ID": ["s1", "s2"], "sample": [1, 2], "batch": [1, 2]})
        validate_description(desc, data)  # no error

    def test_mismatched_ids(self) -> None:
        """Mismatched sample IDs between description and data raise ValueError.

        Failure condition: a description with IDs that do not match the
        data columns is accepted.
        """
        data = pd.DataFrame({"s1": [1.0], "s3": [2.0]})
        desc = pd.DataFrame({"ID": ["s1", "s2"], "sample": [1, 2], "batch": [1, 2]})
        with pytest.raises(ValueError, match="Sample IDs"):
            validate_description(desc, data)

    def test_too_few_columns(self) -> None:
        """Description with fewer than 3 columns raises ValueError.

        Failure condition: a description missing the batch column is accepted.
        """
        data = pd.DataFrame({"s1": [1.0]})
        desc = pd.DataFrame({"ID": ["s1"]})
        with pytest.raises(ValueError, match="at least 3 columns"):
            validate_description(desc, data)


# ---------------------------------------------------------------------------
# validate_combat_input
# ---------------------------------------------------------------------------


class TestValidateCombatInput:
    def test_valid(self) -> None:
        """Valid combat input passes without error.

        Failure condition: a valid array-batch pair is rejected.
        """
        data = np.ones((5, 4), dtype=np.float64)
        batch = np.array([0, 0, 1, 1], dtype=np.intp)
        validate_combat_input(data, batch)  # no error

    def test_contains_nan_passes(self) -> None:
        """NaN in data does NOT raise ValueError.

        Failure condition: the validator rejects NaN input
        (per-cell NaN is handled by combat() via row-dropping,
        matching R sva::ComBat's na.omit).
        """
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        batch = np.array([0, 1], dtype=np.intp)
        validate_combat_input(data, batch)  # should not raise

    def test_no_min_feature_check(self) -> None:
        """validate_combat_input does NOT reject n_features < 2.

        Failure condition: the validator rejects input with < 2 features
        (combat() handles this internally via NaN row dropping or
        pass-through).

        Per-cell NaN may reduce clean features below 2; the validator
        must let the engine decide.
        """
        data = np.ones((1, 4), dtype=np.float64)
        batch = np.array([0, 0, 1, 1], dtype=np.intp)
        validate_combat_input(data, batch)  # should not raise

    def test_one_dimensional(self) -> None:
        """1-D input raises ValueError.

        Failure condition: a flat array is not rejected as 2-D required.
        """
        data = np.array([1.0, 2.0, 3.0, 4.0])
        batch = np.array([0, 0, 1, 1], dtype=np.intp)
        with pytest.raises(ValueError, match="2-D"):
            validate_combat_input(data, batch)

    def test_batch_length_mismatch(self) -> None:
        """Mismatched batch length raises ValueError.

        Failure condition: batch labels do not match the sample dimension
        and the mismatch is not detected.
        """
        data = np.ones((3, 5), dtype=np.float64)
        batch = np.array([0, 0, 1], dtype=np.intp)
        with pytest.raises(ValueError, match="batch length"):
            validate_combat_input(data, batch)


# ---------------------------------------------------------------------------
# validate_limma_input
# ---------------------------------------------------------------------------


class TestValidateLimmaInput:
    def test_valid(self) -> None:
        """Valid limma input passes without error.

        Failure condition: a valid array-batch pair is rejected.
        """
        data = np.ones((5, 4), dtype=np.float64)
        batch = np.array([0, 0, 1, 1])
        validate_limma_input(data, batch)  # no error

    def test_contains_nan_passes(self) -> None:
        """NaN in data does NOT raise ValueError.

        Failure condition: the validator rejects NaN input
        (per-cell NaN is handled by the engine via row-dropping,
        matching R sva::ComBat's na.omit).
        """
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        batch = np.array([0, 1])
        validate_limma_input(data, batch)  # should not raise

    def test_one_dimensional(self) -> None:
        """1-D input raises ValueError.

        Failure condition: a flat array is not rejected.
        """
        data = np.array([1.0, 2.0, 3.0])
        batch = np.array([0, 1, 1])
        with pytest.raises(ValueError, match="2-D"):
            validate_limma_input(data, batch)

    def test_batch_length_mismatch(self) -> None:
        """Mismatched batch length raises ValueError.

        Failure condition: batch-sample mismatch is not detected.
        """
        data = np.ones((3, 5), dtype=np.float64)
        batch = np.array([0, 0, 1])
        with pytest.raises(ValueError, match="batch length"):
            validate_limma_input(data, batch)


# ---------------------------------------------------------------------------
# validate_harmonize_args
# ---------------------------------------------------------------------------


class TestValidateHarmonizeArgs:
    def test_valid_defaults(self) -> None:
        """Default arguments pass without error.

        Failure condition: valid defaults are rejected.
        """
        validate_harmonize_args("ComBat", 1, 2)  # no error

    def test_invalid_algorithm(self) -> None:
        """Invalid algorithm name raises ValueError.

        Failure condition: a bogus algorithm string is accepted.
        """
        with pytest.raises(ValueError, match="algorithm"):
            validate_harmonize_args("invalid", 1, 2)

    def test_invalid_mode(self) -> None:
        """Invalid combat mode raises ValueError.

        Failure condition: out-of-range mode is accepted.
        """
        with pytest.raises(ValueError, match="combat_mode"):
            validate_harmonize_args("ComBat", 5, 2)

    def test_needed_values_zero(self) -> None:
        """needed_values less than 1 raises ValueError.

        Failure condition: zero is accepted as a valid threshold.
        """
        with pytest.raises(ValueError, match="needed_values"):
            validate_harmonize_args("ComBat", 1, 0)

    def test_invalid_sort_strategy(self) -> None:
        """Unknown sort strategy raises ValueError.

        Failure condition: a nonsense strategy string is accepted.
        """
        with pytest.raises(ValueError, match="sort"):
            validate_harmonize_args("ComBat", 1, 2, sort_strategy="invalid")

    def test_block_size_too_small(self) -> None:
        """block_size less than 2 raises ValueError.

        Failure condition: block_size=1 is accepted.
        """
        with pytest.raises(ValueError, match="block"):
            validate_harmonize_args("ComBat", 1, 2, block_size=1, n_batches=5)

    def test_block_size_equals_n_batches(self) -> None:
        """block_size equal to n_batches raises ValueError.

        Failure condition: a block spanning all batches is accepted.
        """
        with pytest.raises(ValueError, match="block"):
            validate_harmonize_args("ComBat", 1, 2, block_size=5, n_batches=5)

    def test_unique_removal_not_bool(self) -> None:
        """Non-bool unique_removal raises TypeError.

        Failure condition: a string is accepted where bool is required.
        """
        with pytest.raises(TypeError, match="unique_removal"):
            validate_harmonize_args("ComBat", 1, 2, unique_removal="true")  # type: ignore[arg-type]

    def test_valid_sort_strategies(self) -> None:
        """All three valid sort strategies are accepted.

        Failure condition: a known strategy string is rejected.
        """
        for s in ("sparsity", "jaccard", "seriation"):
            validate_harmonize_args("ComBat", 1, 2, sort_strategy=s)

    def test_valid_block_size(self) -> None:
        """Valid block sizes (2 and 4) pass without error.

        Failure condition: a valid block size is rejected.
        """
        validate_harmonize_args("ComBat", 1, 2, block_size=2, n_batches=5)
        validate_harmonize_args("ComBat", 1, 2, block_size=4, n_batches=5)


class TestDescriptionColumns:
    """One rule finds the ID and batch columns for validation and for the batch map."""

    _DATA: ClassVar[pd.DataFrame] = pd.DataFrame(np.zeros((1, 4)), columns=["s1", "s2", "s3", "s4"])

    def test_id_by_name_without_batch_name_uses_third_column(self) -> None:
        """With an ``ID`` column but no ``batch`` column, the third column holds the labels.

        Failure condition: validation finds IDs by name while the map reads them by
        position, so samples get another sample's batch.
        """
        desc = pd.DataFrame(
            {"sample": [4, 3, 2, 1], "ID": ["s1", "s2", "s3", "s4"], "group": [1, 1, 2, 2]}
        )
        validate_description(desc, self._DATA)
        assert sample_batch_map(desc) == {"s1": 1, "s2": 1, "s3": 2, "s4": 2}

    def test_id_in_third_column_without_batch_name_raises(self) -> None:
        """If the ID column is also the third column, the batch column cannot be found.

        Failure condition: the IDs are read as batch labels.
        """
        desc = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 1, 2, 2], "ID": ["s1", "s2", "s3", "s4"]})
        with pytest.raises(ValueError, match="Name the batch column 'batch'"):
            validate_description(desc, self._DATA)

    @pytest.mark.parametrize("labels", [[1, 1, 1.5, 2], [1, 1, "A", 2], [1, 1, None, 2]])
    def test_non_whole_batch_labels_raise(self, labels: list[object]) -> None:
        """Fractions, text, and blanks in the batch column are rejected by name.

        Failure condition: 1.5 is truncated to 1, or text fails later with an error
        that names no column.
        """
        desc = pd.DataFrame(
            {"ID": ["s1", "s2", "s3", "s4"], "sample": [1, 2, 3, 4], "batch": labels}
        )
        with pytest.raises(ValueError, match="column 'batch' must be whole numbers"):
            validate_description(desc, self._DATA)

    def test_whole_float_labels_accepted(self) -> None:
        """Labels written as 1.0 and 2.0 are accepted as batches 1 and 2.

        Failure condition: whole numbers stored as floats are rejected.
        """
        desc = pd.DataFrame(
            {"ID": ["s1", "s2", "s3", "s4"], "sample": [1, 2, 3, 4], "batch": [1.0, 1.0, 2.0, 2.0]}
        )
        validate_description(desc, self._DATA)
        assert sample_batch_map(desc) == {"s1": 1, "s2": 1, "s3": 2, "s4": 2}

    def test_repeated_sample_id_raises(self) -> None:
        """A sample listed twice is rejected.

        Failure condition: the later row silently wins.
        """
        desc = pd.DataFrame(
            {
                "ID": ["s1", "s1", "s2", "s3", "s4"],
                "sample": [1, 2, 3, 4, 5],
                "batch": [1, 2, 1, 2, 2],
            }
        )
        with pytest.raises(ValueError, match="listed more than once"):
            validate_description(desc, self._DATA)

    def test_batch_first_without_id_name_raises_naming_the_id_column(self) -> None:
        """A first column named ``batch`` with no ``ID`` column is reported as a missing ID column.

        Failure condition: the message blames the third column, which is unrelated.
        """
        desc = pd.DataFrame(
            {"batch": [1, 1, 2, 2], "sample_id": ["s1", "s2", "s3", "s4"], "x": [0, 0, 0, 0]}
        )
        with pytest.raises(ValueError, match="the ID column cannot be found"):
            validate_description(desc, self._DATA)

    def test_large_integer_labels_stay_distinct(self) -> None:
        """Integer labels above 2**53 keep their exact values.

        Failure condition: the labels pass through float64 and two batches merge.
        """
        big = 2**53
        desc = pd.DataFrame(
            {
                "ID": ["s1", "s2", "s3", "s4"],
                "sample": [1, 2, 3, 4],
                "batch": [big + 1, big + 1, big, 3],
            }
        )
        validate_description(desc, self._DATA)
        assert sample_batch_map(desc) == {"s1": big + 1, "s2": big + 1, "s3": big, "s4": 3}

    def test_float_label_beyond_exact_range_raises(self) -> None:
        """A float label at or beyond 2**53 is rejected, since it may already be rounded.

        Failure condition: 1e19 is cast to a wrapped-around integer.
        """
        desc = pd.DataFrame(
            {
                "ID": ["s1", "s2", "s3", "s4"],
                "sample": [1, 2, 3, 4],
                "batch": [1e19, 1e19, 2.0, 2.0],
            }
        )
        with pytest.raises(ValueError, match="must be whole numbers"):
            validate_description(desc, self._DATA)

    def test_map_rejects_repeated_ids(self) -> None:
        """``sample_batch_map`` rejects a repeated sample ID when called on its own.

        Failure condition: the later row silently wins.
        """
        desc = pd.DataFrame({"ID": ["s1", "s1", "s2"], "sample": [1, 2, 3], "batch": [1, 2, 1]})
        with pytest.raises(ValueError, match="listed more than once"):
            sample_batch_map(desc)


def test_duplicate_sample_names_raise() -> None:
    """A repeated sample name in the data matrix is rejected.

    Failure condition: the matrix passes, and a sorted run fails with a shape error
    or an unsorted run silently maps both columns to one description row.
    """
    df = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["s1", "s1", "s2"])
    with pytest.raises(ValueError, match="duplicate sample names"):
        validate_data_matrix(df)
