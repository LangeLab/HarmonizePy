"""Direct unit tests for harmonizepy.validation.

Every public validation function is tested in isolation, not just through
downstream callers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harmonizepy.validation import (
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
        df = pd.DataFrame({"s1": [1.0, 2.0], "s2": [3.0, 4.0]})
        validate_data_matrix(df)  # no error

    def test_duplicate_features(self) -> None:
        df = pd.DataFrame({"s1": [1.0, 2.0]}, index=["dup", "dup"])
        with pytest.raises(ValueError, match="duplicate feature"):
            validate_data_matrix(df)

    def test_non_numeric_column(self) -> None:
        df = pd.DataFrame({"s1": [1.0, 2.0], "s2": ["a", "b"]})
        with pytest.raises(ValueError, match="non-numeric"):
            validate_data_matrix(df)

    def test_too_few_columns(self) -> None:
        df = pd.DataFrame({"s1": [1.0, 2.0]})
        with pytest.raises(ValueError, match="at least 2 sample columns"):
            validate_data_matrix(df)


# ---------------------------------------------------------------------------
# validate_description
# ---------------------------------------------------------------------------


class TestValidateDescription:
    def test_valid(self) -> None:
        data = pd.DataFrame({"s1": [1.0], "s2": [2.0]})
        desc = pd.DataFrame({"ID": ["s1", "s2"], "sample": [1, 2], "batch": [1, 2]})
        validate_description(desc, data)  # no error

    def test_mismatched_ids(self) -> None:
        data = pd.DataFrame({"s1": [1.0], "s3": [2.0]})
        desc = pd.DataFrame({"ID": ["s1", "s2"], "sample": [1, 2], "batch": [1, 2]})
        with pytest.raises(ValueError, match="Sample IDs"):
            validate_description(desc, data)

    def test_too_few_columns(self) -> None:
        data = pd.DataFrame({"s1": [1.0]})
        desc = pd.DataFrame({"ID": ["s1"]})
        with pytest.raises(ValueError, match="at least 3 columns"):
            validate_description(desc, data)


# ---------------------------------------------------------------------------
# validate_combat_input
# ---------------------------------------------------------------------------


class TestValidateCombatInput:
    def test_valid(self) -> None:
        data = np.ones((5, 4), dtype=np.float64)
        batch = np.array([0, 0, 1, 1], dtype=np.intp)
        validate_combat_input(data, batch)  # no error

    def test_contains_nan(self) -> None:
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        batch = np.array([0, 1], dtype=np.intp)
        with pytest.raises(ValueError, match="NaN"):
            validate_combat_input(data, batch)

    def test_too_few_features(self) -> None:
        data = np.ones((1, 4), dtype=np.float64)
        batch = np.array([0, 0, 1, 1], dtype=np.intp)
        with pytest.raises(ValueError, match="at least 2 features"):
            validate_combat_input(data, batch)

    def test_one_dimensional(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0])
        batch = np.array([0, 0, 1, 1], dtype=np.intp)
        with pytest.raises(ValueError, match="2-D"):
            validate_combat_input(data, batch)

    def test_batch_length_mismatch(self) -> None:
        data = np.ones((3, 5), dtype=np.float64)
        batch = np.array([0, 0, 1], dtype=np.intp)
        with pytest.raises(ValueError, match="batch length"):
            validate_combat_input(data, batch)


# ---------------------------------------------------------------------------
# validate_limma_input
# ---------------------------------------------------------------------------


class TestValidateLimmaInput:
    def test_valid(self) -> None:
        data = np.ones((5, 4), dtype=np.float64)
        batch = np.array([0, 0, 1, 1])
        validate_limma_input(data, batch)  # no error

    def test_contains_nan(self) -> None:
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        batch = np.array([0, 1])
        with pytest.raises(ValueError, match="NaN"):
            validate_limma_input(data, batch)

    def test_one_dimensional(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        batch = np.array([0, 1, 1])
        with pytest.raises(ValueError, match="2-D"):
            validate_limma_input(data, batch)

    def test_batch_length_mismatch(self) -> None:
        data = np.ones((3, 5), dtype=np.float64)
        batch = np.array([0, 0, 1])
        with pytest.raises(ValueError, match="batch length"):
            validate_limma_input(data, batch)


# ---------------------------------------------------------------------------
# validate_harmonize_args
# ---------------------------------------------------------------------------


class TestValidateHarmonizeArgs:
    def test_valid_defaults(self) -> None:
        validate_harmonize_args("ComBat", 1, 2)  # no error

    def test_invalid_algorithm(self) -> None:
        with pytest.raises(ValueError, match="algorithm"):
            validate_harmonize_args("invalid", 1, 2)

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="combat_mode"):
            validate_harmonize_args("ComBat", 5, 2)

    def test_needed_values_zero(self) -> None:
        with pytest.raises(ValueError, match="needed_values"):
            validate_harmonize_args("ComBat", 1, 0)

    def test_invalid_sort_strategy(self) -> None:
        with pytest.raises(ValueError, match="sort"):
            validate_harmonize_args("ComBat", 1, 2, sort_strategy="invalid")

    def test_block_size_too_small(self) -> None:
        with pytest.raises(ValueError, match="block"):
            validate_harmonize_args("ComBat", 1, 2, block_size=1, n_batches=5)

    def test_block_size_equals_n_batches(self) -> None:
        with pytest.raises(ValueError, match="block"):
            validate_harmonize_args("ComBat", 1, 2, block_size=5, n_batches=5)

    def test_unique_removal_not_bool(self) -> None:
        with pytest.raises(TypeError, match="unique_removal"):
            validate_harmonize_args("ComBat", 1, 2, unique_removal="true")  # type: ignore[arg-type]

    def test_valid_sort_strategies(self) -> None:
        for s in ("sparsity", "jaccard", "seriation"):
            validate_harmonize_args("ComBat", 1, 2, sort_strategy=s)

    def test_valid_block_size(self) -> None:
        validate_harmonize_args("ComBat", 1, 2, block_size=2, n_batches=5)
        validate_harmonize_args("ComBat", 1, 2, block_size=4, n_batches=5)
