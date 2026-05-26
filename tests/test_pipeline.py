"""Integration tests for the full HarmonizePy pipeline.

Tests the complete spotting → splitting → adjust → concat flow via
``harmonize()`` and validates against R HarmonizR fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harmonizepy.affiliation import build_affiliation_list
from harmonizepy.core import harmonize

FIXTURE_DIR = Path(__file__).parent / "fixtures"
_has_small = (FIXTURE_DIR / "small_combat_mode1.tsv").exists()
_has_medium = (FIXTURE_DIR / "medium_harmonizr_combat_mode1.tsv").exists()


# ---------------------------------------------------------------------------
# Small test case: no missing data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_small, reason="R fixtures not generated")
class TestSmallPipeline:
    """Full pipeline on the small case (10x6, no NaN) vs R HarmonizR."""

    @pytest.mark.parametrize("mode", [1, 2, 3, 4])
    def test_combat_modes(self, mode):
        """Pipeline output must match R sva::ComBat output for all 4 modes.

        Failure condition: the pipeline produces different corrected
        values than the reference R implementation.

        Tolerances: rtol=1e-4/atol=1e-6 for parametric iterative modes
        (1, 3); closed-form modes (2, 4) converge tighter but this
        single tolerance covers all modes safely.
        """
        result = harmonize(
            str(FIXTURE_DIR / "small_input.tsv"),
            str(FIXTURE_DIR / "small_batch.csv"),
            algorithm="ComBat",
            combat_mode=mode,
        )
        expected = pd.read_csv(
            FIXTURE_DIR / f"small_combat_mode{mode}.tsv",
            sep="\t",
            index_col=0,
        )
        assert result.shape == expected.shape
        assert result.isna().sum().sum() == 0
        np.testing.assert_allclose(
            result.values,
            expected.values,
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"Pipeline mismatch vs R sva::ComBat mode {mode}",
        )

    def test_limma(self):
        """Pipeline output must match R limma::removeBatchEffect output.

        Failure condition: the linear-model subtraction produces
        different values than R's reference implementation.

        Tolerances: rtol=1e-10/atol=1e-10 at float64 machine epsilon
        for this closed-form OLS solution.
        """
        result = harmonize(
            str(FIXTURE_DIR / "small_input.tsv"),
            str(FIXTURE_DIR / "small_batch.csv"),
            algorithm="limma",
        )
        expected = pd.read_csv(
            FIXTURE_DIR / "small_limma.tsv",
            sep="\t",
            index_col=0,
        )
        assert result.shape == expected.shape
        np.testing.assert_allclose(
            result.values,
            expected.values,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Pipeline mismatch vs R limma",
        )


# ---------------------------------------------------------------------------
# Medium test case: 30% structural missingness
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_medium, reason="R medium fixtures not generated")
class TestMediumPipeline:
    """Full pipeline on the medium case (100x12, 30% missing) vs R HarmonizR."""

    @pytest.mark.parametrize("mode", [1, 2, 3, 4])
    def test_combat_modes_with_missing(self, mode):
        """Pipeline output with structural missingness must match R HarmonizR.

        Failure condition: NaN placement or corrected values diverge
        from R's handling of partial batch presence.

        Tolerances: rtol=1e-4/atol=1e-6 same as small dataset; the
        iterative EB solver has identical convergence criteria.
        """
        result = harmonize(
            str(FIXTURE_DIR / "medium_input.tsv"),
            str(FIXTURE_DIR / "medium_batch.csv"),
            algorithm="ComBat",
            combat_mode=mode,
        )
        expected = pd.read_csv(
            FIXTURE_DIR / f"medium_harmonizr_combat_mode{mode}.tsv",
            sep="\t",
            index_col=0,
        )
        assert result.shape == expected.shape

        assert (
            result.isna().sum(axis=1).value_counts().to_dict()
            == expected.isna().sum(axis=1).value_counts().to_dict()
        )

        common = result.index.intersection(expected.index)
        r = result.loc[common]
        e = expected.loc[common]
        mask = r.notna() & e.notna()
        np.testing.assert_allclose(
            r.values[mask.values],
            e.values[mask.values],
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"Pipeline mismatch vs HarmonizR ComBat mode {mode} (medium)",
        )


# ---------------------------------------------------------------------------
# DataFrame input (not file paths)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_small, reason="R fixtures not generated")
class TestDataFrameInput:
    def test_dataframe_input(self):
        """DataFrame input must produce identical results to file-path input.

        Failure condition: the internal code path for DataFrames differs
        from the file-path path (e.g. index handling, dtype casting).

        Tolerances: rtol=1e-4/atol=1e-6 match the R concordance tolerance.
        """
        data = pd.read_csv(FIXTURE_DIR / "small_input.tsv", sep="\t", index_col=0)
        desc = pd.read_csv(FIXTURE_DIR / "small_batch.csv")
        result = harmonize(data, desc, algorithm="ComBat", combat_mode=1)
        expected = pd.read_csv(
            FIXTURE_DIR / "small_combat_mode1.tsv",
            sep="\t",
            index_col=0,
        )
        np.testing.assert_allclose(
            result.values,
            expected.values,
            rtol=1e-4,
            atol=1e-6,
        )

    def test_output_file(self, tmp_path):
        """output_file must write a TSV that matches the in-memory result.

        Failure condition: the serialized file differs from the returned
        DataFrame, e.g. due to separator or index handling issues.

        Tolerances: rtol=1e-12 for identical float64 data through TSV
        text serialization at 17 significant digits.
        """
        data = pd.read_csv(FIXTURE_DIR / "small_input.tsv", sep="\t", index_col=0)
        desc = pd.read_csv(FIXTURE_DIR / "small_batch.csv")
        out = str(tmp_path / "output.tsv")
        result = harmonize(data, desc, algorithm="ComBat", combat_mode=1, output_file=out)
        loaded = pd.read_csv(out, sep="\t", index_col=0)
        np.testing.assert_allclose(loaded.values, result.values, rtol=1e-12)


# ---------------------------------------------------------------------------
# Spotting unit tests
# ---------------------------------------------------------------------------


class TestSpotting:
    def test_complete_data(self):
        """All features present in all batches get the full affiliation.

        Failure condition: a fully present feature has a partial or
        empty affiliation.
        """
        data = pd.DataFrame(
            np.ones((5, 6)),
            index=[f"f{i}" for i in range(5)],
            columns=[f"s{j}" for j in range(6)],
        )
        batch_list = np.array([1, 1, 1, 2, 2, 2])
        result = build_affiliation_list(data, batch_list, batch_list, needed_values=2)
        assert all(a == (1, 2) for a in result)

    def test_one_batch_missing(self):
        """A feature missing one batch entirely has a partial affiliation.

        Failure condition: the missing batch is still included in the
        affiliation tuple.
        """
        data = pd.DataFrame(
            [[1, 2, 3, np.nan, np.nan, np.nan], [1, 2, 3, 4, 5, 6]],
            index=["missing_b2", "complete"],
            columns=[f"s{j}" for j in range(6)],
        )
        batch_list = np.array([1, 1, 1, 2, 2, 2])
        result = build_affiliation_list(data, batch_list, batch_list, needed_values=2)
        assert result[0] == (1,)
        assert result[1] == (1, 2)

    def test_needed_values_threshold(self):
        """needed_values threshold correctly includes or excludes batches.

        Failure condition: a batch with insufficient observations is
        included (nv=2) or a batch with sufficient is excluded (nv=1).
        """
        data = pd.DataFrame(
            [[1, np.nan, 3, 4, 5]],
            index=["f0"],
            columns=[f"s{j}" for j in range(5)],
        )
        batch_list = np.array([1, 1, 2, 2, 2])
        result_2 = build_affiliation_list(data, batch_list, batch_list, needed_values=2)
        result_1 = build_affiliation_list(data, batch_list, batch_list, needed_values=1)
        assert result_2[0] == (2,)
        assert result_1[0] == (1, 2)

    def test_all_missing(self):
        """An all-NaN feature gets an empty affiliation tuple.

        Failure condition: an all-NaN feature is assigned a non-empty
        affiliation, which would cause it to enter adjustment with
        no valid data.
        """
        data = pd.DataFrame(
            [[np.nan] * 4],
            index=["empty"],
            columns=[f"s{j}" for j in range(4)],
        )
        batch_list = np.array([1, 1, 2, 2])
        result = build_affiliation_list(data, batch_list, batch_list, needed_values=1)
        assert result[0] == ()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPipelineEdgeCases:
    def test_description_mismatch_raises(self):
        """Mismatched sample IDs between data and description raise ValueError.

        Failure condition: mismatched IDs are silently accepted.
        """
        data = pd.DataFrame(np.ones((3, 4)))
        desc = pd.DataFrame({"ID": ["a", "b", "c"], "sample": [1, 2, 3], "batch": [1, 1, 2]})
        with pytest.raises(ValueError, match=r"(?i)sample"):
            harmonize(data, desc)

    def test_single_batch_no_crash(self):
        """Single-batch input is returned unchanged, not crashed.

        Failure condition: the pipeline crashes on single-batch data
        or modifies the values.
        """
        data = pd.DataFrame(
            np.random.default_rng(1).normal(10, 2, (5, 4)),
            index=[f"f{i}" for i in range(5)],
            columns=[f"s{j}" for j in range(4)],
        )
        desc = pd.DataFrame(
            {
                "ID": [f"s{j}" for j in range(4)],
                "sample": [1, 2, 3, 4],
                "batch": [1, 1, 1, 1],
            }
        )
        result = harmonize(data, desc)
        np.testing.assert_array_equal(result.values, data.values)
