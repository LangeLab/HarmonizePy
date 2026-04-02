"""Integration tests for the full HarmonizePy pipeline.

Tests the complete spotting → splitting → adjust → concat flow via
``harmonize()`` and validates against R HarmonizR fixtures.
"""

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
# Small test case — no missing data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_small, reason="R fixtures not generated")
class TestSmallPipeline:
    """Full pipeline on the small case (10x6, no NaN) vs R HarmonizR."""

    @pytest.mark.parametrize("mode", [1, 2, 3, 4])
    def test_combat_modes(self, mode):
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
# Medium test case — 30% structural missingness
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_medium, reason="R medium fixtures not generated")
class TestMediumPipeline:
    """Full pipeline on the medium case (100x12, 30% missing) vs R HarmonizR."""

    @pytest.mark.parametrize("mode", [1, 2, 3, 4])
    def test_combat_modes_with_missing(self, mode):
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

        # NaN pattern should match
        assert (
            result.isna().sum(axis=1).value_counts().to_dict()
            == expected.isna().sum(axis=1).value_counts().to_dict()
        )

        # Compare non-NaN values
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
        data = pd.read_csv(FIXTURE_DIR / "small_input.tsv", sep="\t", index_col=0)
        desc = pd.read_csv(FIXTURE_DIR / "small_batch.csv")
        out = str(tmp_path / "output.tsv")
        result = harmonize(data, desc, algorithm="ComBat", combat_mode=1, output_file=out)
        # Verify file was written
        loaded = pd.read_csv(out, sep="\t", index_col=0)
        np.testing.assert_allclose(loaded.values, result.values, rtol=1e-12)


# ---------------------------------------------------------------------------
# Spotting unit tests
# ---------------------------------------------------------------------------


class TestSpotting:
    def test_complete_data(self):
        """All features present in all batches → all get full affiliation."""
        data = pd.DataFrame(
            np.ones((5, 6)),
            index=[f"f{i}" for i in range(5)],
            columns=[f"s{j}" for j in range(6)],
        )
        batch_list = np.array([1, 1, 1, 2, 2, 2])
        result = build_affiliation_list(data, batch_list, batch_list, needed_values=2)
        assert all(a == (1, 2) for a in result)

    def test_one_batch_missing(self):
        """Feature with one full batch missing → only present batch in affiliation."""
        data = pd.DataFrame(
            [[1, 2, 3, np.nan, np.nan, np.nan], [1, 2, 3, 4, 5, 6]],
            index=["missing_b2", "complete"],
            columns=[f"s{j}" for j in range(6)],
        )
        batch_list = np.array([1, 1, 1, 2, 2, 2])
        result = build_affiliation_list(data, batch_list, batch_list, needed_values=2)
        assert result[0] == (1,)  # missing_b2: only batch 1
        assert result[1] == (1, 2)  # complete: both batches

    def test_needed_values_threshold(self):
        """With needed_values=2, a batch with only 1 value is excluded."""
        data = pd.DataFrame(
            [[1, np.nan, 3, 4, 5]],
            index=["f0"],
            columns=[f"s{j}" for j in range(5)],
        )
        # Batch 1 has 2 samples (s0, s1), but s1 is NaN → only 1 value
        batch_list = np.array([1, 1, 2, 2, 2])
        result_2 = build_affiliation_list(data, batch_list, batch_list, needed_values=2)
        result_1 = build_affiliation_list(data, batch_list, batch_list, needed_values=1)
        assert result_2[0] == (2,)  # batch 1 excluded (only 1 value)
        assert result_1[0] == (1, 2)  # batch 1 included (1 value is enough)

    def test_all_missing(self):
        """Feature with all NaN → empty affiliation."""
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
        data = pd.DataFrame(np.ones((3, 4)))
        desc = pd.DataFrame({"ID": ["a", "b", "c"], "sample": [1, 2, 3], "batch": [1, 1, 2]})
        with pytest.raises(ValueError, match=r"(?i)sample"):
            harmonize(data, desc)

    def test_single_batch_no_crash(self):
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
        # Single batch → data returned as-is (no adjustment possible)
        np.testing.assert_array_equal(result.values, data.values)
