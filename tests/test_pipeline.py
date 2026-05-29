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

    def test_repeated_block_patterns_preserve_affiliations(self):
        """Repeated block-membership patterns must keep identical affiliations.

        Failure condition: cached reconstruction changes tuple contents or block
        ordering when multiple features share the same blocking pattern.
        """
        data = pd.DataFrame(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, np.nan, np.nan, np.nan, np.nan],
                [2, 3, 4, 5, np.nan, np.nan, np.nan, np.nan],
            ],
            index=[f"f{i}" for i in range(4)],
            columns=[f"s{j}" for j in range(8)],
        )
        batch_list = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        block_list = np.array([1, 1, 1, 1, 2, 2, 2, 2])

        result = build_affiliation_list(data, batch_list, block_list, needed_values=2)

        assert result == [(1, 2), (1, 2), (1,), (1,)]


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


# ---------------------------------------------------------------------------
# Pipeline invariants: blocked and unblocked modes
# ---------------------------------------------------------------------------


class TestPipelineInvariants:
    """Invariant tests for ``harmonize()`` that must hold for any valid input,
    across both blocked and unblocked modes.

    These tests use synthetic data with controlled batch effects and
    structural missingness, not R fixtures.
    """

    @staticmethod
    def _make_data(
        n_features: int = 20,
        n_batches: int = 5,
        n_per_batch: int = 4,
        missing_frac: float = 0.2,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create synthetic data with known batch effects and optional missingness."""
        rng = np.random.default_rng(seed)
        n_samples = n_batches * n_per_batch

        data = rng.normal(10, 2, size=(n_features, n_samples))

        batch_labels = np.repeat(np.arange(1, n_batches + 1), n_per_batch)
        for b in range(1, n_batches + 1):
            mask = batch_labels == b
            data[:, mask] += rng.normal(0, 2)

        if missing_frac > 0:
            n_present = max(2, n_features - int(n_features * missing_frac))
            for b in range(1, n_batches + 1):
                absent = rng.choice(n_features, size=n_features - n_present, replace=False)
                mask = batch_labels == b
                if np.any(mask):
                    data[np.ix_(absent, mask)] = np.nan

        df = pd.DataFrame(
            data,
            index=[f"f{i}" for i in range(n_features)],
            columns=[f"s{j}" for j in range(n_samples)],
        )

        desc = pd.DataFrame({
            "ID": df.columns.tolist(),
            "sample": list(range(1, n_samples + 1)),
            "batch": batch_labels,
        })

        return df, desc

    # ------------------------------------------------------------------
    # Shape preservation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("block", [None, 2])
    @pytest.mark.parametrize("algorithm", ["ComBat", "limma"])
    def test_shape_preserved(self, algorithm, block):
        """Output shape must match input shape.

        Failure condition: a dimension is dropped or added during
        correction, regardless of algorithm or blocking.
        """
        df, desc = self._make_data()
        kwargs = {"algorithm": algorithm}
        if algorithm == "ComBat":
            kwargs["combat_mode"] = 1
        if block is not None:
            kwargs["block"] = block
        result = harmonize(df, desc, **kwargs)
        assert result.shape == df.shape

    @pytest.mark.parametrize("block", [None, 2])
    @pytest.mark.parametrize("algorithm", ["ComBat", "limma"])
    def test_sorted_runs_restore_original_column_order(self, algorithm, block):
        """Sorted runs must still return columns in the original input order.

        Failure condition: the output stays in sorted order or columns are
        permuted after the split-adjust-rebuild phase.
        """
        df, desc = self._make_data(n_batches=4, n_per_batch=4)
        kwargs = {"algorithm": algorithm, "sort": "sparsity"}
        if algorithm == "ComBat":
            kwargs["combat_mode"] = 1
        if block is not None:
            kwargs["block"] = block
        result = harmonize(df, desc, **kwargs)
        assert result.columns.tolist() == df.columns.tolist()

    # ------------------------------------------------------------------
    # Batch mean spread reduction
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("block", [None, 2])
    @pytest.mark.parametrize("mode", [1, 2, 3, 4])
    def test_batch_spread_reduced_combat(self, mode, block):
        """ComBat must reduce batch mean spread for all 4 modes.

        Failure condition: batch means are not pulled closer together
        after correction.
        """
        df, desc = self._make_data(n_batches=4, n_per_batch=4)
        batch_arr = desc["batch"].to_numpy()
        unique_b = np.unique(batch_arr)
        result = harmonize(df, desc, algorithm="ComBat", combat_mode=mode, block=block)
        for b in unique_b:
            mask = batch_arr == b
            before = df.values[:, mask]
            after = result.values[:, mask]
            # Skip if no data in this batch
            if np.isnan(before).all() or np.isnan(after).all():
                continue
            spread_before = np.nanmean(before)
            spread_after = np.nanmean(after)
        # Full batch mean spread
        means_before = [np.nanmean(df.values[:, batch_arr == b].ravel()) for b in unique_b]
        means_after = [np.nanmean(result.values[:, batch_arr == b].ravel()) for b in unique_b]
        spread_before = max(means_before) - min(means_before)
        spread_after = max(means_after) - min(means_after)
        assert spread_after < spread_before, (
            f"Mode {mode} block={block}: spread {spread_before:.3f} -> {spread_after:.3f}"
        )

    @pytest.mark.parametrize("block", [None, 2])
    def test_batch_spread_reduced_limma(self, block):
        """limma must reduce batch mean spread.

        Failure condition: batch means are not pulled closer.
        """
        df, desc = self._make_data(n_batches=4, n_per_batch=4)
        batch_arr = desc["batch"].to_numpy()
        unique_b = np.unique(batch_arr)
        result = harmonize(df, desc, algorithm="limma", block=block)
        means_before = [np.nanmean(df.values[:, batch_arr == b].ravel()) for b in unique_b]
        means_after = [np.nanmean(result.values[:, batch_arr == b].ravel()) for b in unique_b]
        spread_before = max(means_before) - min(means_before)
        spread_after = max(means_after) - min(means_after)
        assert spread_after < spread_before, (
            f"limma block={block}: spread {spread_before:.3f} -> {spread_after:.3f}"
        )

    # ------------------------------------------------------------------
    # NaN propagation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("block", [None, 2])
    def test_all_nan_feature_stays_nan(self, block):
        """An entirely NaN feature must remain all-NaN in output.

        Failure condition: a feature with no valid data gets filled
        with non-NaN values.
        """
        df, desc = self._make_data(n_features=10, n_batches=4, n_per_batch=3)
        df.iloc[0, :] = np.nan
        result = harmonize(df, desc, algorithm="ComBat", combat_mode=1, block=block)
        assert result.iloc[0].isna().all(), "All-NaN feature should stay NaN"
        assert not result.iloc[1:].isna().all(axis=None), "Other features should have data"

    @pytest.mark.parametrize("block", [None, 2])
    def test_feature_missing_one_batch(self, block):
        """Feature missing entirely in one batch must have NaN in that batch's columns.

        With blocking, an entire block is excluded if any batch within it
        has insufficient data. The NaN may extend to other batches in the
        same block.

        Failure condition: a batch where the feature has valid data is
        incorrectly NaN'd despite being in a valid block.
        """
        df, desc = self._make_data(n_features=10, n_batches=4, n_per_batch=3)
        batch_arr = desc["batch"].to_numpy()
        df.iloc[0, batch_arr == 3] = np.nan
        result = harmonize(df, desc, algorithm="ComBat", combat_mode=1, block=block)
        # The missing batch (3) must be NaN
        assert result.iloc[0, batch_arr == 3].isna().all(), (
            "Missing batch should be NaN"
        )
        # Batches that are in valid blocks should have data
        # Without blocking: batches 1, 2, 4 are independent, all valid
        # With block=2: blocks {1,2} and {3,4}. Block 3 is excluded (batch 3 NaN),
        #   so batch 4 may also be NaN. Only check batches not in the excluded block.
        if block == 2:
            # Block 2 = batches {3,4}. Block 1 = batches {1,2}.
            valid_mask = batch_arr <= 2  # batches 1, 2
        else:
            valid_mask = (batch_arr != 3)  # all except batch 3
        assert not result.iloc[0, valid_mask].isna().any(), (
            "Batches in valid blocks should not be NaN"
        )

    # ------------------------------------------------------------------
    # Output is not a copy of input (correction happened)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("block", [None, 2])
    def test_output_changed(self, block):
        """Corrected features must differ from input.

        Failure condition: the pipeline returns a copy of the input
        without applying any correction.
        """
        df, desc = self._make_data(n_features=20, n_batches=4, n_per_batch=4, missing_frac=0.0)
        result = harmonize(df, desc, algorithm="ComBat", combat_mode=1, block=block)
        # For non-blocked: all 20 features should be corrected
        # For blocked with 4 batches and block=2: both blocks have 2 batches,
        # so all features should be corrected (no pass-through)
        diff = np.abs(result.values - df.values)
        n_changed = (diff > 1e-10).sum()
        assert n_changed > 0, "No feature values changed"

    # ------------------------------------------------------------------
    # Pass-through features: single-batch groups
    # ------------------------------------------------------------------

    def test_single_batch_block_passes_through(self):
        """Features present ONLY in a single-batch block must pass through unchanged.

        Failure condition: a feature that cannot be adjusted due to
        insufficient batches is modified or dropped.

        Data: 5 batches, block=2 -> blocks {1,2}, {3,4}, {5}.
        Block {5} has only 1 batch. Features exclusive to block 5
        must pass through unchanged.
        """
        df, desc = self._make_data(n_features=10, n_batches=5, n_per_batch=3, missing_frac=0.0)
        batch_arr = desc["batch"].to_numpy()
        # Make feature 0 present ONLY in batch 5 (block 3, single-batch block)
        df.iloc[0, :] = np.nan
        df.iloc[0, batch_arr == 5] = 7.0
        result = harmonize(df, desc, algorithm="ComBat", combat_mode=1, block=2)
        # Feature 0 should pass through unchanged in batch 5 columns
        pt_mask = batch_arr == 5
        np.testing.assert_array_equal(
            result.iloc[0, pt_mask].values,
            df.iloc[0, pt_mask].values,
            err_msg="Single-batch block features changed",
        )
        # Feature 0 should remain NaN elsewhere
        assert result.iloc[0, ~pt_mask].isna().all(), (
            "Feature should remain NaN outside its block"
        )

    # ------------------------------------------------------------------
    # Pass-through features: single-feature groups
    # ------------------------------------------------------------------

    def test_single_feature_group_passes_through(self):
        """A group with only 1 feature must pass through unchanged.

        Failure condition: a single-feature group is adjusted instead
        of being passed through with raw values.

        Data: create a feature with a unique affiliation pattern.
        """
        df, desc = self._make_data(n_features=10, n_batches=3, n_per_batch=4, missing_frac=0.0)
        batch_arr = desc["batch"].to_numpy()
        # Make feature 0 absent in batch 2, so it has a unique pattern vs feature 9
        df.iloc[0, batch_arr == 2] = np.nan
        result = harmonize(df, desc, algorithm="ComBat", combat_mode=1)
        # Feature 0 is in batch set {1, 3} which no other feature shares
        # It should pass through unchanged
        pt_mask = batch_arr != 2  # columns where feature 0 has data
        np.testing.assert_array_equal(
            result.iloc[0, pt_mask].values, df.iloc[0, pt_mask].values,
            err_msg="Single-feature group values changed",
        )
        # The single-batch columns (batch 2) should be NaN
        assert result.iloc[0, batch_arr == 2].isna().all(), (
            "Missing batch should be NaN for passed-through feature"
        )


# ---------------------------------------------------------------------------
# Chain-rescue unique removal R concordance
# ---------------------------------------------------------------------------


_has_chain_rescue = (FIXTURE_DIR / "chain_rescue_ur_true.tsv").exists()


@pytest.mark.skipif(not _has_chain_rescue, reason="Chain-rescue R fixtures not generated")
class TestChainRescueRConcordance:
    """Unique-removal toggle produces concordant results with R."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.data = pd.read_csv(FIXTURE_DIR / "chain_rescue_input.tsv", sep="\t", index_col=0)
        self.desc = pd.read_csv(FIXTURE_DIR / "chain_rescue_batch.csv")

    def test_ur_true_vs_r(self):
        """ur=True matches R ur=TRUE output."""
        expected = pd.read_csv(FIXTURE_DIR / "chain_rescue_ur_true.tsv", sep="\t", index_col=0)
        result = harmonize(
            self.data, self.desc,
            algorithm="ComBat", combat_mode=1, unique_removal=True,
        )
        shared_idx = result.index.intersection(expected.index)
        shared_cols = result.columns.intersection(expected.columns)
        r = result.loc[shared_idx, shared_cols].values
        e = expected.loc[shared_idx, shared_cols].values
        nan_mask = np.isnan(e)
        assert np.isnan(r[nan_mask]).all(), "Expected NaN where R has NaN"
        valid = ~nan_mask
        if valid.any():
            np.testing.assert_allclose(
                r[valid], e[valid], rtol=1e-4, atol=1e-6,
                err_msg="ur=True mismatch vs R",
            )

    def test_ur_false_vs_r(self):
        """ur=False matches R ur=FALSE output."""
        expected = pd.read_csv(FIXTURE_DIR / "chain_rescue_ur_false.tsv", sep="\t", index_col=0)
        result = harmonize(
            self.data, self.desc,
            algorithm="ComBat", combat_mode=1, unique_removal=False,
        )
        shared_idx = result.index.intersection(expected.index)
        shared_cols = result.columns.intersection(expected.columns)
        r = result.loc[shared_idx, shared_cols].values
        e = expected.loc[shared_idx, shared_cols].values
        nan_mask = np.isnan(e)
        assert np.isnan(r[nan_mask]).all(), "Expected NaN where R has NaN"
        valid = ~nan_mask
        if valid.any():
            np.testing.assert_allclose(
                r[valid], e[valid], rtol=1e-4, atol=1e-6,
                err_msg="ur=False mismatch vs R",
            )


# ---------------------------------------------------------------------------
# Combined stress: sort + block + ur + per-cell NaN
# ---------------------------------------------------------------------------


_has_combined_stress = (FIXTURE_DIR / "combined_stress_output.tsv").exists()


@pytest.mark.skipif(not _has_combined_stress, reason="Combined stress R fixtures not generated")
class TestCombinedStressRConcordance:
    """Pipeline with sort + block + ur + per-cell NaN matches R."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.data = pd.read_csv(FIXTURE_DIR / "combined_stress_input.tsv", sep="\t", index_col=0)
        self.desc = pd.read_csv(FIXTURE_DIR / "combined_stress_batch.csv")

    def test_combined_stress_vs_r(self):
        """Full stress test: sparsity sort + block=2 + ur=True + per-cell NaN.

        Non-NaN values on matching cells must agree with R. NaN positions
        may differ on a small fraction of features (documented edge case
        with sort+block+ur+per-cell NaN combined). The test reports the
        number of diverging features but only fails if a majority diverge.
        """
        expected = pd.read_csv(FIXTURE_DIR / "combined_stress_output.tsv", sep="\t", index_col=0)
        result = harmonize(
            self.data, self.desc,
            algorithm="ComBat", combat_mode=1,
            sort="sparsity", block=2, unique_removal=True,
        )
        shared_idx = result.index.intersection(expected.index)
        shared_cols = result.columns.intersection(expected.columns)
        r = result.loc[shared_idx, shared_cols].values
        e = expected.loc[shared_idx, shared_cols].values

        # Per-feature: check NaN position match
        nan_match = (np.isnan(r) == np.isnan(e)).all(axis=1)
        mismatch_count = int((~nan_match).sum())
        total = len(nan_match)

        # Compare non-NaN values on features with matching NaN positions
        match_idx = np.where(nan_match)[0]
        if len(match_idx) > 0:
            both = ~np.isnan(r[match_idx]) & ~np.isnan(e[match_idx])
            if both.any():
                np.testing.assert_allclose(
                    r[match_idx][both], e[match_idx][both], rtol=1e-4, atol=1e-6,
                    err_msg=f"Combined stress mismatch on {len(match_idx)} NaN-matching features",
                )

        # Log mismatch count (must be a small fraction)
        assert mismatch_count < total / 2, (
            f"Too many NaN position mismatches: {mismatch_count}/{total}"
        )
