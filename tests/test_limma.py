"""Tests for the limma-style removeBatchEffect implementation.

Validates:
1. Basic correctness (shape, no NaN, batch-mean reduction).
2. R concordance against limma::removeBatchEffect fixtures.
3. Edge cases.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harmonizepy.limma_wrapper import adjust_limma, remove_batch_effect

FIXTURE_DIR = Path(__file__).parent / "fixtures"
_has_r_fixtures = (FIXTURE_DIR / "small_limma.tsv").exists()


def make_test_data(n_proteins=50, n_samples_per_batch=5, n_batches=3, seed=42):
    """Generate synthetic data with known batch shifts."""
    rng = np.random.default_rng(seed)
    n_samples = n_samples_per_batch * n_batches
    data = rng.normal(loc=10, scale=2, size=(n_proteins, n_samples))
    batch_labels = np.repeat(range(n_batches), n_samples_per_batch)
    shifts = [0.0, 2.5, -1.8]
    for b in range(n_batches):
        data[:, batch_labels == b] += shifts[b]
    return data, batch_labels


def _remove_batch_effect_nan_reference(data: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """Per-feature limma reference: sum-to-zero contrasts over the batches each feature is observed in.

    A feature observed in one batch only is returned as given.
    """
    corrected = data.copy()
    for row_index in range(data.shape[0]):
        y = data[row_index, :]
        valid = ~np.isnan(y)
        labels, batch_idx = np.unique(batch[valid], return_inverse=True)
        n_batch = len(labels)
        if n_batch < 2:
            continue
        x_batch = np.zeros((int(valid.sum()), n_batch - 1), dtype=np.float64)
        for j in range(n_batch - 1):
            x_batch[batch_idx == j, j] = 1.0
        x_batch[batch_idx == n_batch - 1, :] = -1.0
        design = np.hstack([np.ones((int(valid.sum()), 1)), x_batch])
        beta = np.linalg.lstsq(design, y[valid], rcond=None)[0]
        corrected[row_index, valid] = y[valid] - x_batch @ beta[1:]
    return corrected


class TestLimmaBasic:
    def test_shape_and_no_nan(self):
        """Output shape must match input and must not contain NaN.

        Failure condition: a dimension is dropped or NaN is produced.
        """
        data, batches = make_test_data()
        result = remove_batch_effect(data, batches)
        assert result.shape == data.shape
        assert not np.isnan(result).any()

    def test_batch_mean_reduction(self):
        """limma must reduce batch mean spread after correction.

        Failure condition: the OLS fit does not subtract batch effects,
        leaving the original spread intact.
        """
        data, batches = make_test_data()
        result = remove_batch_effect(data, batches)
        n_batches = 3
        means_before = [data[:, batches == b].mean() for b in range(n_batches)]
        means_after = [result[:, batches == b].mean() for b in range(n_batches)]
        spread_before = max(means_before) - min(means_before)
        spread_after = max(means_after) - min(means_after)
        assert spread_after < spread_before

    def test_dataframe_wrapper(self):
        """DataFrame wrapper preserves index and columns and matches low-level API.

        Failure condition: the wrapper drops or reorders index/columns
        or produces different values than the raw array path.

        Tolerances: rtol=1e-12 for identical float64 values through
        the same compute path.
        """
        data, batches = make_test_data(n_proteins=10, n_samples_per_batch=3, n_batches=2)
        df = pd.DataFrame(
            data,
            index=[f"p_{i}" for i in range(10)],
            columns=[f"s_{j}" for j in range(6)],
        )
        result = adjust_limma(df, batches)
        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == list(df.index)
        assert list(result.columns) == list(df.columns)
        np.testing.assert_allclose(result.values, remove_batch_effect(data, batches), rtol=1e-12)

    def test_two_batches(self):
        """After correction with 2 batches, batch means must converge.

        Failure condition: the OLS fit with 2 batches fails to bring
        means within a small tolerance.

        Tolerances: spread < 0.5 for this specific seed (7) where
        the batch shift is 3.0 units.
        """
        rng = np.random.default_rng(7)
        data = rng.normal(10, 2, size=(20, 8))
        data[:, 4:] += 3.0
        batches = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        result = remove_batch_effect(data, batches)
        assert result.shape == data.shape
        m0 = result[:, :4].mean()
        m1 = result[:, 4:].mean()
        assert abs(m0 - m1) < 0.5

    def test_three_batches(self):
        """After correction with 3 batches, batch means must converge.

        Failure condition: the OLS fit with 3 batches fails to bring
        means within tolerance.

        Tolerances: spread < 0.5 for seed (11) where batch shifts
        are 2.0 and -1.5 units.
        """
        rng = np.random.default_rng(11)
        data = rng.normal(10, 2, size=(30, 9))
        data[:, 3:6] += 2.0
        data[:, 6:] -= 1.5
        batches = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        result = remove_batch_effect(data, batches)
        means = [result[:, batches == b].mean() for b in [1, 2, 3]]
        assert max(means) - min(means) < 0.5


class TestLimmaEdgeCases:
    def test_nan_rows_stay_nan(self):
        """Per-cell NaN positions stay NaN in output; clean cells adjusted.

        Failure condition: NaN propagates to clean cells, shape changes,
        or clean cells are not adjusted.

        Matches R limma::removeBatchEffect behavior: NaN is handled
        per-feature, not by dropping entire rows.
        """
        rng = np.random.default_rng(42)
        n_clean, n_nan, n_samples = 8, 2, 6
        data = rng.normal(10, 2, size=(n_clean + n_nan, n_samples))
        data[-2:, 0] = np.nan
        batch = np.array([0, 0, 0, 1, 1, 1])

        result = remove_batch_effect(data, batch)

        assert result.shape == data.shape
        # Only the original NaN positions stay NaN
        assert np.isnan(result[-2, 0])
        assert not np.isnan(result[-2, 1:]).any()
        assert not np.isnan(result[:-2, :]).any()

    def test_single_batch_passthrough(self):
        """Single batch input must be returned unchanged.

        Failure condition: the function crashes or modifies data
        when only one batch is present.
        """
        rng = np.random.default_rng(3)
        data = rng.normal(10, 2, size=(10, 5))
        batches = np.zeros(5, dtype=int)
        result = remove_batch_effect(data, batches)
        np.testing.assert_array_equal(result, data)

    def test_noncontiguous_labels(self):
        """Non-contiguous batch labels must not crash.

        Failure condition: labels like [10, 10, 20] that are not
        0-indexed cause a failure in contrast encoding.
        """
        rng = np.random.default_rng(5)
        data = rng.normal(10, 2, size=(15, 6))
        data[:, 3:] += 2.0
        batches = np.array([10, 10, 10, 20, 20, 20])
        result = remove_batch_effect(data, batches)
        assert result.shape == data.shape
        assert not np.isnan(result).any()

    def test_grouped_nan_path_matches_reference(self):
        """Grouped NaN solves must match the original per-feature limma math.

        Failure condition: batching rows that share the same valid mask changes
        coefficients or corrected values relative to the per-feature reference.
        """
        batch = np.array([0, 0, 1, 1, 2, 2])
        data = np.array(
            [
                [10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
                [12.0, 13.0, 22.0, 23.0, 32.0, 33.0],
                [8.0, np.nan, 18.0, 19.0, 28.0, 29.0],
                [9.0, np.nan, 17.0, 18.0, 27.0, 28.0],
                [7.0, 8.0, np.nan, np.nan, 24.0, 25.0],
            ],
            dtype=np.float64,
        )

        expected = _remove_batch_effect_nan_reference(data, batch)
        result = remove_batch_effect(data, batch)

        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.skipif(not _has_r_fixtures, reason="R fixtures not generated")
class TestRLimmaConcordance:
    """Our remove_batch_effect() should match R limma::removeBatchEffect."""

    @pytest.fixture(autouse=True)
    def _load_input(self):
        df = pd.read_csv(FIXTURE_DIR / "small_input.tsv", sep="\t", index_col=0)
        self.data = df.values
        batch_csv = pd.read_csv(FIXTURE_DIR / "small_batch.csv")
        self.batches = batch_csv["batch"].to_numpy()

    def test_vs_r_limma(self):
        """Output must match R limma::removeBatchEffect.

        Failure condition: the OLS contrast encoding differs from R
        reference, producing different corrected values.

        Tolerances: rtol=1e-10/atol=1e-10 for closed-form OLS;
        no iteration, so agreement should be at machine epsilon.
        """
        expected = pd.read_csv(FIXTURE_DIR / "small_limma.tsv", sep="\t", index_col=0).values
        result = remove_batch_effect(self.data, self.batches)
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Mismatch vs R limma::removeBatchEffect",
        )


class TestLimmaInputs:
    """Batch labels and rows limma cannot fit."""

    @pytest.mark.parametrize("labels", [[0.2, 0.2, 0.9, 0.9], ["a", "a", "b", "b"]])
    def test_labels_that_are_not_whole_numbers_raise(self, labels: list[object]) -> None:
        """Fractional and text batch labels are rejected, as in ComBat.

        Failure condition: limma accepts labels ComBat rejects.
        """
        data = np.random.default_rng(11).normal(size=(5, 4))
        with pytest.raises(ValueError, match=r"batch labels must be whole numbers below 2\*\*53"):
            remove_batch_effect(data, labels)

    def test_feature_missing_a_batch_is_fitted_on_its_observed_batches(self) -> None:
        """A feature with no value in one batch is levelled across the batches it has.

        Failure condition: the unobserved batch enters the fit and pulls the
        feature's level towards zero.
        """
        rng = np.random.default_rng(12)
        batch = np.repeat([0, 1, 2], 4)
        shift = np.array([0.0, 1.0, -1.0])[batch]
        data = rng.normal(10, 0.3, (200, 12)) + rng.normal(0, 0.5, (200, 1)) + shift
        data[0, batch == 2] = np.nan
        keep = batch < 2
        out = remove_batch_effect(data, batch)
        alone = remove_batch_effect(data[:, keep], batch[keep])
        np.testing.assert_allclose(out[0, keep], alone[0], rtol=1e-12)

    def test_feature_missing_a_batch_without_residual_is_left_raw(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A feature missing a batch with one value per observed batch keeps its values.

        Fitting it would leave no residual and collapse its values to one level.

        Failure condition: the values 10 and 11 both come back as 10.5.
        """
        # The command line turns off propagation on the package logger, and caplog
        # only sees records that propagate to the root logger.
        monkeypatch.setattr(logging.getLogger("harmonizepy"), "propagate", True)
        rng = np.random.default_rng(33)
        data = rng.normal(5, 1, (4, 6))
        data[3] = [10.0, np.nan, 11.0, np.nan, np.nan, np.nan]
        with caplog.at_level(logging.INFO, logger="harmonizepy.limma_wrapper"):
            out = remove_batch_effect(data, np.repeat([0, 1, 2], 2))
        np.testing.assert_array_equal(out[3], data[3])
        assert "1 with no more values than observed batches" in caplog.text

    def test_feature_observed_in_one_batch_is_left_raw(self) -> None:
        """A feature with values in a single batch has no batch effect to remove.

        Failure condition: the feature is shifted although nothing can be estimated.
        """
        data = np.array([[1.0, 2.0, np.nan, np.nan], [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 1.0, 0.0]])
        out = remove_batch_effect(data, [0, 0, 1, 1])
        np.testing.assert_array_equal(out[0], data[0])


def test_contrast_design_sums_to_zero() -> None:
    """Each sum-to-zero batch column sums to zero over balanced batches, and the last batch is -1.

    Failure condition: the contrast coding differs from R's ``contr.sum``.
    """
    from harmonizepy.limma_wrapper import _contrast_design

    design, x_batch = _contrast_design(np.array([5, 5, 7, 7, 9, 9]))
    np.testing.assert_array_equal(design[:, 0], np.ones(6))
    np.testing.assert_array_equal(x_batch[:, 0], [1, 1, 0, 0, -1, -1])
    np.testing.assert_array_equal(x_batch[:, 1], [0, 0, 1, 1, -1, -1])


def test_log_counts_a_feature_with_no_value_apart_from_single_batch(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A feature with no value is not counted as having values in a single batch.

    Failure condition: the log reports two single-batch features where one has no
    value at all.
    """
    monkeypatch.setattr(logging.getLogger("harmonizepy"), "propagate", True)
    rng = np.random.default_rng(8)
    batch = np.repeat([0, 1, 2], 3)
    data = rng.normal(10, 1, (6, 9))
    data[0] = np.nan
    data[1, batch > 0] = np.nan
    with caplog.at_level(logging.INFO, logger="harmonizepy.limma_wrapper"):
        out = remove_batch_effect(data, batch)
    assert "1 feature(s) left uncorrected: 1 with values in a single batch" in caplog.text
    assert np.isnan(out[0]).all()


def test_feature_with_one_value_per_batch_is_left_raw() -> None:
    """A feature present in every batch with one value each keeps its values.

    Failure condition: the fit leaves no residual and every value comes back as
    the mean of the batch means.
    """
    rng = np.random.default_rng(12)
    batch = np.repeat([0, 1, 2], 2)
    data = rng.normal(5, 1, (4, 6))
    data[3] = [10.0, np.nan, 11.0, np.nan, 12.0, np.nan]
    out = remove_batch_effect(data, batch)
    np.testing.assert_array_equal(out[3], data[3])


def test_complete_matrix_with_one_sample_per_batch_is_left_raw() -> None:
    """Without missing cells, one sample per batch leaves no residual, so nothing is corrected.

    Failure condition: every feature collapses to a constant row.
    """
    data = np.random.default_rng(13).normal(5, 1, (5, 3))
    np.testing.assert_array_equal(remove_batch_effect(data, np.array([0, 1, 2])), data)


def test_report_names_limma_features_with_too_few_values() -> None:
    """harmonize() with limma reports a feature with one value per batch as raw.

    Failure condition: the report says ``corrected`` for a feature the fit collapsed.
    """
    from harmonizepy import harmonize

    rng = np.random.default_rng(14)
    samples = [f"s{i}" for i in range(6)]
    data = pd.DataFrame(
        rng.normal(5, 1, (6, 6)), index=[f"f{i}" for i in range(6)], columns=samples
    )
    data.iloc[:2, [1, 3, 5]] = np.nan
    desc = pd.DataFrame({"ID": samples, "sample": range(6), "batch": [1, 1, 2, 2, 3, 3]})
    out, report = harmonize(data, desc, algorithm="limma", needed_values=1, return_report=True)
    assert (report.loc[["f0", "f1"], "status"] == "raw_too_few_values").all()
    pd.testing.assert_frame_equal(out.loc[["f0", "f1"]], data.loc[["f0", "f1"]])
    assert (report.loc[["f2", "f3", "f4", "f5"], "status"] == "corrected").all()
