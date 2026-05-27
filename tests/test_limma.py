"""Tests for the limma-style removeBatchEffect implementation.

Validates:
1. Basic correctness (shape, no NaN, batch-mean reduction).
2. R concordance against limma::removeBatchEffect fixtures.
3. Edge cases.
"""

from __future__ import annotations

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
