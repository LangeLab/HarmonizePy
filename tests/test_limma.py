"""Tests for the limma-style removeBatchEffect implementation.

Validates:
1. Basic correctness (shape, no NaN, batch-mean reduction).
2. R concordance against limma::removeBatchEffect fixtures.
3. Edge cases.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from harmonizepy.limma_wrapper import remove_batch_effect, adjust_limma

FIXTURE_DIR = Path(__file__).parent / "fixtures"
_has_r_fixtures = (FIXTURE_DIR / "small_limma.tsv").exists()


def make_test_data(n_proteins=50, n_samples_per_batch=5, n_batches=3, seed=42):
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
        data, batches = make_test_data()
        result = remove_batch_effect(data, batches)
        assert result.shape == data.shape
        assert not np.isnan(result).any()

    def test_batch_mean_reduction(self):
        data, batches = make_test_data()
        result = remove_batch_effect(data, batches)
        n_batches = 3
        means_before = [data[:, batches == b].mean() for b in range(n_batches)]
        means_after = [result[:, batches == b].mean() for b in range(n_batches)]
        spread_before = max(means_before) - min(means_before)
        spread_after = max(means_after) - min(means_after)
        assert spread_after < spread_before

    def test_dataframe_wrapper(self):
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
        np.testing.assert_allclose(
            result.values, remove_batch_effect(data, batches), rtol=1e-12
        )

    def test_two_batches(self):
        rng = np.random.default_rng(7)
        data = rng.normal(10, 2, size=(20, 8))
        data[:, 4:] += 3.0
        batches = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        result = remove_batch_effect(data, batches)
        assert result.shape == data.shape
        # After correction, batch means should be close
        m0 = result[:, :4].mean()
        m1 = result[:, 4:].mean()
        assert abs(m0 - m1) < 0.5

    def test_three_batches(self):
        rng = np.random.default_rng(11)
        data = rng.normal(10, 2, size=(30, 9))
        data[:, 3:6] += 2.0
        data[:, 6:] -= 1.5
        batches = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        result = remove_batch_effect(data, batches)
        means = [result[:, batches == b].mean() for b in [1, 2, 3]]
        assert max(means) - min(means) < 0.5


class TestLimmaEdgeCases:
    def test_nan_rejected(self):
        data = np.array([[1.0, 2.0, np.nan, 4.0], [5.0, 6.0, 7.0, 8.0]])
        batches = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="NaN"):
            remove_batch_effect(data, batches)

    def test_single_batch_passthrough(self):
        rng = np.random.default_rng(3)
        data = rng.normal(10, 2, size=(10, 5))
        batches = np.zeros(5, dtype=int)
        result = remove_batch_effect(data, batches)
        np.testing.assert_array_equal(result, data)

    def test_noncontiguous_labels(self):
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
        self.batches = batch_csv["batch"].values

    def test_vs_r_limma(self):
        expected = pd.read_csv(
            FIXTURE_DIR / "small_limma.tsv", sep="\t", index_col=0
        ).values
        result = remove_batch_effect(self.data, self.batches)
        np.testing.assert_allclose(
            result, expected, rtol=1e-10, atol=1e-10,
            err_msg="Mismatch vs R limma::removeBatchEffect",
        )
