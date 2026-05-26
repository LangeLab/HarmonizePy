"""Validation tests for the HarmonizePy ComBat implementation.

Tests verify correctness of all 4 modes by:
1. Checking output shape, no NaN, batch-mean reduction.
2. Cross-validating against R sva::ComBat fixtures.
3. Exercising edge cases (NaN rejection, single batch, ref_batch).

References
----------
Algorithm: Johnson WE, Li C, Rabinovic A. Biostatistics 8(1):118-127, 2007.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harmonizepy.combat import combat
from harmonizepy.combat_wrapper import adjust_combat

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_test_data(n_proteins=50, n_samples_per_batch=5, n_batches=3, seed=42):
    """Generate synthetic proteomics data with known batch effects."""
    rng = np.random.default_rng(seed)
    n_samples = n_samples_per_batch * n_batches

    truth = rng.normal(loc=10, scale=2, size=(n_proteins, n_samples))

    batch_labels = np.repeat(range(n_batches), n_samples_per_batch)
    batch_shifts = [0.0, 2.5, -1.8]
    batch_scales = [1.0, 1.5, 0.7]

    data = truth.copy()
    for b in range(n_batches):
        mask = batch_labels == b
        data[:, mask] = data[:, mask] * batch_scales[b] + batch_shifts[b]

    data += rng.normal(0, 0.3, size=data.shape)

    df = pd.DataFrame(
        data,
        index=[f"protein_{i}" for i in range(n_proteins)],
        columns=[f"sample_{j}" for j in range(n_samples)],
    )
    return df, batch_labels


# ---------------------------------------------------------------------------
# Core correctness tests - all 4 modes
# ---------------------------------------------------------------------------


class TestCombatModes:
    """Verify each mode produces valid, batch-reduced output."""

    @pytest.mark.parametrize(
        "mode,par_prior,mean_only",
        [
            (1, True, False),
            (2, True, True),
            (3, False, False),
            (4, False, True),
        ],
        ids=[
            "mode1-param-full",
            "mode2-param-meanonly",
            "mode3-nonparam-full",
            "mode4-nonparam-meanonly",
        ],
    )
    def test_mode(self, mode, par_prior, mean_only):
        df, batches = make_test_data()

        # Via low-level API
        result = combat(df.values, batches, par_prior=par_prior, mean_only=mean_only)
        assert result.shape == df.shape
        assert not np.isnan(result).any(), "NaN in output"

        # Batch mean spread should shrink
        n_batches = 3
        means_before = [df.values[:, batches == b].mean() for b in range(n_batches)]
        means_after = [result[:, batches == b].mean() for b in range(n_batches)]
        spread_before = max(means_before) - min(means_before)
        spread_after = max(means_after) - min(means_after)
        assert spread_after < spread_before, (
            f"Mode {mode}: spread not reduced {spread_before:.3f} → {spread_after:.3f}"
        )

        # Via wrapper API
        df_result = adjust_combat(df, batches, mode=mode)
        assert isinstance(df_result, pd.DataFrame)
        np.testing.assert_allclose(df_result.values, result, rtol=1e-12)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_nan_rejected(self):
        df, batches = make_test_data(n_proteins=10, n_samples_per_batch=3, n_batches=2)
        data = df.values.copy()
        data[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            combat(data, batches)

    def test_single_batch_passthrough(self):
        df, _ = make_test_data(n_proteins=10, n_samples_per_batch=6, n_batches=1)
        batches = np.zeros(6, dtype=int)
        result = combat(df.values, batches)
        np.testing.assert_array_equal(result, df.values)

    def test_too_few_features(self):
        data = np.array([[1.0, 2.0, 3.0]])  # 1 row
        batches = np.array([0, 0, 1])
        with pytest.raises(ValueError, match="at least 2 features"):
            combat(data, batches)

    def test_wrapper_invalid_mode(self):
        df, batches = make_test_data(n_proteins=5, n_samples_per_batch=3, n_batches=2)
        with pytest.raises(ValueError, match="mode"):
            adjust_combat(df, batches, mode=5)

    def test_ref_batch(self):
        df, batches = make_test_data()
        result = combat(df.values, batches, ref_batch=0)
        # Reference batch columns should be unchanged
        ref_cols = batches == 0
        np.testing.assert_array_equal(result[:, ref_cols], df.values[:, ref_cols])
        # Other batches should still be adjusted
        assert result.shape == df.shape

    def test_noncontiguous_batch_labels(self):
        """Batch labels [5, 5, 10, 10, 10] should work via remapping."""
        rng = np.random.default_rng(99)
        data = rng.normal(10, 2, size=(20, 5))
        data[:, 2:] += 3.0
        batches = np.array([5, 5, 10, 10, 10])
        result = combat(data, batches)
        assert result.shape == data.shape
        assert not np.isnan(result).any()

    def test_adjust_combat_ref_batch(self):
        """Wrapper-level ref_batch leaves reference batch unchanged."""
        from harmonizepy import adjust_combat

        df, batches = make_test_data()
        result = adjust_combat(df, batches, mode=1, ref_batch=0)
        ref_cols = batches == 0
        np.testing.assert_array_equal(
            result.values[:, ref_cols],
            df.values[:, ref_cols],
        )


# ---------------------------------------------------------------------------
# R sva::ComBat concordance (fixtures generated by generate_r_fixtures.R)
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures"
_has_r_fixtures = (FIXTURE_DIR / "small_combat_mode1.tsv").exists()


def _load_fixture(name: str) -> np.ndarray:
    """Load a TSV fixture (feature column as index) and return the numeric matrix."""
    df = pd.read_csv(FIXTURE_DIR / name, sep="\t", index_col=0)
    return df.values


@pytest.mark.skipif(not _has_r_fixtures, reason="R fixtures not generated")
class TestRConcordance:
    """Our combat() output should match R sva::ComBat within tolerance."""

    @pytest.fixture(autouse=True)
    def _load_input(self):
        """Load shared small test case input and batch labels."""
        df = pd.read_csv(FIXTURE_DIR / "small_input.tsv", sep="\t", index_col=0)
        self.data = df.to_numpy()
        batch_csv = pd.read_csv(FIXTURE_DIR / "small_batch.csv")
        self.batches = batch_csv["batch"].to_numpy()

    @pytest.mark.parametrize(
        "mode,par_prior,mean_only",
        [
            (1, True, False),
            (2, True, True),
            (3, False, False),
            (4, False, True),
        ],
        ids=["mode1", "mode2", "mode3", "mode4"],
    )
    def test_sva_combat(self, mode, par_prior, mean_only):
        """Compare against direct sva::ComBat output."""
        expected = _load_fixture(f"small_combat_mode{mode}.tsv")
        result = combat(self.data, self.batches, par_prior=par_prior, mean_only=mean_only)
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"Mismatch vs R sva::ComBat mode {mode}",
        )
