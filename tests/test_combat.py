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

from harmonizepy.combat import _beta_na, _beta_na_grouped, _int_eprior, _it_sol, _row_var_nan, _row_var_nan_grouped, combat
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


def _int_eprior_reference(
    s_data: np.ndarray,
    g_hat: np.ndarray,
    d_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation matching the pre-optimization broadcast formula."""
    n_genes, _ = s_data.shape
    g_star = np.empty(n_genes)
    d_star = np.empty(n_genes)

    d = np.maximum(d_hat, 1e-12)
    not_nan = ~np.isnan(s_data)

    mask = np.ones(n_genes, dtype=bool)
    for i in range(n_genes):
        mask[i] = False
        g = g_hat[mask]
        d_i = d[mask]
        n_i = float(not_nan[i].sum())
        if n_i < 1:
            g_star[i] = g_hat[i]
            d_star[i] = d_hat[i]
            mask[i] = True
            continue

        x_i = s_data[i, not_nan[i]]
        dat = np.broadcast_to(x_i, (len(g), int(n_i)))
        resid2 = (dat - g[:, np.newaxis]) ** 2
        sum2 = resid2.sum(axis=1)

        log_lh = -0.5 * n_i * (np.log(2.0 * np.pi * d_i)) - sum2 / (2.0 * d_i)
        log_lh -= log_lh.max()
        lh = np.exp(log_lh)

        total = lh.sum()
        if total == 0.0 or not np.isfinite(total):
            total = 1.0
            lh = np.ones_like(lh) / len(lh)

        g_star[i] = (g * lh).sum() / total
        d_star[i] = (d_i * lh).sum() / total

        mask[i] = True

    return g_star, d_star


def _beta_na_grouped_reference(data: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Reference grouped Beta.NA using the existing per-row implementation."""
    return np.column_stack([_beta_na(data[i, :], design) for i in range(data.shape[0])])


def _row_var_nan_grouped_reference(data: np.ndarray) -> np.ndarray:
    """Reference grouped row variance using the existing per-row helper."""
    return np.array([_row_var_nan(data[i, :]) for i in range(data.shape[0])], dtype=np.float64)


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
        """Each ComBat mode must reduce batch mean spread and produce no NaN.

        Failure condition: a mode produces NaN, does not correct batch
        effects, or the wrapper API diverges from the low-level API.

        Tolerances: rtol=1e-12 for comparing wrapper vs low-level;
        both use the same combat() path so agreement is near epsilon.
        """
        df, batches = make_test_data()

        result = combat(df.values, batches, par_prior=par_prior, mean_only=mean_only)
        assert result.shape == df.shape
        assert not np.isnan(result).any(), "NaN in output"

        n_batches = 3
        means_before = [df.values[:, batches == b].mean() for b in range(n_batches)]
        means_after = [result[:, batches == b].mean() for b in range(n_batches)]
        spread_before = max(means_before) - min(means_before)
        spread_after = max(means_after) - min(means_after)
        assert spread_after < spread_before, (
            f"Mode {mode}: spread not reduced {spread_before:.3f} -> {spread_after:.3f}"
        )

        df_result = adjust_combat(df, batches, mode=mode)
        assert isinstance(df_result, pd.DataFrame)
        np.testing.assert_allclose(df_result.values, result, rtol=1e-12)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_it_sol_all_nan_short_circuits_without_warning(self, caplog: pytest.LogCaptureFixture):
        """All-NaN input to `_it_sol` must return immediately.

        Failure condition: the parametric solver enters its iteration
        loop for genes with no observed values and logs a non-
        convergence warning.
        """
        s_data = np.full((3, 4), np.nan, dtype=np.float64)
        g_hat = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        d_hat = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        with caplog.at_level("WARNING"):
            result_g, result_d = _it_sol(s_data, g_hat, d_hat, g_bar=0.0, t2=1.0, a=2.0, b=2.0)

        np.testing.assert_array_equal(result_g, g_hat)
        np.testing.assert_array_equal(result_d, d_hat)
        assert "did not converge" not in caplog.text

    def test_int_eprior_matches_reference_formula(self):
        """Optimized `_int_eprior` must match the broadcast reference math.

        Failure condition: the binomial-form optimization changes the
        posterior estimates relative to the original likelihood formula.

        Tolerances: rtol=1e-12/atol=1e-12 because both implementations
        evaluate the same formula in float64 with only algebraic
        rearrangement.
        """
        s_data = np.array(
            [
                [0.5, np.nan, 1.5, 0.0],
                [1.0, 1.5, np.nan, 0.5],
                [np.nan, np.nan, np.nan, np.nan],
                [0.2, -0.3, 0.1, np.nan],
            ],
            dtype=np.float64,
        )
        g_hat = np.array([0.4, -0.2, 0.1, 0.8], dtype=np.float64)
        d_hat = np.array([1.2, 0.9, 0.7, 1.5], dtype=np.float64)

        expected_g, expected_d = _int_eprior_reference(s_data, g_hat, d_hat)
        result_g, result_d = _int_eprior(s_data, g_hat, d_hat)

        np.testing.assert_allclose(result_g, expected_g, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result_d, expected_d, rtol=1e-12, atol=1e-12)

    def test_beta_na_grouped_matches_rowwise_reference(self):
        """Grouped Beta.NA must match the existing per-feature OLS exactly.

        Failure condition: batching rows with identical NaN masks changes
        coefficients, all-NaN handling, or row ordering.

        Tolerances: rtol=1e-12/atol=1e-12 because both paths call the same
        float64 least-squares solver on the same reduced designs.
        """
        design = np.array(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        data = np.array(
            [
                [10.0, 11.0, 20.0, 21.0, 22.0],
                [8.0, 9.0, np.nan, 19.0, 18.0],
                [7.0, 8.0, np.nan, 17.0, 16.0],
                [np.nan, np.nan, 5.0, 6.0, 7.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        )

        expected = _beta_na_grouped_reference(data, design)
        result = _beta_na_grouped(data, design)

        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_row_var_nan_grouped_matches_rowwise_reference(self):
        """Grouped row variance must match the current per-feature helper.

        Failure condition: batching identical NaN masks changes sample
        variance values or special-case handling for 0/1 valid entries.

        Tolerances: rtol=1e-12/atol=1e-12 because both paths compute the
        same float64 sample variance on the same reduced row data.
        """
        data = np.array(
            [
                [10.0, 11.0, 20.0, 21.0, 22.0],
                [8.0, 9.0, np.nan, 19.0, 18.0],
                [7.0, 8.0, np.nan, 17.0, 16.0],
                [np.nan, np.nan, 5.0, 6.0, 7.0],
                [np.nan, np.nan, 4.0, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        )

        expected = _row_var_nan_grouped_reference(data)
        result = _row_var_nan_grouped(data)

        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_nan_rows_stay_nan(self):
        """Per-cell NaN positions stay NaN in output; quantified cells adjusted.

        Failure condition: NaN propagates to clean cells, or clean cells
        are not adjusted, or the output shape changes.

        Matches R sva::ComBat v3.60.0's Beta.NA behavior: NaN is handled
        per-feature, not by dropping entire rows.
        """
        rng = np.random.default_rng(42)
        n_clean, n_nan, n_samples = 8, 2, 6
        data = rng.normal(10, 2, size=(n_clean + n_nan, n_samples))
        data[-2:, 0] = np.nan  # last 2 rows have NaN in column 0 only
        batch = np.array([0, 0, 0, 1, 1, 1])

        result = combat(data, batch, par_prior=True, mean_only=True)

        # Shape preserved
        assert result.shape == data.shape
        # NaN rows: only the original NaN position is NaN; rest are adjusted
        assert np.isnan(result[-2, 0])
        assert not np.isnan(result[-2, 1:]).any()
        assert np.isnan(result[-1, 0])
        assert not np.isnan(result[-1, 1:]).any()
        # Clean rows have no NaN
        assert not np.isnan(result[:-2, :]).any()
        # Clean rows were actually adjusted (different from input)
        assert not np.allclose(result[:-2, :], data[:-2, :])

    def test_all_nan_rows_returns_all_nan(self):
        """All rows have NaN -> output is all-NaN (nothing to adjust).

        Failure condition: a crash or partial output.
        """
        data = np.full((5, 6), np.nan)
        batch = np.array([0, 0, 0, 1, 1, 1])
        result = combat(data, batch)
        assert result.shape == data.shape
        assert np.isnan(result).all()

    def test_single_batch_passthrough(self):
        """Single batch input must be returned unchanged.

        Failure condition: the function crashes or modifies data
        when only one batch is present.
        """
        df, _ = make_test_data(n_proteins=10, n_samples_per_batch=6, n_batches=1)
        batches = np.zeros(6, dtype=int)
        result = combat(df.values, batches)
        np.testing.assert_array_equal(result, df.values)

    def test_too_few_features_passed_through(self):
        """Single feature with no NaN is passed through (combat can't adjust).

        Failure condition: the function crashes or produces NaN for the
        single feature.

        R sva::ComBat requires >= 2 features.  The pipeline handles this
        via single-feature pass-through in splitting.py; the low-level
        combat() returns raw data for < 2 clean features.
        """
        data = np.array([[1.0, 2.0, 3.0]])
        batches = np.array([0, 0, 1])
        result = combat(data, batches, par_prior=True, mean_only=True)
        np.testing.assert_array_equal(result, data)

    def test_wrapper_invalid_mode(self):
        """Invalid combat mode passed to wrapper must raise ValueError.

        Failure condition: an out-of-range mode is accepted.
        """
        df, batches = make_test_data(n_proteins=5, n_samples_per_batch=3, n_batches=2)
        with pytest.raises(ValueError, match="mode"):
            adjust_combat(df, batches, mode=5)

    def test_ref_batch(self):
        """Reference batch data must be left unchanged.

        Failure condition: the reference batch is adjusted or the
        output shape is wrong.
        """
        df, batches = make_test_data()
        result = combat(df.values, batches, ref_batch=0)
        ref_cols = batches == 0
        np.testing.assert_array_equal(result[:, ref_cols], df.values[:, ref_cols])
        assert result.shape == df.shape

    def test_noncontiguous_batch_labels(self):
        """Non-contiguous batch labels like [5, 5, 10, 10] must work via remapping.

        Failure condition: the label-remapping step crashes or
        produces incorrect results for non-0-indexed labels.
        """
        rng = np.random.default_rng(99)
        data = rng.normal(10, 2, size=(20, 5))
        data[:, 2:] += 3.0
        batches = np.array([5, 5, 10, 10, 10])
        result = combat(data, batches)
        assert result.shape == data.shape
        assert not np.isnan(result).any()

    def test_adjust_combat_ref_batch(self):
        """Wrapper-level ref_batch must leave the reference batch unchanged.

        Failure condition: the reference batch is adjusted.
        """
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
        """Output must match R sva::ComBat for all 4 modes.

        Failure condition: the Python implementation diverges from R
        reference beyond documented tolerances.

        Tolerances: rtol=1e-4/atol=1e-6 for parametric iterative
        modes (1, 3); closed-form modes are tighter but this
        tolerance covers all modes.
        """
        expected = _load_fixture(f"small_combat_mode{mode}.tsv")
        result = combat(self.data, self.batches, par_prior=par_prior, mean_only=mean_only)
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"Mismatch vs R sva::ComBat mode {mode}",
        )
