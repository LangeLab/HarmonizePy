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

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harmonizepy.combat import (
    _beta_na_grouped_batch_design,
    _int_eprior,
    _it_sol,
    _row_var_nan_grouped,
    combat,
    group_valid_rows,
)
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
    """Reference grouped Beta.NA using explicit grouped `lstsq` solves."""
    valid_masks = ~np.isnan(data)
    packed_masks = np.packbits(valid_masks, axis=1)
    grouped_rows: dict[bytes, list[int]] = {}
    for row_index in range(data.shape[0]):
        grouped_rows.setdefault(packed_masks[row_index].tobytes(), []).append(row_index)

    betas = np.empty((design.shape[0], data.shape[0]), dtype=np.float64)
    for row_indices in grouped_rows.values():
        row_indexer = np.asarray(row_indices, dtype=np.intp)
        valid = valid_masks[row_indexer[0]]
        if not valid.any():
            betas[:, row_indexer] = np.nan
            continue

        reduced_design = design[:, valid].T
        reduced_data = data[row_indexer][:, valid].T
        betas[:, row_indexer] = np.linalg.lstsq(reduced_design, reduced_data, rcond=None)[0]

    return betas


def _row_var_nan_grouped_reference(data: np.ndarray) -> np.ndarray:
    """Reference grouped row variance using explicit rowwise sample variance."""
    variances = np.empty(data.shape[0], dtype=np.float64)
    for row_index in range(data.shape[0]):
        valid = ~np.isnan(data[row_index])
        n_valid = int(valid.sum())
        if n_valid <= 1:
            variances[row_index] = 1.0
            continue
        variances[row_index] = np.var(data[row_index, valid], ddof=1)
    return variances


def _it_sol_reference(
    s_data: np.ndarray,
    g_hat: np.ndarray,
    d_hat: np.ndarray,
    g_bar: float,
    t2: float,
    a: float,
    b: float,
    conv: float = 1e-4,
    max_iter: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference `_it_sol` using the pre-optimization iteration structure."""
    nan_mask = np.isnan(s_data)
    all_nan = nan_mask.all(axis=1)
    active = ~all_nan

    if not np.any(active):
        return g_hat.copy(), d_hat.copy()

    n_per_gene = np.float64((~nan_mask).sum(axis=1))
    n_per_gene = np.maximum(n_per_gene, 1.0)
    sum_x = np.nansum(s_data, axis=1)
    sum_x2 = np.nansum(s_data * s_data, axis=1)

    t2_n = t2 * n_per_gene
    t2_n_g_hat = t2_n * g_hat

    g_old = g_hat[active].copy()
    d_old = d_hat[active].copy()
    t2_n_active = t2_n[active]
    t2_n_g_hat_active = t2_n_g_hat[active]
    sum_x_active = sum_x[active]
    sum_x2_active = sum_x2[active]
    n_per_gene_active = n_per_gene[active]

    for _ in range(max_iter):
        g_new = (t2_n_g_hat_active + d_old * g_bar) / (t2_n_active + d_old)
        sum2 = sum_x2_active - 2.0 * g_new * sum_x_active + n_per_gene_active * g_new * g_new
        d_new = (0.5 * sum2 + b) / (0.5 * n_per_gene_active + a - 1.0)

        delta_g = np.abs(g_new - g_old)
        delta_d = np.abs(d_new - d_old)
        denom_g = np.maximum(np.abs(g_old), 1e-12)
        denom_d = np.maximum(np.abs(d_old), 1e-12)
        change = max(np.max(delta_g / denom_g), np.max(delta_d / denom_d))

        g_old = g_new
        d_old = d_new

        if change < conv:
            break

    gamma_star = g_hat.copy()
    delta_star = d_hat.copy()
    gamma_star[active] = g_new
    delta_star[active] = d_new

    return gamma_star, delta_star


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

    def test_it_sol_matches_reference_on_finite_nan_mixture(self):
        """Optimized `_it_sol` must match the pre-optimization iteration math.

        Failure condition: buffer reuse or in-place updates change the fixed
        point reached by the parametric EB solver on mixed-observation data.
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

        expected_g, expected_d = _it_sol_reference(
            s_data,
            g_hat,
            d_hat,
            g_bar=0.3,
            t2=1.4,
            a=2.3,
            b=1.7,
        )
        result_g, result_d = _it_sol(
            s_data,
            g_hat,
            d_hat,
            g_bar=0.3,
            t2=1.4,
            a=2.3,
            b=1.7,
        )

        np.testing.assert_allclose(result_g, expected_g, rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(result_d, expected_d, rtol=1e-12, atol=1e-12, equal_nan=True)

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

    def test_row_var_nan_grouped_reuses_shared_valid_row_groups(self):
        """Grouped variance should accept shared mask metadata.

        Failure condition: `_combat_nan()` has to repack and regroup the same
        full-width NaN mask layout before switching from grouped OLS to grouped
        variance on arrays that preserve the same NaN positions.
        """
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

        valid_row_groups = group_valid_rows(data)
        expected_var = _row_var_nan_grouped_reference(data)
        result_var = _row_var_nan_grouped(data, valid_row_groups)

        np.testing.assert_allclose(result_var, expected_var, rtol=1e-12, atol=1e-12)

    def test_beta_na_grouped_batch_design_matches_lstsq_reference(self):
        """The one-hot batch-design fast path must match grouped lstsq output.

        Failure condition: replacing grouped `lstsq` dispatch in `_combat_nan()`
        changes coefficients for observed batches or the zero-fill behavior for
        batches with no observed values.
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

        valid_row_groups = group_valid_rows(data)

        expected = _beta_na_grouped_reference(data, design)
        result = _beta_na_grouped_batch_design(data, design, valid_row_groups)

        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12, equal_nan=True)

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

    def test_nan_ref_batch_preserves_reference_and_nan_mask(self):
        """NaN-safe ref_batch path must preserve the reference slice exactly.

        Failure condition: the NaN-aware ref_batch branch adjusts reference
        samples, changes NaN locations, or leaves the non-reference slice
        untouched despite a batch effect.
        """
        rng = np.random.default_rng(7)
        data = rng.normal(10, 1.5, size=(12, 6))
        data[:, 3:] += 4.0
        data[0, 0] = np.nan
        data[1, 4] = np.nan
        data[2, 1] = np.nan
        data[3, 5] = np.nan
        batch = np.array([0, 0, 0, 1, 1, 1])

        result = combat(data, batch, par_prior=True, mean_only=False, ref_batch=0)
        ref_cols = batch == 0

        np.testing.assert_array_equal(result[:, ref_cols], data[:, ref_cols])
        np.testing.assert_array_equal(np.isnan(result), np.isnan(data))

        non_ref_before = data[:, ~ref_cols]
        non_ref_after = result[:, ~ref_cols]
        observed = ~np.isnan(non_ref_before)
        assert not np.allclose(non_ref_after[observed], non_ref_before[observed])

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


class TestFeaturesLeftRaw:
    """Features the shared prior cannot use keep their raw values and change nothing else.

    A constant feature has zero variance inside each batch. If it entered the
    empirical-Bayes prior, it would switch shrinkage off for every feature (#18).
    """

    @staticmethod
    def _matrix(n_features: int = 500) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(1)
        batch = np.array([0] * 6 + [1] * 6)
        data = (
            rng.normal(20, 2, (n_features, 1))
            + rng.normal(0, 0.3, (n_features, 12))
            + rng.normal(1, 0.5, (n_features, 1)) * (batch == 1)
        )
        return data, batch

    # A constant, a constant that is not an exact binary fraction, and a constant
    # with one-ulp noise, whose variance is far below the floor but not zero.
    @pytest.mark.parametrize(
        "low, high", [(18.0, 19.0), (0.1, 0.7), (0.1, float(np.nextafter(0.1, 1.0)))]
    )
    @pytest.mark.parametrize("mean_only", [False, True])
    def test_constant_row_left_raw_and_others_unchanged(
        self, low: float, high: float, mean_only: bool
    ) -> None:
        """A constant row keeps its values, and every other row is as if it were absent.

        Failure condition: the constant row enters the prior and changes the other
        rows, or is itself corrected.
        """
        data, batch = self._matrix()
        constant = np.array([[low] * 6 + [high] * 6])
        alone = combat(data, batch, mean_only=mean_only)
        out = combat(np.vstack([data, constant]), batch, mean_only=mean_only)
        # Exact equality: the other rows go through the same computation as without
        # the constant row.
        np.testing.assert_array_equal(out[:500], alone)
        np.testing.assert_array_equal(out[500], constant[0])

    def test_constant_row_with_missing_cells(self) -> None:
        """The missing-value path leaves a constant row raw and the others unchanged.

        Failure condition: the constant row, observed in four samples per batch,
        enters the prior on the missing-value path.
        """
        data, batch = self._matrix()
        data[0, 0] = np.nan
        constant = np.array([[0.1] * 6 + [0.7] * 6])
        constant[0, [0, 6]] = np.nan
        alone = combat(data, batch)
        out = combat(np.vstack([data, constant]), batch)
        np.testing.assert_array_equal(out[:500], alone)
        np.testing.assert_array_equal(out[500], constant[0])

    def test_constant_reference_batch_left_raw(self) -> None:
        """A row constant in the reference batch alone is left raw in mean-only mode.

        Failure condition: the row's pooled variance, taken from the reference batch,
        is zero and the row enters the prior.
        """
        data, batch = self._matrix()
        row = np.array([[5.0] * 6 + [4.0, 6.0, 5.0, 4.5, 5.5, 5.2]])
        alone = combat(data, batch, mean_only=True, ref_batch=0)
        out = combat(np.vstack([data, row]), batch, mean_only=True, ref_batch=0)
        np.testing.assert_array_equal(out[:500], alone)
        np.testing.assert_array_equal(out[500], row[0])

    def test_single_sample_batch_constant_elsewhere_is_corrected(self) -> None:
        """With a one-sample batch, a batch with constant values does not block correction.

        A one-sample batch forces location-only correction, which never divides by a
        batch's variance, so only a zero pooled variance makes a row unestimable.

        Failure condition: the row is left raw although the engine corrects it.
        """
        rng = np.random.default_rng(2)
        batch = np.array([0, 0, 0, 1, 2, 2])
        data = rng.normal(5, 1, (20, 6))
        row = np.array([[5.0, 5.0, 5.0, 6.0, 1.0, 2.0]])
        out = combat(np.vstack([data, row]), batch)
        assert not np.array_equal(out[20], row[0])
        assert np.isfinite(out[20]).all()

    def test_sparse_row_with_spread_is_corrected(self) -> None:
        """A row with two differing values in one batch and none elsewhere is corrected.

        Failure condition: the row is left raw because it has few observations,
        although its variance is positive.
        """
        rng = np.random.default_rng(3)
        batch = np.repeat([0, 1, 2, 3], 2)
        data = rng.normal(5, 1, (30, 8))
        row = np.array([[1.0, 5.0] + [np.nan] * 6])
        out = combat(np.vstack([data, row]), batch)
        assert not np.array_equal(out[30, :2], row[0, :2])
        assert np.isfinite(out[30, :2]).all()

    def test_unknown_ref_batch_raises_when_every_row_is_constant(self) -> None:
        """An unknown reference batch is rejected before any row is left raw.

        Failure condition: the call returns the raw matrix without an error.
        """
        batch = np.array([0] * 6 + [1] * 6)
        data = np.repeat([[1.0] * 6 + [2.0] * 6], 3, axis=0)
        with pytest.raises(ValueError, match="ref_batch=99"):
            combat(data, batch, ref_batch=99)

    @pytest.mark.parametrize(
        "row",
        [
            pytest.param([3.0 + 1e-9 * k for k in range(12)], id="spread-1e-9"),
            pytest.param([1e-200 * (1.0 + 0.1 * (k % 10)) for k in range(12)], id="underflow"),
        ],
    )
    def test_row_below_the_variance_floor_left_raw(self, row: list[float]) -> None:
        """A row whose pooled variance is below the 1e-12 floor keeps its values.

        Failure condition: the engine floors the variance and returns values that do
        not come from the row, or the row changes the other rows.
        """
        data, batch = self._matrix()
        alone = combat(data, batch)
        out = combat(np.vstack([data, row]), batch)
        np.testing.assert_array_equal(out[:500], alone)
        np.testing.assert_array_equal(out[500], row)

    def test_single_observed_value_is_corrected(self) -> None:
        """A row with one observed value is corrected, since the engine gives it unit variance.

        Failure condition: the row is left raw although the engine estimates it.
        """
        data, batch = self._matrix(50)
        row = np.array([[7.0] + [np.nan] * 11])
        out = combat(np.vstack([data, row]), batch)
        assert np.isfinite(out[50, 0])
        assert out[50, 0] != 7.0

    def test_single_reference_value_mean_only_is_corrected(self) -> None:
        """In mean-only mode, one reference value and a constant batch are still corrected.

        Failure condition: the row is left raw although the engine estimates it.
        """
        data, batch = self._matrix(50)
        row = np.array([[7.0] + [np.nan] * 5 + [3.0] * 6])
        out = combat(np.vstack([data, row]), batch, mean_only=True, ref_batch=0)
        assert np.isfinite(out[50, 6:]).all()
        assert not np.array_equal(out[50, 6:], row[0, 6:])

    def test_flag_matches_the_engine_floor_on_sparse_rows(self) -> None:
        """On random sparse rows, a row is left raw exactly when its pooled variance is floored.

        The pooled variance is recomputed here from its definition: the sample variance
        of the residuals from the batch means over observed cells, 1.0 with at most one
        observed value.

        Failure condition: the flag and the engine disagree for some row.
        """
        from harmonizepy.combat import _VARIANCE_FLOOR, _unestimable_rows

        rng = np.random.default_rng(6)
        batch = np.repeat([0, 1, 2], 3)
        data = rng.choice([1.0, 2.0, np.nan], size=(400, 9), p=[0.4, 0.2, 0.4])
        flags = _unestimable_rows(data, batch, mean_only=True, ref_batch=None)
        for row, flagged in zip(data, flags, strict=True):
            means = {
                b: np.nanmean(row[batch == b])
                for b in range(3)
                if (~np.isnan(row[batch == b])).any()
            }
            resid = np.array(
                [v - means[b] for v, b in zip(row, batch, strict=True) if not np.isnan(v)]
            )
            var = 1.0 if resid.size <= 1 else float(resid @ resid) / (resid.size - 1)
            assert flagged == (var < _VARIANCE_FLOOR)

    def test_infinite_values_rejected(self) -> None:
        """An infinite value, as log2 of zero gives, is rejected with the feature named.

        Failure condition: the solver runs on the infinite row and every output value
        becomes missing.
        """
        data, batch = self._matrix(50)
        data[3, 0] = -np.inf
        with pytest.raises(
            ValueError, match=r"infinite values in 1 feature\(s\) \(first: \['3'\]\)"
        ):
            combat(data, batch)

    def test_single_sample_reference_batch_rejected(self) -> None:
        """A reference batch with one sample is rejected, since it has no variance to pool.

        Failure condition: every row's pooled variance is zero and the output depends
        on whether the data has a missing cell.
        """
        rng = np.random.default_rng(7)
        batch = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        with pytest.raises(ValueError, match="single sample"):
            combat(rng.normal(10, 1, (20, 12)), batch, mean_only=True, ref_batch=0)

    def test_one_row_input_returned_raw(self) -> None:
        """A single feature is returned raw, including a row with a missing cell.

        Failure condition: the solver runs on one row and returns missing values.
        """
        row = np.array([[1.0, np.nan, 3.0, 4.0, 5.0, 6.0]])
        out = combat(row, np.array([0, 0, 0, 1, 1, 1]))
        np.testing.assert_array_equal(out, row)


def test_solver_stops_on_a_non_finite_change(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The empirical-Bayes solver stops at the first iteration when its change is not finite.

    Failure condition: the loop runs to its iteration limit, which by default takes
    about a minute, on a missing scale estimate.
    """
    # The command line turns off propagation on the package logger, and caplog
    # only sees records that propagate to the root logger.
    monkeypatch.setattr(logging.getLogger("harmonizepy"), "propagate", True)
    s_data = np.array([[0.1, -0.2, 0.3], [0.2, 0.1, -0.1]])
    d_hat = np.array([np.nan, 1.0])
    with caplog.at_level(logging.WARNING, logger="harmonizepy.combat"):
        _, delta = _it_sol(s_data, np.zeros(2), d_hat, 0.0, 1.0, 2.0, 1.0)
    assert np.isnan(delta[0])
    assert "stopped at iteration 1" in caplog.text


class TestEngineInputs:
    """Inputs that the engines must reject or set aside before fitting."""

    def test_feature_without_values_leaves_others_unchanged(self) -> None:
        """A feature with no observed value stays missing and does not change the others.

        Failure condition: the missing feature enters the prior, and every output
        value becomes missing.
        """
        rng = np.random.default_rng(8)
        batch = np.repeat([0, 1], 6)
        data = rng.normal(10, 1, (20, 12)) + np.repeat([0.0, 1.0], 6)
        with_empty = np.vstack([data, np.full((1, 12), np.nan)])
        out = combat(with_empty, batch)
        np.testing.assert_array_equal(out[:20], combat(data, batch))
        assert np.isnan(out[20]).all()

    @pytest.mark.parametrize(
        "labels",
        [
            [0.2, 0.2, 0.9, 0.9],
            [0, 0, np.nan, 1],
            ["a", "a", "b", "b"],
            [1 + 2j, 1 + 2j, 2, 2],
            [1e16, 1e16, 2e16, 2e16],
        ],
    )
    def test_labels_that_are_not_whole_numbers_raise(self, labels: list[object]) -> None:
        """Fractional, missing, or text batch labels are rejected.

        Failure condition: 0.2 and 0.9 are cast to the same integer and treated as
        one batch, so nothing is corrected; a complex label loses its imaginary part;
        a float too large to be exact is rounded to a neighbouring batch.
        """
        data = np.random.default_rng(9).normal(size=(5, 4))
        with pytest.raises(ValueError, match=r"batch labels must be whole numbers below 2\*\*53"):
            combat(data, labels)

    def test_whole_float_labels_match_integer_labels(self) -> None:
        """Labels written as 0.0 and 1.0 give the same result as 0 and 1.

        Failure condition: whole numbers stored as floats are rejected or mapped
        differently.
        """
        data = np.random.default_rng(10).normal(size=(10, 6))
        np.testing.assert_array_equal(
            combat(data, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), combat(data, [0, 0, 0, 1, 1, 1])
        )

    def test_error_lists_each_offending_label_once(self) -> None:
        """The error names each offending label once, quoted.

        Failure condition: the message repeats one label per sample.
        """
        data = np.random.default_rng(14).normal(size=(5, 4))
        with pytest.raises(ValueError, match=r"found 0\.2, 0\.9\.$"):
            combat(data, [0.2, 0.2, 0.9, 0.9])

    @pytest.mark.parametrize("ref_batch", [None, 0])
    @pytest.mark.parametrize("n_live", [1, 20])
    def test_features_without_values_with_reference_and_few_rows(
        self, ref_batch: int | None, n_live: int
    ) -> None:
        """Features without values stay missing with a reference batch and with one live row.

        Failure condition: the set-aside breaks the reference batch or the single
        feature rule, or an unestimable row next to it.
        """
        rng = np.random.default_rng(15)
        batch = np.repeat([0, 1], 6)
        live = rng.normal(10, 1, (n_live, 12)) + np.repeat([0.0, 1.0], 6)
        constant = np.array([[5.0] * 6 + [6.0] * 6])
        empty = np.full((2, 12), np.nan)
        out = combat(np.vstack([live, constant, empty]), batch, ref_batch=ref_batch)
        np.testing.assert_array_equal(out[:n_live], combat(live, batch, ref_batch=ref_batch))
        np.testing.assert_array_equal(out[n_live], constant[0])
        assert np.isnan(out[n_live + 1 :]).all()

    def test_every_feature_without_values_returns_a_copy(self) -> None:
        """A matrix with no observed value is returned unchanged.

        Failure condition: the set-aside recurses on an empty matrix or raises.
        """
        empty = np.full((3, 4), np.nan)
        assert np.isnan(combat(empty, [0, 0, 1, 1])).all()

    def test_adjust_combat_checks_batch_length_first(self) -> None:
        """``adjust_combat`` rejects a batch of the wrong length even with one batch label.

        Failure condition: the single-batch shortcut returns a copy without the check.
        """
        frame = pd.DataFrame(np.ones((3, 4)), columns=list("abcd"))
        with pytest.raises(ValueError, match="batch length"):
            adjust_combat(frame, [0, 0])
