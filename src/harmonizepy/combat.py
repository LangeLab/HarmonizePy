"""Empirical Bayes batch-effect correction (ComBat).

Pure NumPy re-implementation of the Johnson, Li & Rabinovic (2007)
algorithm [1]_ with full vectorisation and no legacy ``np.matrix`` usage.

Four modes are supported, matching R ``sva::ComBat`` conventions:

====  =============  ===========
Mode  ``par_prior``  ``mean_only``
====  =============  ===========
1     True           False
2     True           True
3     False          False
4     False          True
====  =============  ===========

References
----------
.. [1] Johnson WE, Li C, Rabinovic A. "Adjusting batch effects in
    microarray expression data using empirical Bayes methods."
    *Biostatistics* 8(1):118-127, 2007.

Acknowledgements
----------------
Algorithm logic cross-referenced against:
- R ``sva::ComBat``  (Leek JT et al., Bioconductor)
- ``inmoose.pycombat`` (Behdenna A et al., GPL-3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .types import FloatArray
from .validation import batch_codes, validate_engine_input, validate_ref_batch

logger = logging.getLogger(__name__)

# Lower bound applied to the pooled variance and to the batch scale estimates,
# so that a zero variance never divides. A feature whose variance falls below it
# is not estimated from its data, so it is left raw instead (see combat()).
_VARIANCE_FLOOR = 1e-12

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidRowGroups:
    """Rows grouped by their pattern of observed cells.

    ``valid_masks`` marks the observed cells of every row; each entry of
    ``row_groups`` holds the indices of rows with the same pattern, in order of
    first appearance.
    """

    valid_masks: npt.NDArray[np.bool_]
    row_groups: tuple[npt.NDArray[np.intp], ...]


@dataclass(frozen=True)
class _GroupedBatchDesignLayouts:
    """Precomputed reduced one-hot layouts for grouped Beta.NA."""

    reduced_designs: tuple[FloatArray | None, ...]
    counts: tuple[FloatArray | None, ...]


# Keyword arguments of combat() for each ComBat mode of HarmonizR and the CLI.
COMBAT_MODES: dict[int, dict[str, bool]] = {
    1: {"par_prior": True, "mean_only": False},
    2: {"par_prior": True, "mean_only": True},
    3: {"par_prior": False, "mean_only": False},
    4: {"par_prior": False, "mean_only": True},
}


# ---------------------------------------------------------------------------
# Hyper-prior helpers
# ---------------------------------------------------------------------------


def _hyper_priors(delta_hat: FloatArray) -> tuple[float, float]:
    """Return the hyper-priors *a* and *b* of the inverse-gamma prior on one batch's scale.

    Method-of-moments estimates from the mean and variance of the batch's
    scale estimates; both are 1.0 (a flat prior) when that variance is zero or
    undefined.
    """
    m = float(delta_hat.mean())
    s2 = float(delta_hat.var(ddof=1))
    if s2 <= 0.0 or np.isnan(s2):
        return 1.0, 1.0
    return float((2.0 * s2 + m * m) / s2), float((m * s2 + m**3) / s2)


# ---------------------------------------------------------------------------
# Posterior estimates
# ---------------------------------------------------------------------------


def _postmean_unit_scale(g_bar: float, t2: float, g_hat: FloatArray) -> FloatArray:
    """Posterior mean of the location effect with unit scale and n = 1.

    This is R ``sva``'s ``postmean`` as location-only mode calls it.
    """
    posterior: FloatArray = (t2 * g_hat + g_bar) / (t2 + 1.0)
    return posterior


# ---------------------------------------------------------------------------
# Parametric iterative solver
# ---------------------------------------------------------------------------


def _it_sol(
    s_data: FloatArray,
    g_hat: FloatArray,
    d_hat: FloatArray,
    g_bar: float,
    t2: float,
    a: float,
    b: float,
    conv: float = 1e-4,
    max_iter: int = 1_000_000,
) -> tuple[FloatArray, FloatArray]:
    """Iterative EB solver for one batch (parametric path).

    NaN-safe: per-gene statistics are computed on non-NaN entries,
    matching R sva::ComBat's handling of missing data within batches.

    Parameters
    ----------
    s_data : (n_genes, n_batch_samples)
        Standardised data for a single batch. May contain NaN.
    g_hat, d_hat : (n_genes,)
        Initial estimates of additive / multiplicative effects.
    g_bar, t2 : float
        Prior mean and variance for gamma.
    a, b : float
        Hyper-priors for the inverse-gamma on delta.

    Returns
    -------
    gamma_star, delta_star : (n_genes,)
    """
    nan_mask = np.isnan(s_data)
    all_nan = nan_mask.all(axis=1)
    active = ~all_nan

    # Genes with no observed values cannot inform the EB update. Keep their
    # initial estimates and skip the iterative solver entirely.
    if not np.any(active):
        return g_hat.copy(), d_hat.copy()

    n_per_gene = np.float64((~nan_mask).sum(axis=1))
    n_per_gene = np.maximum(n_per_gene, 1.0)

    sum_x = np.nansum(s_data, axis=1)
    sum_x2 = np.nansum(s_data * s_data, axis=1)

    t2_n = t2 * n_per_gene
    t2_n_g_hat = t2_n * g_hat  # uses ORIGINAL g_hat, not updated g_new

    g_prev = g_hat[active].copy()
    d_prev = d_hat[active].copy()
    t2_n_active = t2_n[active]
    t2_n_g_hat_active = t2_n_g_hat[active]
    sum_x_active = sum_x[active]
    sum_x2_active = sum_x2[active]
    n_per_gene_active = n_per_gene[active]
    d_denom = 0.5 * n_per_gene_active + a - 1.0

    g_curr = np.empty_like(g_prev)
    d_curr = np.empty_like(d_prev)
    sum2 = np.empty_like(g_prev)
    delta_g = np.empty_like(g_prev)
    delta_d = np.empty_like(d_prev)
    denom_g = np.empty_like(g_prev)
    denom_d = np.empty_like(d_prev)
    latest_g = g_prev
    latest_d = d_prev

    for iteration in range(max_iter):
        np.multiply(d_prev, g_bar, out=g_curr)
        g_curr += t2_n_g_hat_active
        np.divide(g_curr, t2_n_active + d_prev, out=g_curr)

        np.multiply(g_curr, g_curr, out=sum2)
        sum2 *= n_per_gene_active
        sum2 -= 2.0 * g_curr * sum_x_active
        sum2 += sum_x2_active

        d_curr[:] = 0.5 * sum2 + b
        np.divide(d_curr, d_denom, out=d_curr)

        np.subtract(g_curr, g_prev, out=delta_g)
        np.abs(delta_g, out=delta_g)
        np.copyto(denom_g, g_prev)
        np.abs(denom_g, out=denom_g)
        np.maximum(denom_g, 1e-12, out=denom_g)
        np.divide(delta_g, denom_g, out=delta_g)

        np.subtract(d_curr, d_prev, out=delta_d)
        np.abs(delta_d, out=delta_d)
        np.copyto(denom_d, d_prev)
        np.abs(denom_d, out=denom_d)
        np.maximum(denom_d, 1e-12, out=denom_d)
        np.divide(delta_d, denom_d, out=delta_d)

        change = max(float(delta_g.max()), float(delta_d.max()))
        latest_g = g_curr
        latest_d = d_curr

        # A non-finite change never falls below the threshold, so without this
        # stop the loop would run to max_iter (about a minute) to the same result.
        if not np.isfinite(change):
            logger.warning(
                "Solver stopped at iteration %d: the change is not finite", iteration + 1
            )
            break

        if change < conv:
            logger.debug("Converged in %d iterations (change=%.2e)", iteration + 1, change)
            break

        g_prev, g_curr = g_curr, g_prev
        d_prev, d_curr = d_curr, d_prev
    else:
        logger.warning(
            "Batch did not converge after %d iterations (final change=%.2e)",
            max_iter,
            change,
        )

    gamma_star = g_hat.copy()
    delta_star = d_hat.copy()
    gamma_star[active] = latest_g
    delta_star[active] = latest_d

    return gamma_star, delta_star


# ---------------------------------------------------------------------------
# Non-parametric solver
# ---------------------------------------------------------------------------


def _int_eprior(
    s_data: FloatArray,
    g_hat: FloatArray,
    d_hat: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Monte-Carlo integration for non-parametric EB estimation (one batch).

    NaN-safe: each gene uses only its non-NaN entries for the likelihood,
    matching R sva::ComBat's per-gene handling where ``x <- sdat[i, !is.na(sdat[i, ])]``
    and ``n <- length(x)``.  Likelihood for
    gene *i* uses gene *i*'s non-NA count *n_i*, not the global batch
    size.

    Parameters
    ----------
    s_data : (n_genes, n_batch_samples)
    g_hat, d_hat : (n_genes,)

    Returns
    -------
    gamma_star, delta_star : (n_genes,)
    """
    n_genes, _ = s_data.shape
    g_star = g_hat.copy()
    d_star = d_hat.copy()

    if n_genes <= 1:
        return g_star, d_star

    d = np.maximum(d_hat, _VARIANCE_FLOOR)
    not_nan = ~np.isnan(s_data)
    n_per_gene = not_nan.sum(axis=1).astype(np.float64)
    sum_x = np.nansum(s_data, axis=1)
    sum_x2 = np.nansum(s_data * s_data, axis=1)
    g_hat_sq = g_hat * g_hat
    log_two_pi_d = np.log(2.0 * np.pi * d)
    neg_half_log_two_pi_d = -0.5 * log_two_pi_d

    block_size = 64
    for start in range(0, n_genes, block_size):
        stop = min(start + block_size, n_genes)
        block_indices = np.arange(start, stop, dtype=np.intp)
        n_block = n_per_gene[start:stop]
        valid_rows = n_block >= 1.0
        if not np.any(valid_rows):
            continue

        target_indices = block_indices[valid_rows]
        n_valid = n_block[valid_rows]
        sum_x_valid = sum_x[start:stop][valid_rows]
        sum_x2_valid = sum_x2[start:stop][valid_rows]

        log_lh = (
            sum_x2_valid[:, np.newaxis]
            - 2.0 * sum_x_valid[:, np.newaxis] * g_hat[np.newaxis, :]
            + n_valid[:, np.newaxis] * g_hat_sq[np.newaxis, :]
        )
        np.divide(log_lh, 2.0 * d[np.newaxis, :], out=log_lh)
        np.negative(log_lh, out=log_lh)
        log_lh += n_valid[:, np.newaxis] * neg_half_log_two_pi_d[np.newaxis, :]

        row_ids = np.arange(target_indices.size, dtype=np.intp)
        log_lh[row_ids, target_indices] = -np.inf

        row_max = np.max(log_lh, axis=1, keepdims=True)
        log_lh -= row_max
        np.exp(log_lh, out=log_lh)
        totals = log_lh.sum(axis=1)

        bad_totals = (totals == 0.0) | ~np.isfinite(totals)
        if np.any(bad_totals):
            log_lh[bad_totals] = 1.0
            log_lh[bad_totals, target_indices[bad_totals]] = 0.0
            totals = log_lh.sum(axis=1)

        g_star[target_indices] = (log_lh * g_hat[np.newaxis, :]).sum(axis=1) / totals
        d_star[target_indices] = (log_lh * d[np.newaxis, :]).sum(axis=1) / totals

    return g_star, d_star


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

_design_cache: dict[
    tuple[int, bytes], tuple[FloatArray, FloatArray, list[npt.NDArray[np.intp]]]
] = {}


def _make_design(batch: npt.NDArray[np.intp], n_batch: int) -> FloatArray:
    """One-hot batch design matrix, shape (n_batch, n_samples)."""
    n_samples = len(batch)
    design = np.zeros((n_batch, n_samples), dtype=np.float64)
    for i in range(n_batch):
        design[i, batch == i] = 1.0
    return design


def _get_cached_design(
    batch_int: npt.NDArray[np.intp], n_batch: int
) -> tuple[FloatArray, FloatArray, list[npt.NDArray[np.intp]]]:
    """Return the one-hot design, its Gram matrix, and the column indices of each batch.

    Consecutive groups often share a batch layout, so results are cached by the
    layout; the cache holds at most 64 layouts and is cleared when full.
    """
    key = (n_batch, batch_int.tobytes())
    cached = _design_cache.get(key)
    if cached is not None:
        return cached

    design = _make_design(batch_int, n_batch)
    layout = (design, design @ design.T, [np.where(batch_int == i)[0] for i in range(n_batch)])
    if len(_design_cache) > 64:
        _design_cache.clear()
    _design_cache[key] = layout
    return layout


def _unestimable_rows(
    data: FloatArray,
    batch_int: npt.NDArray[np.intp],
    *,
    mean_only: bool,
    ref_batch: int | None,
) -> npt.NDArray[np.bool_]:
    """Return the rows whose variance the engine would replace with the floor.

    The pooled variance is computed as the engine computes it: the mean square
    of the residuals from the batch means when ``data`` has no missing cell, and
    their sample variance otherwise, where a row with at most one observed value
    gets 1.0; with a reference batch, over that batch's columns only. A row
    whose pooled variance is below ``_VARIANCE_FLOOR`` is unestimable. Outside
    mean-only mode, so is a row with a batch of two or more observed values
    whose variance is below ``_VARIANCE_FLOOR`` times the pooled variance,
    which is the engine's batch scale estimate. Mean-only mode is in effect
    when the caller asks for it or when any batch has a single sample, as in
    the engines.
    """
    obs = ~np.isnan(data)
    has_nan = not bool(obs.all())
    batch_ids, batch_sizes = np.unique(batch_int, return_counts=True)
    mean_only = mean_only or bool(np.any(batch_sizes < 2))

    def pooled(ss: FloatArray, n_obs: npt.NDArray[np.intp], n_cols: int) -> FloatArray:
        if has_nan:
            return np.where(n_obs <= 1, 1.0, ss / np.maximum(n_obs - 1, 1))
        return ss / n_cols

    per_batch = {}
    for batch_id, n_cols in zip(batch_ids, batch_sizes, strict=True):
        cols = batch_int == batch_id
        observed = obs[:, cols]
        n_obs = observed.sum(axis=1)
        mean_b = np.nansum(data[:, cols], axis=1) / np.maximum(n_obs, 1)
        resid = np.where(observed, data[:, cols] - mean_b[:, np.newaxis], 0.0)
        per_batch[batch_id] = (n_obs, (resid * resid).sum(axis=1), int(n_cols))

    if ref_batch is None:
        stats = list(per_batch.values())
        ss_all = np.sum([ss for _, ss, _ in stats], axis=0)
        n_all = np.sum([n_obs for n_obs, _, _ in stats], axis=0)
        var_pooled = pooled(ss_all, n_all, data.shape[1])
    else:
        n_ref, ss_ref, n_ref_cols = per_batch[ref_batch]
        var_pooled = pooled(ss_ref, n_ref, n_ref_cols)

    unestimable = var_pooled < _VARIANCE_FLOOR
    if not mean_only:
        scale = np.maximum(var_pooled, _VARIANCE_FLOOR)
        for n_obs, ss, _ in per_batch.values():
            var_b = ss / np.maximum(n_obs - 1, 1)
            unestimable |= (n_obs >= 2) & (var_b < _VARIANCE_FLOOR * scale)
    return unestimable


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def combat(
    data: FloatArray,
    batch: npt.ArrayLike,
    *,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> FloatArray:
    """Apply ComBat batch-effect correction.

    Per-cell NaN is handled per-feature by omitting NaN observations from
    OLS, variance, and mean computations (matching R ``sva::ComBat``
    v3.60.0's ``Beta.NA`` approach).  NaN stays in the same positions in
    the output.

    Some features are returned with their raw values, and the run log counts
    them:

    - In the parametric modes, a feature whose variance the engine would raise
      to the 1e-12 floor: a pooled within-batch variance below 1e-12 (with
      ``ref_batch``, the reference batch's variance), or, outside mean-only
      mode, a batch whose variance is below 1e-12 of the pooled variance.
      Such a feature is not estimated from its data, and it would otherwise
      set the empirical-Bayes prior for every other feature; the others are
      corrected as if it were absent.
    - A single feature, which has no other features to share a prior with.

    A feature with no observed value is returned as missing and does not enter
    the fit of the others.

    Parameters
    ----------
    data : ndarray, shape (n_features, n_samples)
        Expression / abundance matrix.  Per-cell NaN is allowed.
    batch : array_like, shape (n_samples,)
        Batch label per sample: whole numbers, one batch per distinct value.
    par_prior : bool
        ``True`` for parametric EB (modes 1/2).
        ``False`` for non-parametric EB (modes 3/4).
    mean_only : bool
        ``True`` to correct location only, leave scale untouched (modes 2/4).
    ref_batch : int or None
        If given, this batch is treated as the reference and is not adjusted.

    Returns
    -------
    ndarray, shape (n_features, n_samples)
        Batch-corrected data.  NaN positions from input are preserved
        in the output.

    Raises
    ------
    ValueError
        On wrong dimensionality, batch length mismatch, batch labels that are
        not whole numbers, infinite values, or a ``ref_batch`` that is not
        among the batch labels or has one sample.

    Examples
    --------
    >>> import numpy as np
    >>> from harmonizepy import combat
    >>> data = np.random.default_rng(0).normal(10, 2, (50, 12))
    >>> batch = np.array([0]*4 + [1]*4 + [2]*4)
    >>> corrected = combat(data, batch)
    >>> corrected.shape
    (50, 12)
    """
    data = np.asarray(data, dtype=np.float64)
    batch_int = batch_codes(batch)

    validate_engine_input(data, batch_int)
    validate_ref_batch(batch_int, ref_batch)
    return _combat_validated(
        data, batch_int, par_prior=par_prior, mean_only=mean_only, ref_batch=ref_batch
    )


def _combat_validated(
    data: FloatArray,
    batch_int: npt.NDArray[np.intp],
    *,
    par_prior: bool,
    mean_only: bool,
    ref_batch: int | None,
) -> FloatArray:
    """Set aside the features ComBat cannot fit, then run the dense or missing-value path.

    Features with no observed value stay missing, and in the parametric modes
    features whose variance the engine would floor keep their raw values; the
    rest are corrected as if those features were absent. A single remaining
    feature, or a matrix with no observed value, is returned as given.
    """
    no_values = np.isnan(data).all(axis=1)
    if no_values.any() and not no_values.all():
        out = data.copy()
        out[~no_values] = _combat_validated(
            data[~no_values],
            batch_int,
            par_prior=par_prior,
            mean_only=mean_only,
            ref_batch=ref_batch,
        )
        return out

    if par_prior:
        unestimable = _unestimable_rows(data, batch_int, mean_only=mean_only, ref_batch=ref_batch)
        if unestimable.any():
            logger.info(
                "%d feature(s) with within-batch variance below the floor left uncorrected",
                int(unestimable.sum()),
            )
            out = data.copy()
            keep = ~unestimable
            if keep.sum() >= 2:
                out[keep] = _combat_validated(
                    data[keep],
                    batch_int,
                    par_prior=par_prior,
                    mean_only=mean_only,
                    ref_batch=ref_batch,
                )
            return out

    if data.shape[0] < 2:
        logger.debug("Single feature input, returning copy")
        return data.copy()

    missing = np.isnan(data)
    if not missing.any():
        return _combat_dense(
            data, batch_int, par_prior=par_prior, mean_only=mean_only, ref_batch=ref_batch
        )
    if missing.all():
        logger.debug("All-NaN input, returning copy")
        return data.copy()
    return _combat_nan(
        data, batch_int, par_prior=par_prior, mean_only=mean_only, ref_batch=ref_batch
    )


# ---------------------------------------------------------------------------
# NaN-mask grouping helpers
# ---------------------------------------------------------------------------


def group_valid_rows(data: FloatArray) -> ValidRowGroups:
    """Group row indices by their pattern of observed cells, in order of first appearance."""
    valid_masks = ~np.isnan(data)
    packed_masks = np.packbits(valid_masks, axis=1)
    grouped_rows: dict[bytes, list[int]] = {}
    for row_index in range(data.shape[0]):
        grouped_rows.setdefault(packed_masks[row_index].tobytes(), []).append(row_index)
    return ValidRowGroups(
        valid_masks=valid_masks,
        row_groups=tuple(
            np.asarray(row_indices, dtype=np.intp) for row_indices in grouped_rows.values()
        ),
    )


def _prepare_grouped_batch_design_layouts(
    design: FloatArray,
    valid_row_groups: ValidRowGroups,
) -> _GroupedBatchDesignLayouts:
    """Precompute reduced one-hot designs and counts for each valid-mask group."""
    if valid_row_groups.valid_masks.shape[1] != design.shape[1]:
        raise ValueError("valid_row_groups width does not match design")

    reduced_designs: list[FloatArray | None] = []
    counts: list[FloatArray | None] = []
    for row_indexer in valid_row_groups.row_groups:
        valid = valid_row_groups.valid_masks[row_indexer[0]]
        if not valid.any():
            reduced_designs.append(None)
            counts.append(None)
            continue

        reduced_design = design[:, valid]
        reduced_designs.append(reduced_design)
        counts.append(reduced_design.sum(axis=1))

    return _GroupedBatchDesignLayouts(
        reduced_designs=tuple(reduced_designs),
        counts=tuple(counts),
    )


def _beta_na_grouped_batch_design(
    data: FloatArray,
    design: FloatArray,
    valid_row_groups: ValidRowGroups,
    grouped_design_layouts: _GroupedBatchDesignLayouts | None = None,
) -> FloatArray:
    """Grouped Beta.NA specialized for the one-hot batch design.

    `_combat_nan()` always uses the cached one-hot batch design from
    `_make_design()`. For that design, the least-squares coefficient for each
    batch is just the mean over the observed samples in that batch, with the
    minimum-norm solution `0` for batches with no observed values.
    """
    n_features, _ = data.shape
    betas = np.empty((design.shape[0], n_features), dtype=np.float64)

    if valid_row_groups.valid_masks.shape != data.shape:
        raise ValueError("valid_row_groups shape does not match data")

    layouts = grouped_design_layouts or _prepare_grouped_batch_design_layouts(
        design,
        valid_row_groups,
    )

    for row_indexer, reduced_design, counts in zip(
        valid_row_groups.row_groups,
        layouts.reduced_designs,
        layouts.counts,
        strict=True,
    ):
        valid = valid_row_groups.valid_masks[row_indexer[0]]
        if reduced_design is None or counts is None:
            betas[:, row_indexer] = np.nan
            continue

        reduced_data = data[row_indexer][:, valid]
        batch_sums = reduced_data @ reduced_design.T
        batch_means = np.divide(
            batch_sums,
            counts[np.newaxis, :],
            out=np.zeros_like(batch_sums),
            where=counts[np.newaxis, :] > 0,
        )
        betas[:, row_indexer] = batch_means.T

    return betas


def _row_var_nan_grouped(
    data: FloatArray,
    valid_row_groups: ValidRowGroups | None = None,
) -> FloatArray:
    """Row-wise sample variance grouped by identical valid-observation masks.

    For a fixed validity mask, rows share the same reduced submatrix.
    Compute `ddof=1` variances on that reduced view in one batched pass
    while preserving the existing 0/1-valid-entry handling.
    """
    n_features, _ = data.shape
    variances = np.empty(n_features, dtype=np.float64)

    groups = valid_row_groups or group_valid_rows(data)
    if groups.valid_masks.shape != data.shape:
        raise ValueError("valid_row_groups shape does not match data")

    for row_indexer in groups.row_groups:
        valid = groups.valid_masks[row_indexer[0]]
        n_valid = int(valid.sum())

        if n_valid <= 1:
            variances[row_indexer] = 1.0
            continue

        group_data = data[row_indexer]
        sum_x = np.nansum(group_data, axis=1)
        sum_x2 = np.nansum(group_data * group_data, axis=1)
        variances[row_indexer] = (sum_x2 - (sum_x * sum_x) / n_valid) / (n_valid - 1)

    return variances


# ---------------------------------------------------------------------------
# NaN-aware ComBat path (Beta.NA-style per-feature computation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BatchLayout:
    """Batches of one ComBat fit, relabelled 0 to n_batch - 1."""

    batch_idx: npt.NDArray[np.intp]
    n_batch: int
    ref_idx: int | None
    design: FloatArray
    xx_t: FloatArray
    batches_ind: list[npt.NDArray[np.intp]]
    batch_sizes: FloatArray
    mean_only: bool


def _batch_layout(
    batch_int: npt.NDArray[np.intp], ref_batch: int | None, *, mean_only: bool
) -> _BatchLayout:
    """Relabel the batches, build the design, and settle location-only mode.

    Location-only mode is used when the caller asks for it or when any batch
    has a single sample, which has no variance of its own.
    """
    labels, batch_idx = np.unique(batch_int, return_inverse=True)
    batch_idx = batch_idx.astype(np.intp)
    n_batch = len(labels)
    ref_idx = None if ref_batch is None else int(np.flatnonzero(labels == ref_batch)[0])
    if ref_idx is not None:
        logger.debug("Reference batch: original %s -> remapped %d", ref_batch, ref_idx)
    design, xx_t, batches_ind = _get_cached_design(batch_idx, n_batch)
    batch_sizes = np.array([len(b) for b in batches_ind], dtype=np.float64)
    if not mean_only and np.any(batch_sizes < 2):
        logger.debug("Forcing mean_only=True: batch with < 2 samples detected")
        mean_only = True
    return _BatchLayout(
        batch_idx, n_batch, ref_idx, design, xx_t, batches_ind, batch_sizes, mean_only
    )


def _estimate_batch_effects(
    s_data: FloatArray,
    gamma_hat: FloatArray,
    delta_hat: FloatArray,
    layout: _BatchLayout,
    *,
    par_prior: bool,
) -> tuple[FloatArray, FloatArray]:
    """Shrink the per-batch location and scale estimates with the empirical-Bayes prior.

    Parametric modes use the normal and inverse-gamma priors (the iterative
    solver, or the closed-form posterior mean in location-only mode);
    non-parametric modes use the likelihood-weighted average over features.
    The reference batch, if any, gets location 0 and scale 1.
    """
    gamma_bar = gamma_hat.mean(axis=1)
    t2 = gamma_hat.var(axis=1, ddof=1)
    if par_prior and not layout.mean_only:
        priors = [_hyper_priors(delta_hat[i]) for i in range(layout.n_batch)]
    else:
        priors = [(1.0, 1.0)] * layout.n_batch

    gamma_star = np.empty_like(gamma_hat)
    delta_star = np.empty_like(gamma_hat)
    for i, idx in enumerate(layout.batches_ind):
        if par_prior and layout.mean_only:
            gamma_star[i] = _postmean_unit_scale(gamma_bar[i], t2[i], gamma_hat[i])
            delta_star[i] = 1.0
        elif par_prior:
            a_prior, b_prior = priors[i]
            gamma_star[i], delta_star[i] = _it_sol(
                s_data[:, idx], gamma_hat[i], delta_hat[i], gamma_bar[i], t2[i], a_prior, b_prior
            )
        else:
            gamma_star[i], d_star = _int_eprior(s_data[:, idx], gamma_hat[i], delta_hat[i])
            delta_star[i] = 1.0 if layout.mean_only else d_star

    if layout.ref_idx is not None:
        gamma_star[layout.ref_idx] = 0.0
        delta_star[layout.ref_idx] = 1.0
    return gamma_star, delta_star


def _adjust(
    s_data: FloatArray,
    data: FloatArray,
    gamma_star: FloatArray,
    delta_star: FloatArray,
    std_pooled: FloatArray,
    stand_mean: FloatArray,
    layout: _BatchLayout,
) -> FloatArray:
    """Remove the batch effects from the standardised data and return to the data scale.

    ``s_data`` is overwritten. Reference-batch columns are returned as given.
    """
    for i, idx in enumerate(layout.batches_ind):
        sqrt_delta = np.sqrt(np.maximum(delta_star[i], _VARIANCE_FLOOR))[:, np.newaxis]
        s_data[:, idx] = (s_data[:, idx] - gamma_star[i][:, np.newaxis]) / sqrt_delta
    s_data *= std_pooled[:, np.newaxis]
    s_data += stand_mean
    if layout.ref_idx is not None:
        ref_cols = layout.batches_ind[layout.ref_idx]
        s_data[:, ref_cols] = data[:, ref_cols]
    return s_data


def _log_fit(data: FloatArray, layout: _BatchLayout, *, par_prior: bool, label: str) -> None:
    logger.info(
        "%s %s, %s: %d features x %d samples across %d batches",
        label,
        "parametric" if par_prior else "non-parametric",
        "location+scale" if not layout.mean_only else "location only",
        data.shape[0],
        data.shape[1],
        layout.n_batch,
    )


def _combat_nan(
    data: FloatArray,
    batch_int: npt.NDArray[np.intp],
    *,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> FloatArray:
    """ComBat correction with per-feature handling of missing cells.

    Each feature's batch means and variances use its observed cells only
    (R ``sva::ComBat`` 3.60.0's ``Beta.NA``). The pooled variance is the sample
    variance of the residuals (``rowVars(..., na.rm = TRUE)`` in sva), not the
    mean square the dense path uses; sva switches formula the same way.
    Missing cells stay missing.
    """
    if len(np.unique(batch_int)) < 2:
        logger.debug("Single batch input, returning copy")
        return data.copy()
    layout = _batch_layout(batch_int, ref_batch, mean_only=mean_only)
    _log_fit(data, layout, par_prior=par_prior, label="ComBat (NaN-safe)")

    valid_row_groups = group_valid_rows(data)
    grouped_layouts = _prepare_grouped_batch_design_layouts(layout.design, valid_row_groups)
    b_hat = _beta_na_grouped_batch_design(data, layout.design, valid_row_groups, grouped_layouts)

    if layout.ref_idx is not None:
        grand_mean = b_hat[layout.ref_idx]
        ref_cols = layout.batches_ind[layout.ref_idx]
        fitted_ref = (layout.design[:, ref_cols].T @ b_hat).T
        var_n = _row_var_nan_grouped(data[:, ref_cols] - fitted_ref)
    else:
        grand_mean = (layout.batch_sizes / data.shape[1]) @ b_hat
        fitted = (layout.design.T @ b_hat).T
        var_n = _row_var_nan_grouped(data - fitted, valid_row_groups)

    std_pooled = np.sqrt(np.maximum(var_n, _VARIANCE_FLOOR))
    stand_mean = grand_mean[:, np.newaxis]
    s_data = (data - stand_mean) / std_pooled[:, np.newaxis]

    gamma_hat = _beta_na_grouped_batch_design(
        s_data, layout.design, valid_row_groups, grouped_layouts
    )
    delta_hat = np.ones_like(gamma_hat)
    if not layout.mean_only:
        for b, idx in enumerate(layout.batches_ind):
            delta_hat[b] = _row_var_nan_grouped(s_data[:, idx])

    gamma_star, delta_star = _estimate_batch_effects(
        s_data, gamma_hat, delta_hat, layout, par_prior=par_prior
    )
    return _adjust(s_data, data, gamma_star, delta_star, std_pooled, stand_mean, layout)


def _combat_dense(
    data: FloatArray,
    batch_int: npt.NDArray[np.intp],
    *,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> FloatArray:
    """ComBat correction of a matrix without missing cells.

    The pooled variance is the mean square of the residuals from the batch
    means, as in R ``sva::ComBat`` for complete data. See ``combat()``.
    """
    if len(np.unique(batch_int)) < 2:
        logger.debug("Single batch input, returning copy")
        return data.copy()
    layout = _batch_layout(batch_int, ref_batch, mean_only=mean_only)
    _log_fit(data, layout, par_prior=par_prior, label="ComBat")

    b_hat = np.linalg.solve(layout.xx_t, layout.design @ data.T)
    if layout.ref_idx is not None:
        grand_mean = b_hat[layout.ref_idx]
        ref_cols = layout.batches_ind[layout.ref_idx]
        residuals = data[:, ref_cols].T - layout.design[:, ref_cols].T @ b_hat
    else:
        grand_mean = (layout.batch_sizes / data.shape[1]) @ b_hat
        residuals = data.T - layout.design.T @ b_hat
    var_pooled = (residuals**2).mean(axis=0)

    std_pooled = np.sqrt(np.maximum(var_pooled, _VARIANCE_FLOOR))
    stand_mean = grand_mean[:, np.newaxis]
    s_data = (data - stand_mean) / std_pooled[:, np.newaxis]

    gamma_hat = np.linalg.solve(layout.xx_t, layout.design @ s_data.T)
    delta_hat = np.ones_like(gamma_hat)
    if not layout.mean_only:
        for i, idx in enumerate(layout.batches_ind):
            delta_hat[i] = s_data[:, idx].var(axis=1, ddof=1)

    gamma_star, delta_star = _estimate_batch_effects(
        s_data, gamma_hat, delta_hat, layout, par_prior=par_prior
    )
    return _adjust(s_data, data, gamma_star, delta_star, std_pooled, stand_mean, layout)
