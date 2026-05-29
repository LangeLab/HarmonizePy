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
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from .validation import validate_combat_input

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
_Array = npt.NDArray[np.floating[Any]]


@dataclass(frozen=True)
class _ValidRowGroups:
    """Grouped row indices for a shared valid-observation mask layout."""

    valid_masks: npt.NDArray[np.bool_]
    row_groups: tuple[npt.NDArray[np.intp], ...]


@dataclass(frozen=True)
class _GroupedBatchDesignLayouts:
    """Precomputed reduced one-hot layouts for grouped Beta.NA."""

    reduced_designs: tuple[_Array | None, ...]
    counts: tuple[_Array | None, ...]

# ---------------------------------------------------------------------------
# Hyper-prior helpers
# ---------------------------------------------------------------------------


def _aprior(gamma_hat: _Array) -> float:
    """Hyper-prior *a* for the inverse-gamma on delta (scale effect)."""
    m = float(gamma_hat.mean())
    s2 = float(gamma_hat.var(ddof=1))
    if s2 <= 0.0 or np.isnan(s2):
        return 1.0  # flat prior when variance cannot be estimated
    return float((2.0 * s2 + m * m) / s2)


def _bprior(gamma_hat: _Array) -> float:
    """Hyper-prior *b* for the inverse-gamma on delta (scale effect)."""
    m = float(gamma_hat.mean())
    s2 = float(gamma_hat.var(ddof=1))
    if s2 <= 0.0 or np.isnan(s2):
        return 1.0  # flat prior when variance cannot be estimated
    return float((m * s2 + m**3) / s2)


# ---------------------------------------------------------------------------
# Posterior estimates
# ---------------------------------------------------------------------------


def _postmean(g_bar: Any, d_star: Any, t2_n: Any, t2_n_g_hat: Any) -> _Array:
    """Posterior mean of the additive batch effect (gamma)."""
    return (t2_n_g_hat + d_star * g_bar) / (t2_n + d_star)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Parametric iterative solver
# ---------------------------------------------------------------------------


def _it_sol(
    s_data: _Array,
    g_hat: _Array,
    d_hat: _Array,
    g_bar: float,
    t2: float,
    a: float,
    b: float,
    conv: float = 1e-4,
    max_iter: int = 1_000_000,
) -> tuple[_Array, _Array]:
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
    s_data: _Array,
    g_hat: _Array,
    d_hat: _Array,
) -> tuple[_Array, _Array]:
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

    d = np.maximum(d_hat, 1e-12)
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

_design_cache: dict[tuple[int, ...], tuple[_Array, _Array, list[npt.NDArray[np.intp]], int]] = {}


def _make_design(batch: npt.NDArray[np.intp], n_batch: int) -> _Array:
    """One-hot batch design matrix, shape (n_batch, n_samples)."""
    n_samples = len(batch)
    design = np.zeros((n_batch, n_samples), dtype=np.float64)
    for i in range(n_batch):
        design[i, batch == i] = 1.0
    return design  # np.float64 array satisfies _Array (NDArray[floating[Any]])


def _get_cached_design(batch_int: npt.NDArray[np.intp], n_batch: int) -> tuple[_Array, _Array, list[npt.NDArray[np.intp]], int]:
    """Return cached design matrix and XXT, or compute and cache.

    When consecutive affiliation groups share the same batch layout,
    this avoids recomputing the one-hot design matrix and its XXT
    decomposition for every group.
    """
    key = tuple(batch_int.tolist())
    if key in _design_cache:
        design, xx_t, batches_ind, cached_n_batch = _design_cache[key]
        if cached_n_batch == n_batch:
            return design, xx_t, batches_ind, n_batch

    design = _make_design(batch_int, n_batch)
    xx_t = design @ design.T
    batches_ind = [np.where(batch_int == i)[0] for i in range(n_batch)]

    # Keep cache bounded to avoid unbounded memory growth
    if len(_design_cache) > 64:
        _design_cache.clear()
    _design_cache[key] = (design, xx_t, batches_ind, n_batch)

    return design, xx_t, batches_ind, n_batch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def combat(
    data: _Array,
    batch: npt.ArrayLike,
    *,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> _Array:
    """Apply ComBat batch-effect correction.

    Per-cell NaN is handled per-feature by omitting NaN observations from
    OLS, variance, and mean computations (matching R ``sva::ComBat``
    v3.60.0's ``Beta.NA`` approach).  NaN stays in the same positions in
    the output.

    Parameters
    ----------
    data : ndarray, shape (n_features, n_samples)
        Expression / abundance matrix.  Per-cell NaN is allowed.
    batch : ndarray, shape (n_samples,)
        Integer batch labels, 0-indexed and contiguous (``0 .. n_batch-1``).
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
        On wrong dimensionality or batch length mismatch.

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
    batch_int: npt.NDArray[np.intp] = np.asarray(batch, dtype=np.intp).ravel()

    # ---- Input validation --------------------------------------------------
    validate_combat_input(data, batch_int)

    n_features = data.shape[0]

    # ---- Dispatch ----------------------------------------------------------
    has_nan = np.isnan(data).any()
    if not has_nan:
        if n_features < 2:
            logger.debug("Single feature input, returning copy")
            return data.copy()
        return _combat_dense(
            data, batch_int,
            par_prior=par_prior, mean_only=mean_only, ref_batch=ref_batch,
        )

    # All-NaN data: return copy immediately (nothing to adjust)
    if np.isnan(data).all():
        logger.debug("All-NaN input, returning copy")
        return data.copy()

    # Per-cell NaN present: use per-feature NaN-safe path (matches R
    # sva::ComBat v3.60.0 Beta.NA approach).
    return _combat_nan(
        data, batch_int,
        par_prior=par_prior, mean_only=mean_only, ref_batch=ref_batch,
    )


# ---------------------------------------------------------------------------
# NaN-mask grouping helpers
# ---------------------------------------------------------------------------


def _group_valid_rows(data: _Array) -> _ValidRowGroups:
    """Group row indices by identical non-NaN masks."""
    valid_masks = ~np.isnan(data)
    packed_masks = np.packbits(valid_masks, axis=1)
    grouped_rows: dict[bytes, list[int]] = {}
    for row_index in range(data.shape[0]):
        grouped_rows.setdefault(packed_masks[row_index].tobytes(), []).append(row_index)
    return _ValidRowGroups(
        valid_masks=valid_masks,
        row_groups=tuple(
            np.asarray(row_indices, dtype=np.intp) for row_indices in grouped_rows.values()
        ),
    )


def _prepare_grouped_batch_design_layouts(
    design: _Array,
    valid_row_groups: _ValidRowGroups,
) -> _GroupedBatchDesignLayouts:
    """Precompute reduced one-hot designs and counts for each valid-mask group."""
    if valid_row_groups.valid_masks.shape[1] != design.shape[1]:
        raise ValueError("valid_row_groups width does not match design")

    reduced_designs: list[_Array | None] = []
    counts: list[_Array | None] = []
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
    data: _Array,
    design: _Array,
    valid_row_groups: _ValidRowGroups,
    grouped_design_layouts: _GroupedBatchDesignLayouts | None = None,
) -> _Array:
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
    data: _Array,
    valid_row_groups: _ValidRowGroups | None = None,
) -> _Array:
    """Row-wise sample variance grouped by identical valid-observation masks.

    For a fixed validity mask, rows share the same reduced submatrix.
    Compute `ddof=1` variances on that reduced view in one batched pass
    while preserving the existing 0/1-valid-entry handling.
    """
    n_features, _ = data.shape
    variances = np.empty(n_features, dtype=np.float64)

    groups = valid_row_groups or _group_valid_rows(data)
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


def _combat_nan(
    data: _Array,
    batch_int: npt.NDArray[np.intp],
    *,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> _Array:
    """ComBat correction with per-feature NaN handling.

    For each feature independently, NaN observations are omitted from OLS,
    variance, and mean computations.  NaN positions are preserved in the
    output.  This matches R sva::ComBat v3.60.0's ``Beta.NA`` approach.
    """
    n_features, n_samples = data.shape

    unique_batches = np.unique(batch_int)
    n_batch = len(unique_batches)
    if n_batch < 2:
        logger.debug("Single batch input, returning copy")
        return data.copy()

    logger.info(
        "ComBat (NaN-safe) %s, %s: %d features x %d samples across %d batches",
        "parametric" if par_prior else "non-parametric",
        "location+scale" if not mean_only else "location only",
        n_features, n_samples, n_batch,
    )

    # Remap batch labels to 0..n_batch-1
    label_map = {old: new for new, old in enumerate(unique_batches)}
    batch_int = np.array([label_map[b] for b in batch_int], dtype=np.intp)
    if ref_batch is not None:
        if ref_batch not in label_map:
            raise ValueError(
                f"ref_batch={ref_batch!r} not found in batch labels {sorted(label_map.keys())}"
            )
        ref_idx = label_map[ref_batch]
    else:
        ref_idx = None

    design, _, batches_ind, _ = _get_cached_design(batch_int, n_batch)
    batch_sizes = np.array([len(b) for b in batches_ind], dtype=np.float64)

    if not mean_only and np.any(batch_sizes < 2):
        logger.debug("Forcing mean_only=True: batch with < 2 samples detected")
        mean_only = True

    valid_row_groups = _group_valid_rows(data)
    grouped_design_layouts = _prepare_grouped_batch_design_layouts(design, valid_row_groups)

    # ---- Per-feature B.hat (Beta.NA) ---------------------------------------
    b_hat = _beta_na_grouped_batch_design(
        data,
        design,
        valid_row_groups,
        grouped_design_layouts,
    )

    # ---- Grand mean and pooled variance (per-feature, NaN-safe) ------------
    # NOTE: R sva::ComBat uses DIFFERENT formulas:
    #   No NaN: mean(residuals^2)
    #   Has NaN: rowVars(residuals, na.rm=TRUE)  (sample variance, ddof=1)
    if ref_idx is not None:
        grand_mean = b_hat[ref_idx]
        ref_cols = batches_ind[ref_idx]
        fitted_ref = (design[:, ref_cols].T @ b_hat).T
        var_n = _row_var_nan_grouped(data[:, ref_cols] - fitted_ref)
    else:
        grand_mean = (batch_sizes / n_samples) @ b_hat
        fitted = (design.T @ b_hat).T
        var_n = _row_var_nan_grouped(data - fitted, valid_row_groups)

    var_pooled = np.maximum(var_n, 1e-12)
    std_pooled = np.sqrt(var_pooled)
    stand_mean = grand_mean[:, np.newaxis]
    std_pooled_2d = std_pooled[:, np.newaxis]

    # ---- Standardise (per-feature, NaN-safe) -------------------------------
    s_data = (data - stand_mean) / std_pooled_2d

    # ---- Per-feature gamma.hat (Beta.NA) -----------------------------------
    gamma_hat = _beta_na_grouped_batch_design(
        s_data,
        design,
        valid_row_groups,
        grouped_design_layouts,
    )

    # ---- delta.hat (per-batch, per-feature, NaN-safe) ----------------------
    if mean_only:
        delta_hat = np.ones_like(gamma_hat)
    else:
        delta_hat = np.empty_like(gamma_hat)
        for b in range(n_batch):
            idx = batches_ind[b]
            delta_hat[b] = _row_var_nan_grouped(s_data[:, idx])

    # ---- EB prior (same as dense path) ------------------------------------
    gamma_bar = gamma_hat.mean(axis=1)
    t2 = gamma_hat.var(axis=1, ddof=1)

    if par_prior and not mean_only:
        a_prior = np.array([_aprior(delta_hat[i]) for i in range(n_batch)])
        b_prior = np.array([_bprior(delta_hat[i]) for i in range(n_batch)])
    else:
        a_prior = np.ones(n_batch)
        b_prior = np.ones(n_batch)

    # ---- Solve for batch effects (NaN-safe solvers) -----------------------
    gamma_star = np.empty_like(gamma_hat)
    delta_star = np.empty_like(gamma_hat)

    for i, idx in enumerate(batches_ind):
        batch_s_data = s_data[:, idx]
        if par_prior:
            if mean_only:
                # R's postmean gets n=1 for mean_only mode
                t2_n = t2[i] * 1.0
                t2_n_g_hat = t2_n * gamma_hat[i]
                gamma_star[i] = _postmean(gamma_bar[i], 1.0, t2_n, t2_n_g_hat)
                delta_star[i] = 1.0
            else:
                gamma_star[i], delta_star[i] = _it_sol(
                    batch_s_data, gamma_hat[i], delta_hat[i],
                    gamma_bar[i], t2[i], a_prior[i], b_prior[i],
                )
        else:
            d_hat_i = np.ones_like(delta_hat[i]) if mean_only else delta_hat[i]
            g_s, d_s = _int_eprior(batch_s_data, gamma_hat[i], d_hat_i)
            gamma_star[i] = g_s
            delta_star[i] = d_s if not mean_only else np.ones_like(g_s)

    if ref_idx is not None:
        gamma_star[ref_idx] = 0.0
        delta_star[ref_idx] = 1.0

    # ---- Adjust data (only non-NaN entries) -------------------------------
    corrected = s_data.copy()
    for i, idx in enumerate(batches_ind):
        sqrt_delta = np.sqrt(np.maximum(delta_star[i], 1e-12))[:, np.newaxis]
        corrected[:, idx] = (corrected[:, idx] - gamma_star[i][:, np.newaxis]) / sqrt_delta

    corrected = corrected * std_pooled_2d + stand_mean

    if ref_idx is not None:
        corrected[:, batches_ind[ref_idx]] = data[:, batches_ind[ref_idx]]

    return cast(_Array, corrected)


def _combat_dense(
    data: _Array,
    batch_int: npt.NDArray[np.intp],
    *,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> _Array:
    """Dense (NaN-free) ComBat correction.  See ``combat()`` for docs."""
    n_features, n_samples = data.shape

    unique_batches = np.unique(batch_int)
    n_batch = len(unique_batches)
    if n_batch < 2:
        logger.debug("Single batch input, returning copy")
        return data.copy()

    mod_label = "parametric" if par_prior else "non-parametric"
    scale_label = "location+scale" if not mean_only else "location only"
    logger.info(
        "ComBat %s, %s: %d features x %d samples across %d batches",
        mod_label,
        scale_label,
        n_features,
        n_samples,
        n_batch,
    )

    # Remap batch labels to 0..n_batch-1
    label_map = {old: new for new, old in enumerate(unique_batches)}
    batch_int = np.array([label_map[b] for b in batch_int], dtype=np.intp)

    if ref_batch is not None:
        if ref_batch not in label_map:
            raise ValueError(
                f"ref_batch={ref_batch!r} not found in batch labels {sorted(label_map.keys())}"
            )
        ref_idx = label_map[ref_batch]
        logger.debug("Reference batch: original %s -> remapped %d", ref_batch, ref_idx)
    else:
        ref_idx = None

    design, xx_t, batches_ind, _ = _get_cached_design(batch_int, n_batch)
    batch_sizes = np.array([len(b) for b in batches_ind], dtype=np.float64)

    if not mean_only and np.any(batch_sizes < 2):
        logger.debug("Forcing mean_only=True: batch with < 2 samples detected")
        mean_only = True

    # ---- Standardise -------------------------------------------------------
    B_hat = np.linalg.solve(xx_t, design @ data.T)  # noqa: N806

    if ref_idx is not None:
        grand_mean = B_hat[ref_idx]
    else:
        grand_mean = (batch_sizes / n_samples) @ B_hat

    if ref_idx is not None:
        ref_cols = batches_ind[ref_idx]
        fitted = design[:, ref_cols].T @ B_hat
        residuals = data[:, ref_cols].T - fitted
        var_pooled = (residuals**2).mean(axis=0)
    else:
        fitted = design.T @ B_hat
        residuals = data.T - fitted
        var_pooled = (residuals**2).mean(axis=0)

    var_pooled = np.maximum(var_pooled, 1e-12)
    std_pooled = np.sqrt(var_pooled)
    stand_mean = grand_mean[:, np.newaxis]
    s_data = (data - stand_mean) / std_pooled[:, np.newaxis]

    # ---- Estimate batch effects --------------------------------------------
    gamma_hat = np.linalg.solve(xx_t, design @ s_data.T)

    if mean_only:
        delta_hat = np.ones_like(gamma_hat)
    else:
        delta_hat = np.empty_like(gamma_hat)
        for i, idx in enumerate(batches_ind):
            delta_hat[i] = s_data[:, idx].var(axis=1, ddof=1)

    gamma_bar = gamma_hat.mean(axis=1)
    t2 = gamma_hat.var(axis=1, ddof=1)

    if par_prior and not mean_only:
        a_prior = np.array([_aprior(delta_hat[i]) for i in range(n_batch)])
        b_prior = np.array([_bprior(delta_hat[i]) for i in range(n_batch)])
    else:
        a_prior = np.ones(n_batch)
        b_prior = np.ones(n_batch)

    gamma_star = np.empty_like(gamma_hat)
    delta_star = np.empty_like(gamma_hat)

    for i, idx in enumerate(batches_ind):
        batch_s_data = s_data[:, idx]
        if par_prior:
            if mean_only:
                t2_n = t2[i] * 1.0
                t2_n_g_hat = t2_n * gamma_hat[i]
                gamma_star[i] = _postmean(gamma_bar[i], 1.0, t2_n, t2_n_g_hat)
                delta_star[i] = 1.0
            else:
                gamma_star[i], delta_star[i] = _it_sol(
                    batch_s_data, gamma_hat[i], delta_hat[i],
                    gamma_bar[i], t2[i], a_prior[i], b_prior[i],
                )
        else:
            d_hat_i = np.ones_like(delta_hat[i]) if mean_only else delta_hat[i]
            g_star_i, d_star_i = _int_eprior(batch_s_data, gamma_hat[i], d_hat_i)
            gamma_star[i] = g_star_i
            delta_star[i] = d_star_i if not mean_only else 1.0

    if ref_idx is not None:
        gamma_star[ref_idx] = 0.0
        delta_star[ref_idx] = 1.0

    # ---- Adjust data -------------------------------------------------------
    corrected = s_data.copy()
    for i, idx in enumerate(batches_ind):
        sqrt_delta = np.sqrt(np.maximum(delta_star[i], 1e-12))[:, np.newaxis]
        corrected[:, idx] = (corrected[:, idx] - gamma_star[i][:, np.newaxis]) / sqrt_delta

    corrected = corrected * std_pooled[:, np.newaxis] + stand_mean

    if ref_idx is not None:
        corrected[:, batches_ind[ref_idx]] = data[:, batches_ind[ref_idx]]

    return corrected  # type: ignore[no-any-return]
