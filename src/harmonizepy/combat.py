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
from typing import Any

import numpy as np
import numpy.typing as npt

from .validation import validate_combat_input

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
_Array = npt.NDArray[np.floating[Any]]

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


def _postvar(sum2: _Array, n: Any, a: float, b: float) -> _Array:
    """Posterior mean of the multiplicative batch effect (delta)."""
    return (0.5 * sum2 + b) / (0.5 * n + a - 1.0)  # type: ignore[no-any-return]


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

    g_old = g_hat[active].copy()
    d_old = d_hat[active].copy()
    t2_n_active = t2_n[active]
    t2_n_g_hat_active = t2_n_g_hat[active]
    sum_x_active = sum_x[active]
    sum_x2_active = sum_x2[active]
    n_per_gene_active = n_per_gene[active]

    for _ in range(max_iter):
        g_new = _postmean(g_bar, d_old, t2_n_active, t2_n_g_hat_active)
        sum2 = sum_x2_active - 2.0 * g_new * sum_x_active + n_per_gene_active * g_new * g_new
        d_new = _postvar(sum2, n_per_gene_active, a, b)

        delta_g = np.abs(g_new - g_old)
        delta_d = np.abs(d_new - d_old)
        denom_g = np.maximum(np.abs(g_old), 1e-12)
        denom_d = np.maximum(np.abs(d_old), 1e-12)
        change = max(np.max(delta_g / denom_g), np.max(delta_d / denom_d))

        g_old = g_new
        d_old = d_new

        if change < conv:
            logger.debug("Converged in %d iterations (change=%.2e)", _ + 1, change)
            break
    else:
        logger.warning(
            "Batch did not converge after %d iterations (final change=%.2e)",
            max_iter,
            change,
        )

    gamma_star = g_hat.copy()
    delta_star = d_hat.copy()
    gamma_star[active] = g_new
    delta_star[active] = d_new

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
    g_star = np.empty(n_genes)
    d_star = np.empty(n_genes)

    d = np.maximum(d_hat, 1e-12)
    not_nan = ~np.isnan(s_data)
    n_per_gene = np.float64(not_nan.sum(axis=1))
    sum_x = np.nansum(s_data, axis=1)
    sum_x2 = np.nansum(s_data * s_data, axis=1)
    log_two_pi_d = np.log(2.0 * np.pi * d)

    mask = np.ones(n_genes, dtype=bool)
    for i in range(n_genes):
        mask[i] = False
        g = g_hat[mask]
        d_i = d[mask]
        n_i = n_per_gene[i]
        if n_i < 1:
            g_star[i] = g_hat[i]
            d_star[i] = d_hat[i]
            mask[i] = True
            continue

        # For gene i and candidate prior mean g_j:
        # sum((x_i - g_j)^2) = sum_x2_i - 2*g_j*sum_x_i + n_i*g_j^2
        sum2 = sum_x2[i] - 2.0 * g * sum_x[i] + n_i * (g * g)

        log_lh = -0.5 * n_i * log_two_pi_d[mask] - sum2 / (2.0 * d_i)
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


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------


def _make_design(batch: npt.NDArray[np.intp], n_batch: int) -> _Array:
    """One-hot batch design matrix, shape (n_batch, n_samples)."""
    n_samples = len(batch)
    design = np.zeros((n_batch, n_samples), dtype=np.float64)
    for i in range(n_batch):
        design[i, batch == i] = 1.0
    return design  # np.float64 array satisfies _Array (NDArray[floating[Any]])


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
# Nan-safe helpers: Beta.NA-style per-feature OLS
# ---------------------------------------------------------------------------


def _beta_na(y: _Array, design: _Array) -> _Array:
    """Per-feature OLS on non-NA observations (R sva Beta.NA equivalent).

    Parameters
    ----------
    y : (n_samples,) feature row, may contain NaN
    design : (n_batch, n_samples) design matrix

    Returns
    -------
    (n_batch,) coefficient vector
    """
    valid = ~np.isnan(y)
    if not valid.any():
        return np.full(design.shape[0], np.nan)
    des = design[:, valid].T
    y1 = y[valid]
    return np.linalg.lstsq(des, y1, rcond=None)[0]


def _group_valid_rows(data: _Array) -> tuple[npt.NDArray[np.bool_], dict[bytes, list[int]]]:
    """Group row indices by identical non-NaN masks."""
    valid_masks = ~np.isnan(data)
    packed_masks = np.packbits(valid_masks, axis=1)
    grouped_rows: dict[bytes, list[int]] = {}
    for row_index in range(data.shape[0]):
        grouped_rows.setdefault(packed_masks[row_index].tobytes(), []).append(row_index)
    return valid_masks, grouped_rows


def _beta_na_grouped(data: _Array, design: _Array) -> _Array:
    """Per-feature OLS grouped by identical valid-observation masks.

    For a fixed validity mask, Beta.NA solves the same reduced design
    matrix against different feature vectors. Solve that system once with
    batched right-hand sides to preserve the rowwise result while reducing
    repeated `lstsq` calls.

    Parameters
    ----------
    data : (n_features, n_samples) matrix, may contain NaN
    design : (n_batch, n_samples) design matrix

    Returns
    -------
    (n_batch, n_features) coefficient matrix
    """
    n_features, _ = data.shape
    betas = np.empty((design.shape[0], n_features), dtype=np.float64)

    valid_masks, grouped_rows = _group_valid_rows(data)

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


def _row_var_nan_grouped(data: _Array) -> _Array:
    """Row-wise sample variance grouped by identical valid-observation masks.

    For a fixed validity mask, rows share the same reduced submatrix.
    Compute `ddof=1` variances on that reduced view in one batched pass
    while preserving the rowwise `_row_var_nan` behavior for 0/1 valid
    observations.
    """
    n_features, _ = data.shape
    variances = np.empty(n_features, dtype=np.float64)

    valid_masks, grouped_rows = _group_valid_rows(data)
    for row_indices in grouped_rows.values():
        row_indexer = np.asarray(row_indices, dtype=np.intp)
        valid = valid_masks[row_indexer[0]]
        n_valid = int(valid.sum())

        if n_valid <= 1:
            variances[row_indexer] = 1.0
            continue

        reduced_data = data[row_indexer][:, valid]
        variances[row_indexer] = np.var(reduced_data, axis=1, ddof=1)

    return variances



def _row_var_nan(x: _Array) -> float:
    """Sample variance, excluding NaN (``var(x, na.rm=TRUE)``)."""
    valid = ~np.isnan(x)
    n = valid.sum()
    if n <= 1:
        return 1.0
    return float(np.var(x[valid], ddof=1))


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

    batches_ind = [np.where(batch_int == i)[0] for i in range(n_batch)]
    batch_sizes = np.array([len(b) for b in batches_ind], dtype=np.float64)

    if not mean_only and np.any(batch_sizes < 2):
        logger.debug("Forcing mean_only=True: batch with < 2 samples detected")
        mean_only = True

    design = _make_design(batch_int, n_batch)

    # ---- Per-feature B.hat (Beta.NA) ---------------------------------------
    B_hat = _beta_na_grouped(data, design)  # noqa: N806

    # ---- Grand mean and pooled variance (per-feature, NaN-safe) ------------
    # NOTE: R sva::ComBat uses DIFFERENT formulas:
    #   No NaN: mean(residuals^2)
    #   Has NaN: rowVars(residuals, na.rm=TRUE)  (sample variance, ddof=1)
    if ref_idx is not None:
        grand_mean = B_hat[ref_idx]
        ref_cols = batches_ind[ref_idx]
        fitted_ref = (design[:, ref_cols].T @ B_hat).T
        var_n = _row_var_nan_grouped(data[:, ref_cols] - fitted_ref)
    else:
        grand_mean = (batch_sizes / n_samples) @ B_hat
        fitted = (design.T @ B_hat).T
        var_n = _row_var_nan_grouped(data - fitted)

    var_pooled = np.maximum(var_n, 1e-12)
    std_pooled = np.sqrt(var_pooled)

    # ---- Standardise (per-feature, NaN-safe) -------------------------------
    s_data = np.empty_like(data)
    for i in range(n_features):
        valid = ~np.isnan(data[i, :])
        s_data[i, valid] = (data[i, valid] - grand_mean[i]) / std_pooled[i]
        s_data[i, ~valid] = np.nan

    # ---- Per-feature gamma.hat (Beta.NA) -----------------------------------
    gamma_hat = _beta_na_grouped(s_data, design)

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
    corrected = np.empty_like(s_data)
    for i in range(n_features):
        corrected[i, :] = s_data[i, :].copy()
        for b, idx in enumerate(batches_ind):
            valid = ~np.isnan(corrected[i, idx])
            if valid.any():
                sqrt_delta = np.sqrt(np.maximum(delta_star[b, i], 1e-12))
                corrected[i, idx[valid]] = (
                    corrected[i, idx[valid]] - gamma_star[b, i]
                ) / sqrt_delta
        # De-standardise (only non-NaN entries)
        valid = ~np.isnan(corrected[i, :])
        corrected[i, valid] = corrected[i, valid] * std_pooled[i] + grand_mean[i]

    if ref_idx is not None:
        corrected[:, batches_ind[ref_idx]] = data[:, batches_ind[ref_idx]]

    return corrected


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

    batches_ind = [np.where(batch_int == i)[0] for i in range(n_batch)]
    batch_sizes = np.array([len(b) for b in batches_ind], dtype=np.float64)

    if not mean_only and np.any(batch_sizes < 2):
        logger.debug("Forcing mean_only=True: batch with < 2 samples detected")
        mean_only = True

    design = _make_design(batch_int, n_batch)

    # ---- Standardise -------------------------------------------------------
    XXT = design @ design.T  # noqa: N806
    B_hat = np.linalg.solve(XXT, design @ data.T)  # noqa: N806

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
    gamma_hat = np.linalg.solve(
        design @ design.T, design @ s_data.T,
    )

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
