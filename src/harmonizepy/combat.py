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

    Parameters
    ----------
    s_data : (n_genes, n_batch_samples)
        Standardised data for a single batch.
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
    n_samples = np.float64(s_data.shape[1])
    t2_n = t2 * n_samples
    t2_n_g_hat = t2_n * g_hat

    # Pre-compute per-gene sums so the iterative loop can use the
    # binomial formula instead of broadcasting to a 2-D temp array.
    sum_x = s_data.sum(axis=1)  # (n_genes,)
    sum_x2 = (s_data * s_data).sum(axis=1)  # (n_genes,)

    g_old = g_hat.copy()
    d_old = d_hat.copy()

    for _ in range(max_iter):
        # Update additive effect
        g_new = _postmean(g_bar, d_old, t2_n, t2_n_g_hat)
        # Residual sum of squares: sum_k (x_i[k] - g_new[i])^2
        # = sum_x2_i - 2 * g_new[i] * sum_x_i + n * g_new[i]^2
        sum2 = sum_x2 - 2.0 * g_new * sum_x + n_samples * g_new * g_new
        # Update multiplicative effect
        d_new = _postvar(sum2, n_samples, a, b)

        # Convergence: max relative change
        delta_g = np.abs(g_new - g_old)
        delta_d = np.abs(d_new - d_old)
        # Guard against division by zero on the first iteration
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

    return g_new, d_new


# ---------------------------------------------------------------------------
# Non-parametric solver
# ---------------------------------------------------------------------------


def _int_eprior(
    s_data: _Array,
    g_hat: _Array,
    d_hat: _Array,
) -> tuple[_Array, _Array]:
    """Monte-Carlo integration for non-parametric EB estimation (one batch).

    Vectorised over genes; for each gene *i* the leave-one-out kernel
    density is evaluated across every other gene simultaneously.

    Parameters
    ----------
    s_data : (n_genes, n_batch_samples)
    g_hat, d_hat : (n_genes,)

    Returns
    -------
    gamma_star, delta_star : (n_genes,)
    """
    n_genes, n_samples = s_data.shape
    g_star = np.empty(n_genes)
    d_star = np.empty(n_genes)

    d = np.maximum(d_hat, 1e-12)

    # Pre-compute per-gene statistics so the inner loop can compute
    # sum-of-squared-residuals via the binomial formula:
    #   sum_k (x_i[k] - g_j)^2  =  sum_x2_i - 2*g_j*sum_x_i + n*g_j^2
    # This avoids creating the (n_genes-1, n_samples) temp array.
    sum_x = s_data.sum(axis=1)  # (n_genes,)
    sum_x2 = (s_data * s_data).sum(axis=1)  # (n_genes,)

    const_term = -0.5 * n_samples * np.log(2.0 * np.pi)
    half_n = -0.5 * n_samples
    log_d = np.log(d)  # pre-compute log so the inner loop avoids n_genes log() calls

    mask = np.ones(n_genes, dtype=bool)
    for i in range(n_genes):
        mask[i] = False
        g = g_hat[mask]  # (n_genes-1,)
        d_i = d[mask]  # (n_genes-1,)
        log_d_i = log_d[mask]  # (n_genes-1,)

        # Squared residuals without a 2-D temp array
        sum2 = sum_x2[i] - 2.0 * g * sum_x[i] + n_samples * g * g

        log_lh = const_term + half_n * log_d_i - sum2 / (2.0 * d_i)
        log_lh -= log_lh.max()
        lh = np.exp(log_lh)

        total = lh.sum()
        if total == 0.0:
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

    Per-cell NaN is handled by dropping affected feature rows before
    computation, matching R ``sva::ComBat``'s internal ``na.omit``.
    NaN rows stay NaN in the output.

    Parameters
    ----------
    data : ndarray, shape (n_features, n_samples)
        Expression / abundance matrix.  Per-cell NaN is allowed;
        rows with any NaN are omitted from the EB computation.
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
        Batch-corrected data.  Rows with any NaN in the input remain
        all-NaN in the output.  All other rows are adjusted.

    Raises
    ------
    ValueError
        On wrong dimensionality, fewer than 2 clean features, or
        batch length mismatch.

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

    # ---- Handle per-cell NaN (match R sva::ComBat na.omit) -----------------
    # R sva::ComBat drops feature rows that contain any NaN, adjusts only
    # the fully-observed rows, and the dropped rows are absent from output.
    # We keep the same shape but leave dropped rows as all-NaN.
    nan_rows = np.isnan(data).any(axis=1)
    if nan_rows.any():
        clean_data = data[~nan_rows]
        n_clean = clean_data.shape[0]
        logger.debug(
            "Removed %d feature row(s) with NaN before adjustment; "
            "%d clean feature(s) remain",
            int(nan_rows.sum()),
            n_clean,
        )
        result = np.full_like(data, np.nan)
        if n_clean >= 2:
            corrected_clean = _combat_dense(
                clean_data, batch_int,
                par_prior=par_prior, mean_only=mean_only, ref_batch=ref_batch,
            )
            result[~nan_rows] = corrected_clean
        elif n_clean == 1:
            # Single clean feature: can't estimate variance (t2 is NaN).
            # Return the raw values unchanged.
            logger.debug(
                "Single clean feature after NaN removal: passing through raw values"
            )
            result[~nan_rows] = clean_data
        return result

    # No NaN in data.  Single-feature input cannot estimate variance
    # across features (ddof=1 produces NaN for t2).  Return copy.
    if n_features < 2:
        logger.debug("Single feature input, returning copy")
        return data.copy()

    return _combat_dense(
        data, batch_int,
        par_prior=par_prior, mean_only=mean_only, ref_batch=ref_batch,
    )


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
    XXT = design @ design.T
    B_hat = np.linalg.solve(XXT, design @ data.T)

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
