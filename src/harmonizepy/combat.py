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

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
_Array = npt.NDArray[np.floating]

# ---------------------------------------------------------------------------
# Hyper-prior helpers
# ---------------------------------------------------------------------------


def _aprior(gamma_hat: _Array) -> float:
    """Hyper-prior *a* for the inverse-gamma on delta (scale effect)."""
    m = gamma_hat.mean()
    s2 = gamma_hat.var(ddof=1)
    return (2.0 * s2 + m * m) / s2


def _bprior(gamma_hat: _Array) -> float:
    """Hyper-prior *b* for the inverse-gamma on delta (scale effect)."""
    m = gamma_hat.mean()
    s2 = gamma_hat.var(ddof=1)
    return (m * s2 + m ** 3) / s2


# ---------------------------------------------------------------------------
# Posterior estimates
# ---------------------------------------------------------------------------


def _postmean(g_bar: _Array, d_star: _Array, t2_n: _Array,
              t2_n_g_hat: _Array) -> _Array:
    """Posterior mean of the additive batch effect (gamma)."""
    return (t2_n_g_hat + d_star * g_bar) / (t2_n + d_star)


def _postvar(sum2: _Array, n: _Array, a: float, b: float) -> _Array:
    """Posterior mean of the multiplicative batch effect (delta)."""
    return (0.5 * sum2 + b) / (0.5 * n + a - 1.0)


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

    g_old = g_hat.copy()
    d_old = d_hat.copy()

    for _ in range(max_iter):
        # Update additive effect
        g_new = _postmean(g_bar, d_old, t2_n, t2_n_g_hat)
        # Residual sum of squares with new gamma
        sum2 = np.square(s_data - g_new[:, np.newaxis]).sum(axis=1)
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
            break

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

    for i in range(n_genes):
        # Leave-one-out: all other genes' estimates
        mask = np.ones(n_genes, dtype=bool)
        mask[i] = False
        g = g_hat[mask]                      # (n_genes-1,)
        d = d_hat[mask]                      # (n_genes-1,)
        x = s_data[i]                        # (n_samples,)

        # Squared residuals: (n_genes-1, n_samples)
        resid2 = np.square(x[np.newaxis, :] - g[:, np.newaxis])
        sum2 = resid2.sum(axis=1)            # (n_genes-1,)

        # Log-likelihood to avoid underflow
        log_lh = -0.5 * n_samples * np.log(2.0 * np.pi * d) - sum2 / (2.0 * d)
        # Shift for numerical stability
        log_lh -= log_lh.max()
        lh = np.exp(log_lh)

        total = lh.sum()
        if total == 0.0:
            # Fallback: uniform weights
            total = 1.0
            lh[:] = 1.0 / len(lh)

        g_star[i] = (g * lh).sum() / total
        d_star[i] = (d * lh).sum() / total

    return g_star, d_star


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------


def _make_design(batch: _Array, n_batch: int) -> _Array:
    """One-hot batch design matrix, shape (n_batch, n_samples)."""
    n_samples = len(batch)
    design = np.zeros((n_batch, n_samples), dtype=np.float64)
    for i in range(n_batch):
        design[i, batch == i] = 1.0
    return design


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def combat(
    data: _Array,
    batch: _Array,
    *,
    par_prior: bool = True,
    mean_only: bool = False,
    ref_batch: int | None = None,
) -> _Array:
    """Apply ComBat batch-effect correction.

    Parameters
    ----------
    data : ndarray, shape (n_features, n_samples)
        Expression / abundance matrix.  **Must not contain NaN.**
    batch : ndarray, shape (n_samples,)
        Integer batch labels, 0-indexed and contiguous (``0 .. n_batch-1``).
    par_prior : bool
        ``True`` → parametric EB (modes 1/2).
        ``False`` → non-parametric EB (modes 3/4).
    mean_only : bool
        ``True`` → correct location only, leave scale untouched (modes 2/4).
    ref_batch : int or None
        If given, this batch is treated as the reference and is not adjusted.

    Returns
    -------
    ndarray, shape (n_features, n_samples)
        Batch-corrected data (same dtype as input promoted to float64).

    Raises
    ------
    ValueError
        On NaN in *data*, fewer than 2 features, or fewer than 2 batches.
    """
    data = np.asarray(data, dtype=np.float64)

    # ---- Input validation --------------------------------------------------
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got {data.ndim}-D")
    n_features, n_samples = data.shape
    if n_features < 2:
        raise ValueError("ComBat requires >= 2 features (rows)")
    if np.isnan(data).any():
        raise ValueError("data must not contain NaN")

    batch = np.asarray(batch, dtype=np.intp).ravel()
    if batch.shape[0] != n_samples:
        raise ValueError(
            f"batch length ({batch.shape[0]}) != number of samples ({n_samples})"
        )

    unique_batches = np.unique(batch)
    n_batch = len(unique_batches)
    if n_batch < 2:
        return data.copy()

    # Remap batch labels to 0..n_batch-1 (in case the caller passes e.g. [1,1,2,2])
    label_map = {old: new for new, old in enumerate(unique_batches)}
    batch = np.array([label_map[b] for b in batch], dtype=np.intp)

    if ref_batch is not None:
        if ref_batch not in label_map:
            raise ValueError(
                f"ref_batch={ref_batch!r} not found in batch labels "
                f"{sorted(label_map.keys())}"
            )
        ref_idx = label_map[ref_batch]
    else:
        ref_idx = None

    # Batch membership indices
    batches_ind = [np.where(batch == i)[0] for i in range(n_batch)]
    batch_sizes = np.array([len(b) for b in batches_ind], dtype=np.float64)

    # ---- Design matrix (one-hot) -------------------------------------------
    design = _make_design(batch, n_batch)  # (n_batch, n_samples)

    # ---- Standardise -------------------------------------------------------
    # Regression: B_hat = (X X')^{-1} X Y'   where X=design, Y=data
    XXT = design @ design.T                       # (n_batch, n_batch) diagonal
    B_hat = np.linalg.solve(XXT, design @ data.T)  # (n_batch, n_features)

    if ref_idx is not None:
        grand_mean = B_hat[ref_idx]                # (n_features,)
    else:
        grand_mean = (batch_sizes / n_samples) @ B_hat  # (n_features,)

    # Pooled variance
    if ref_idx is not None:
        ref_cols = batches_ind[ref_idx]
        fitted = design[:, ref_cols].T @ B_hat      # (n_ref, n_features)
        residuals = data[:, ref_cols].T - fitted     # (n_ref, n_features)
        var_pooled = (residuals ** 2).mean(axis=0)   # (n_features,)
    else:
        fitted = design.T @ B_hat                    # (n_samples, n_features)
        residuals = data.T - fitted                  # (n_samples, n_features)
        var_pooled = (residuals ** 2).mean(axis=0)   # (n_features,)

    # Avoid division by zero for constant features
    var_pooled = np.maximum(var_pooled, 1e-12)
    std_pooled = np.sqrt(var_pooled)                 # (n_features,)

    # stand_mean: grand_mean broadcast to (n_features, n_samples)
    stand_mean = grand_mean[:, np.newaxis]           # will broadcast

    # Standardised data
    s_data = (data - stand_mean) / std_pooled[:, np.newaxis]

    # ---- Estimate batch effects --------------------------------------------
    # gamma_hat: additive effect per batch  (n_batch, n_features)
    batch_design = design                            # (n_batch, n_samples)
    gamma_hat = np.linalg.solve(
        batch_design @ batch_design.T,
        batch_design @ s_data.T,
    ).T  # → (n_features, n_batch) after transpose? No:
    # solve gives (n_batch, n_features), let's keep it that way
    gamma_hat = np.linalg.solve(
        batch_design @ batch_design.T,
        batch_design @ s_data.T,
    )  # (n_batch, n_features)

    # delta_hat: multiplicative effect per batch
    if mean_only:
        delta_hat = np.ones_like(gamma_hat)
    else:
        delta_hat = np.empty_like(gamma_hat)
        for i, idx in enumerate(batches_ind):
            delta_hat[i] = s_data[:, idx].var(axis=1, ddof=1)

    # Prior parameters
    gamma_bar = gamma_hat.mean(axis=1)               # (n_batch,)
    t2 = gamma_hat.var(axis=1, ddof=1)                # (n_batch,)

    if not mean_only:
        a_prior = np.array([_aprior(delta_hat[i]) for i in range(n_batch)])
        b_prior = np.array([_bprior(delta_hat[i]) for i in range(n_batch)])
    else:
        a_prior = np.ones(n_batch)
        b_prior = np.ones(n_batch)

    # ---- Solve for batch effects -------------------------------------------
    gamma_star = np.empty_like(gamma_hat)            # (n_batch, n_features)
    delta_star = np.empty_like(gamma_hat)            # (n_batch, n_features)

    for i, idx in enumerate(batches_ind):
        batch_s_data = s_data[:, idx]                # (n_features, n_batch_i)
        if par_prior:
            if mean_only:
                t2_n = t2[i] * 1.0
                t2_n_g_hat = t2_n * gamma_hat[i]
                gamma_star[i] = _postmean(
                    gamma_bar[i], 1.0, t2_n, t2_n_g_hat
                )
                delta_star[i] = 1.0
            else:
                gamma_star[i], delta_star[i] = _it_sol(
                    batch_s_data,
                    gamma_hat[i],
                    delta_hat[i],
                    gamma_bar[i],
                    t2[i],
                    a_prior[i],
                    b_prior[i],
                )
        else:
            if mean_only:
                d_hat_i = np.ones_like(delta_hat[i])
            else:
                d_hat_i = delta_hat[i]
            g_star_i, d_star_i = _int_eprior(
                batch_s_data, gamma_hat[i], d_hat_i
            )
            gamma_star[i] = g_star_i
            delta_star[i] = d_star_i if not mean_only else 1.0

    # Reference batch stays unadjusted
    if ref_idx is not None:
        gamma_star[ref_idx] = 0.0
        delta_star[ref_idx] = 1.0

    # ---- Adjust data -------------------------------------------------------
    corrected = s_data.copy()
    for i, idx in enumerate(batches_ind):
        corrected[:, idx] = (
            (corrected[:, idx] - gamma_star[i][:, np.newaxis])
            / np.sqrt(delta_star[i][:, np.newaxis])
        )

    # De-standardise
    corrected = corrected * std_pooled[:, np.newaxis] + stand_mean

    # Restore reference batch from original data
    if ref_idx is not None:
        corrected[:, batches_ind[ref_idx]] = data[:, batches_ind[ref_idx]]

    return corrected
