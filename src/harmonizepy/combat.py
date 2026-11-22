"""Empirical-Bayes batch-effect correction (ComBat) in NumPy.

The method of Johnson, Li and Rabinovic (2007) [1]_, following R
``sva::ComBat`` 3.60.0 [2]_, including its handling of missing cells, in which
each feature is estimated on its observed cells. Four modes, as in HarmonizR:

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
.. [1] Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray
    expression data using empirical Bayes methods. Biostatistics
    8(1):118-127, 2007. doi:10.1093/biostatistics/kxj037
.. [2] Leek JT, Johnson WE, Parker HS, et al. sva: Surrogate Variable Analysis.
    R package, Bioconductor. The ``inmoose`` Python port (Behdenna A et al.)
    was also consulted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .types import FloatArray
from .validation import (
    batch_codes,
    reject_combat_overflow,
    validate_engine_input,
    validate_ref_batch,
)

logger = logging.getLogger(__name__)

# Lower bound on the pooled variance and the batch scale estimates, so that a
# zero variance never divides; features below it are left raw by combat().
_VARIANCE_FLOOR = 1e-12

# Size of a block of the non-parametric prior, which holds one row per target
# feature and one column per feature. A block of 2**16 cells (512 KiB) stays in
# the CPU cache; at 32000 features, 16 rows took about half the time of 64. The
# block never has fewer than 16 rows, so a small group is done in one block.
# Rows are computed independently; the block size changes the output only by
# rounding in the matrix product.
_EPRIOR_BLOCK_CELLS = 2**16
_EPRIOR_MIN_BLOCK_ROWS = 16

# ---------------------------------------------------------------------------
# Grouping records
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
    """Per group of rows, the one-hot design over its observed samples and their counts."""

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
    if delta_hat.size < 2:
        return 1.0, 1.0
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
    """Solve the parametric empirical-Bayes estimates for one batch.

    A one-batch call of ``_it_sol_batches``; see there for the method.

    Parameters
    ----------
    s_data : ndarray, shape (n_features, n_batch_samples)
        Standardised data of the batch. May contain NaN.
    g_hat, d_hat : ndarray, shape (n_features,)
        Location and scale estimates of the batch.
    g_bar, t2 : float
        Mean and variance of the normal prior on the location.
    a, b : float
        Hyper-priors of the inverse-gamma prior on the scale.
    conv : float
        Relative change below which the iteration stops (``sva``'s 1e-4).
    max_iter : int
        Iteration limit; ``sva`` has none.

    Returns
    -------
    gamma_star, delta_star : ndarray, shape (n_features,)
    """
    gamma_star, delta_star = _it_sol_batches(
        s_data,
        [np.arange(s_data.shape[1], dtype=np.intp)],
        g_hat[np.newaxis, :],
        d_hat[np.newaxis, :],
        np.array([g_bar]),
        np.array([t2]),
        np.array([a]),
        np.array([b]),
        conv=conv,
        max_iter=max_iter,
    )
    return gamma_star[0], delta_star[0]


def _it_sol_batches(
    s_data: FloatArray,
    batches_ind: list[npt.NDArray[np.intp]],
    g_hat: FloatArray,
    d_hat: FloatArray,
    g_bar: FloatArray,
    t2: FloatArray,
    a: FloatArray,
    b: FloatArray,
    conv: float = 1e-4,
    max_iter: int = 1_000_000,
) -> tuple[FloatArray, FloatArray]:
    """Solve the parametric empirical-Bayes estimates of every batch in one loop.

    For each batch this is R ``sva``'s ``it.sol``: alternate the posterior
    location and scale of every feature until the largest relative change of
    either falls below ``conv``. Each feature uses its observed cells only, and
    a feature with no observed cell in a batch keeps its initial estimates.

    All batches iterate together, and each stops at its own convergence, so
    the result equals solving the batches one at a time. The working arrays
    shrink as batches converge, so a converged batch costs nothing further.
    Rows of a batch are not frozen individually: as in ``sva``, every row
    updates until the batch as a whole converges. Log records are written per
    batch, in batch order, after the loop.

    Parameters
    ----------
    s_data : ndarray, shape (n_features, n_samples)
        Standardised data. May contain NaN.
    batches_ind : list of ndarray
        Column indices of each batch.
    g_hat, d_hat : ndarray, shape (n_batches, n_features)
        Location and scale estimates.
    g_bar, t2, a, b : ndarray, shape (n_batches,)
        Prior mean and variance of the location, and hyper-priors of the scale.
    conv : float
        Relative change below which a batch stops (``sva``'s 1e-4).
    max_iter : int
        Iteration limit; a batch that reaches it is logged as not converged.

    Returns
    -------
    gamma_star, delta_star : ndarray, shape (n_batches, n_features)
    """
    n_obs = np.empty_like(g_hat)
    sum_x = np.empty_like(g_hat)
    sum_x2 = np.empty_like(g_hat)
    for i, idx in enumerate(batches_ind):
        batch_data = s_data[:, idx]
        n_obs[i] = (~np.isnan(batch_data)).sum(axis=1)
        sum_x[i] = np.nansum(batch_data, axis=1)
        sum_x2[i] = np.nansum(batch_data * batch_data, axis=1)
    active = n_obs > 0
    n_obs = np.maximum(n_obs, 1.0)

    t2_n = t2[:, np.newaxis] * n_obs
    t2_n_g_hat = t2_n * g_hat
    d_denom = 0.5 * n_obs + a[:, np.newaxis] - 1.0

    g_star = g_hat.copy()
    d_star = d_hat.copy()
    stop_iteration = np.zeros(len(batches_ind), dtype=np.intp)
    stop_change = np.full(len(batches_ind), np.nan)
    rows = np.flatnonzero(active.any(axis=1))
    g_curr, d_curr = g_hat[rows], d_hat[rows]
    run_active = active[rows]
    every_row_active = bool(run_active.all())
    run_t2_n, run_t2_n_g_hat, run_n_obs = t2_n[rows], t2_n_g_hat[rows], n_obs[rows]
    run_sum_x, run_sum_x2, run_d_denom = sum_x[rows], sum_x2[rows], d_denom[rows]
    run_g_bar, run_b = g_bar[rows, np.newaxis], b[rows, np.newaxis]

    for iteration in range(max_iter):
        if rows.size == 0:
            break
        g_prev, d_prev = g_curr, d_curr

        g_curr = d_prev * run_g_bar
        g_curr += run_t2_n_g_hat
        np.divide(g_curr, run_t2_n + d_prev, out=g_curr)

        sum2 = g_curr * g_curr
        sum2 *= run_n_obs
        sum2 -= 2.0 * g_curr * run_sum_x
        sum2 += run_sum_x2
        d_curr = 0.5 * sum2 + run_b
        np.divide(d_curr, run_d_denom, out=d_curr)

        # Relative changes, guarded against a zero previous value.
        delta_g = np.abs(g_curr - g_prev) / np.maximum(np.abs(g_prev), 1e-12)
        delta_d = np.abs(d_curr - d_prev) / np.maximum(np.abs(d_prev), 1e-12)
        if not every_row_active:
            delta_g = np.where(run_active, delta_g, -np.inf)
            delta_d = np.where(run_active, delta_d, -np.inf)
            g_curr = np.where(run_active, g_curr, g_prev)
            d_curr = np.where(run_active, d_curr, d_prev)
        max_g, max_d = delta_g.max(axis=1), delta_d.max(axis=1)
        # Python's max(a, b): b only when b > a, so a NaN scale change is ignored
        # while the location change is finite, as in the per-batch solver.
        change = np.where(max_d > max_g, max_d, max_g)
        stop_change[rows] = change

        finished = ~np.isfinite(change) | (change < conv)
        if not finished.any():
            continue
        done = rows[finished]
        stop_iteration[done] = iteration + 1
        g_star[done] = g_curr[finished]
        d_star[done] = d_curr[finished]
        keep = ~finished
        rows = rows[keep]
        g_curr, d_curr, run_active = g_curr[keep], d_curr[keep], run_active[keep]
        run_t2_n, run_t2_n_g_hat, run_n_obs = run_t2_n[keep], run_t2_n_g_hat[keep], run_n_obs[keep]
        run_sum_x, run_sum_x2 = run_sum_x[keep], run_sum_x2[keep]
        run_d_denom, run_g_bar, run_b = run_d_denom[keep], run_g_bar[keep], run_b[keep]

    unconverged = set(rows.tolist())
    for k, row in enumerate(rows.tolist()):
        g_star[row] = g_curr[k]
        d_star[row] = d_curr[k]
    for row in range(len(batches_ind)):
        if row in unconverged:
            logger.warning(
                "Batch did not converge after %d iterations (final change=%.2e)",
                max_iter,
                stop_change[row],
            )
        elif stop_iteration[row] and not np.isfinite(stop_change[row]):
            logger.warning(
                "Solver stopped at iteration %d: the change is not finite", stop_iteration[row]
            )
        elif stop_iteration[row]:
            logger.debug(
                "Converged in %d iterations (change=%.2e)", stop_iteration[row], stop_change[row]
            )
    return g_star, d_star


# ---------------------------------------------------------------------------
# Non-parametric solver
# ---------------------------------------------------------------------------


def _int_eprior(
    s_data: FloatArray,
    g_hat: FloatArray,
    d_hat: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Non-parametric empirical-Bayes location and scale for one batch.

    Each feature's estimate is the average of the other features' estimates,
    weighted by the likelihood of its observed cells under each of them
    (R ``sva:::int.eprior``). The computation is deterministic: a weighted sum
    over features, in blocks of target rows of at most ``_EPRIOR_BLOCK_CELLS``
    cells, or ``_EPRIOR_MIN_BLOCK_ROWS`` rows if more.

    The log-likelihood of target *i* under support *j* is written as
    ``-(SS_i + n_i (m_i - g_j)^2) / (2 d_j) - n_i log(2 pi d_j) / 2``, with
    *m_i* the target's mean and *SS_i* its sum of squared deviations. Expanding
    the square into sums of ``x^2`` and ``x`` loses precision when a target's
    values lie far from the support estimates relative to their spread, so the
    deviation from the mean is formed directly, as R does. The row maximum is
    subtracted before ``exp``, so the weights cannot all underflow; a row whose
    weights are not finite falls back to equal weights over the other features.
    The weighted sums use one matrix product per block.

    Each target uses its own observed cells and their count, as R does with
    ``x <- sdat[i, !is.na(sdat[i, ])]``.

    Parameters
    ----------
    s_data : ndarray, shape (n_features, n_batch_samples)
        Standardised data of the batch; missing cells are allowed.
    g_hat, d_hat : ndarray, shape (n_features,)
        Location and scale estimates of the batch.

    Returns
    -------
    gamma_star, delta_star : ndarray, shape (n_features,)
        Posterior estimates; a feature with no observed cell keeps its input
        estimates.
    """
    n_features, _ = s_data.shape
    g_star = g_hat.copy()
    d_star = d_hat.copy()

    if n_features <= 1:
        return g_star, d_star

    d = np.maximum(d_hat, _VARIANCE_FLOOR)
    observed = ~np.isnan(s_data)
    if observed.all():
        n_obs = np.full(n_features, float(s_data.shape[1]))
        mean = s_data.sum(axis=1) / s_data.shape[1]
        resid = s_data - mean[:, np.newaxis]
    else:
        n_obs = observed.sum(axis=1).astype(np.float64)
        mean = np.nansum(s_data, axis=1) / np.maximum(n_obs, 1.0)
        resid = np.where(observed, s_data - mean[:, np.newaxis], 0.0)
    sum_sq = (resid * resid).sum(axis=1)
    half_inv_d = 0.5 / d
    half_log_two_pi_d = 0.5 * np.log(2.0 * np.pi * d)
    support = np.column_stack([g_hat, d, np.ones(n_features)])
    targets = np.flatnonzero(n_obs >= 1.0)
    block_rows = min(
        max(targets.size, 1), max(_EPRIOR_MIN_BLOCK_ROWS, _EPRIOR_BLOCK_CELLS // n_features)
    )
    buffer = np.empty((block_rows, n_features))

    for start in range(0, targets.size, block_rows):
        rows = targets[start : start + block_rows]
        log_lh = buffer[: rows.size]
        np.subtract(mean[rows, np.newaxis], g_hat[np.newaxis, :], out=log_lh)
        np.square(log_lh, out=log_lh)
        log_lh *= n_obs[rows, np.newaxis]
        log_lh += sum_sq[rows, np.newaxis]
        log_lh *= half_inv_d
        log_lh += n_obs[rows, np.newaxis] * half_log_two_pi_d
        np.negative(log_lh, out=log_lh)
        log_lh[np.arange(rows.size), rows] = -np.inf
        log_lh -= log_lh.max(axis=1, keepdims=True)
        np.exp(log_lh, out=log_lh)
        sums = log_lh @ support

        bad = (sums[:, 2] == 0.0) | ~np.isfinite(sums[:, 2])
        if np.any(bad):
            log_lh[bad] = 1.0
            log_lh[np.flatnonzero(bad), rows[bad]] = 0.0
            sums[bad] = log_lh[bad] @ support

        g_star[rows] = sums[:, 0] / sums[:, 2]
        d_star[rows] = sums[:, 1] / sums[:, 2]

    return g_star, d_star


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

_design_cache: dict[
    tuple[int, bytes], tuple[FloatArray, FloatArray, list[npt.NDArray[np.intp]]]
] = {}
# Upper bound on the memory the design cache keeps alive between calls.
_DESIGN_CACHE_MAX_BYTES = 16 * 2**20


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
    layout. After each insertion the oldest layouts are dropped until the cache
    holds at most 64 layouts and at most ``_DESIGN_CACHE_MAX_BYTES``, so that
    wide matrices do not keep large designs alive after a run; the newest
    layout is always kept.
    """
    key = (n_batch, batch_int.tobytes())
    cached = _design_cache.get(key)
    if cached is not None:
        return cached

    design = _make_design(batch_int, n_batch)
    layout = (design, design @ design.T, [np.where(batch_int == i)[0] for i in range(n_batch)])
    _design_cache[key] = layout
    while len(_design_cache) > 1 and (
        len(_design_cache) > 64 or _design_cache_bytes() > _DESIGN_CACHE_MAX_BYTES
    ):
        del _design_cache[next(iter(_design_cache))]
    return layout


def _design_cache_bytes() -> int:
    """Return the bytes held by the design cache."""
    return sum(
        entry[0].nbytes + entry[1].nbytes + sum(ind.nbytes for ind in entry[2])
        for entry in _design_cache.values()
    )


@dataclass(frozen=True)
class _BatchColumns:
    """The columns of each batch, made contiguous by one stable sort.

    ``sum`` reduces each batch's columns to one column and ``expand`` repeats a
    per-batch column over that batch's columns, so per-batch statistics take a
    few array operations whatever the number of batches.
    """

    ids: npt.NDArray[np.intp]
    sizes: npt.NDArray[np.intp]
    order: npt.NDArray[np.intp]
    starts: npt.NDArray[np.intp]

    @classmethod
    def of(cls, batch_int: npt.NDArray[np.intp]) -> _BatchColumns:
        ids, sizes = np.unique(batch_int, return_counts=True)
        starts = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.intp)
        return cls(ids, sizes, np.argsort(batch_int, kind="stable"), starts)

    def sum(self, values: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Sum the columns of each batch of ``values``, whose columns are in ``order``."""
        return np.add.reduceat(values, self.starts, axis=1)

    def expand(self, per_batch: FloatArray) -> FloatArray:
        """Repeat each batch's column over that batch's columns, in ``order``."""
        return np.repeat(per_batch, self.sizes, axis=1)


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
    the engines; the non-parametric modes pass ``mean_only=True`` to check the
    pooled variance only.
    """
    layout = _BatchColumns.of(batch_int)
    mean_only = mean_only or bool(np.any(layout.sizes < 2))
    x = data[:, layout.order]
    obs = ~np.isnan(x)
    has_nan = not bool(obs.all())
    if has_nan:
        n_obs = layout.sum(obs.astype(np.intp))
        means = layout.sum(np.where(obs, x, 0.0)) / np.maximum(n_obs, 1)
        resid = np.where(obs, x - layout.expand(means), 0.0)
    else:
        n_obs = np.broadcast_to(layout.sizes, (x.shape[0], layout.sizes.size))
        resid = x - layout.expand(layout.sum(x) / layout.sizes)
    ss = layout.sum(resid * resid)

    if ref_batch is None:
        ss_pooled, n_pooled, n_cols = ss.sum(axis=1), n_obs.sum(axis=1), data.shape[1]
    else:
        ref = int(np.searchsorted(layout.ids, ref_batch))
        ss_pooled, n_pooled, n_cols = ss[:, ref], n_obs[:, ref], int(layout.sizes[ref])
    if has_nan:
        var_pooled = np.where(n_pooled <= 1, 1.0, ss_pooled / np.maximum(n_pooled - 1, 1))
    else:
        var_pooled = ss_pooled / n_cols

    unestimable = var_pooled < _VARIANCE_FLOOR
    if not mean_only:
        scale = np.maximum(var_pooled, _VARIANCE_FLOOR)[:, np.newaxis]
        var_b = ss / np.maximum(n_obs - 1, 1)
        unestimable |= ((n_obs >= 2) & (var_b < _VARIANCE_FLOOR * scale)).any(axis=1)
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
    """Correct batch effects with ComBat.

    Missing cells are allowed: each feature's batch means and variances use
    its observed cells only, as R ``sva::ComBat`` 3.60.0 does (``Beta.NA``),
    and missing cells stay missing in the output.

    Some features are returned with their raw values, and the run log counts
    them:

    - A feature whose pooled within-batch variance is below the 1e-12 floor
      (with ``ref_batch``, the reference batch's variance). Its values cannot
      be standardised, so its correction would not be estimated from its data.
    - In the parametric modes outside mean-only mode, a feature with a batch
      whose variance is below 1e-12 of the pooled variance. It would otherwise
      set the empirical-Bayes prior for every other feature. The non-parametric
      modes correct such a feature: its batch locations are estimable, and the
      floored batch gives it no weight in the other features' priors.

    The other features are corrected as if those left raw were absent.
    - A single feature, which has no other features to share a prior with.
    - A feature with values in a single batch, which has no batch effect to
      estimate.
    - With ``ref_batch``, a feature with no value in the reference batch.

    A feature with no observed value is returned as missing and does not enter
    the fit of the others. A feature with no value in some batches is fitted on
    the batches it has: the missing batches enter neither its grand mean nor
    their priors. Because each batch's prior is shared, such a feature also
    contributes to the priors of the batches it is observed in.

    Parameters
    ----------
    data : array_like, shape (n_features, n_samples)
        Log-scale values; missing cells are allowed. Not modified.
    batch : array_like, shape (n_samples,)
        Batch label per sample: whole numbers, one batch per distinct value.
    par_prior : bool
        Parametric empirical-Bayes priors (modes 1 and 2) when ``True``,
        non-parametric (modes 3 and 4) when ``False``.
    mean_only : bool
        Correct location only and leave the scale unchanged (modes 2 and 4).
        A batch with a single sample makes the fit location-only whatever this
        value is.
    ref_batch : int or None
        A batch left unchanged, to which the others are adjusted. It needs at
        least two samples.

    Returns
    -------
    ndarray, shape (n_features, n_samples)
        The corrected values, as a new array.

    Raises
    ------
    ValueError
        On wrong dimensionality, batch length mismatch, batch labels that are
        not whole numbers, infinite values, values of magnitude 1e150 or more
        (whose squares overflow the variance sums), or a ``ref_batch`` that is
        not among the batch labels or has one sample.

    Examples
    --------
    >>> import numpy as np
    >>> from harmonizepy import combat
    >>> data = np.random.default_rng(0).normal(10, 2, (50, 12))
    >>> batch = np.array([0] * 4 + [1] * 4 + [2] * 4)
    >>> corrected = combat(data, batch)
    >>> corrected.shape
    (50, 12)

    Notes
    -----
    As in ``sva::ComBat``, the pooled variance is the mean square of the
    residuals when the matrix has no missing cell and their sample variance
    when it has any, so one missing cell multiplies every feature's pooled
    variance by n / (n - 1), and its standard deviation by the square root of
    that. The fit is logged at DEBUG level, features left raw at INFO level,
    and a batch whose solver does not converge at WARNING level, on the
    ``harmonizepy.combat`` logger.
    """
    data = np.asarray(data, dtype=np.float64)
    batch_int = batch_codes(batch)

    validate_engine_input(data, batch_int)
    reject_combat_overflow(data, range(data.shape[0]))
    validate_ref_batch(batch_int, ref_batch)
    corrected, _ = _combat_validated(
        data, batch_int, par_prior=par_prior, mean_only=mean_only, ref_batch=ref_batch
    )
    return corrected


def _combat_with_reasons(
    data: FloatArray,
    batch: npt.ArrayLike,
    *,
    par_prior: bool = True,
    mean_only: bool = False,
) -> tuple[FloatArray, npt.NDArray[np.str_]]:
    """Validate and correct as ``combat()`` does, and return why each feature was left raw.

    The reasons are those of ``_combat_validated``; an empty string means the
    feature was corrected.
    """
    data = np.asarray(data, dtype=np.float64)
    batch_int = batch_codes(batch)
    validate_engine_input(data, batch_int)
    reject_combat_overflow(data, range(data.shape[0]))
    return _combat_validated(
        data, batch_int, par_prior=par_prior, mean_only=mean_only, ref_batch=None
    )


def _combat_validated(
    data: FloatArray,
    batch_int: npt.NDArray[np.intp],
    *,
    par_prior: bool,
    mean_only: bool,
    ref_batch: int | None,
) -> tuple[FloatArray, npt.NDArray[np.str_]]:
    """Set aside the features ComBat cannot fit, then run the dense or missing-value path.

    Features with no observed value stay missing. Features with values in a
    single batch, features with no value in the reference batch, and features
    whose variance the engine would floor keep their raw values; the
    non-parametric modes check the pooled variance only. The rest are corrected as if those features were
    absent. When fewer than two features remain, they are returned as given,
    since a prior needs at least two.

    Returns the corrected matrix and, per feature, an empty string when it was
    corrected or the reason it was not: ``"no_values"``, ``"single_batch"``,
    ``"no_reference"``, ``"zero_variance"``, or ``"too_few_features"``.
    """
    reasons = np.full(data.shape[0], "", dtype="<U16")

    def set_aside(
        mask: npt.NDArray[np.bool_], reason: str
    ) -> tuple[FloatArray, npt.NDArray[np.str_]]:
        out = data.copy()
        reasons[mask] = reason
        keep = ~mask
        if keep.sum() >= 2:
            out[keep], reasons[keep] = _combat_validated(
                data[keep], batch_int, par_prior=par_prior, mean_only=mean_only, ref_batch=ref_batch
            )
        else:
            reasons[keep] = "too_few_features"
        return out, reasons

    no_values = np.isnan(data).all(axis=1)
    if no_values.all():
        reasons[:] = "no_values"
        return data.copy(), reasons
    if no_values.any():
        out = data.copy()
        reasons[no_values] = "no_values"
        out[~no_values], reasons[~no_values] = _combat_validated(
            data[~no_values],
            batch_int,
            par_prior=par_prior,
            mean_only=mean_only,
            ref_batch=ref_batch,
        )
        return out, reasons

    columns = _BatchColumns.of(batch_int)
    observed_batches = (columns.sum((~np.isnan(data[:, columns.order])).astype(np.intp)) > 0).sum(
        axis=1
    )
    one_batch = observed_batches < 2
    if one_batch.all():
        reasons[:] = "single_batch"
        return data.copy(), reasons
    if one_batch.any():
        logger.info(
            "%d feature(s) with values in a single batch left uncorrected", int(one_batch.sum())
        )
        out = data.copy()
        reasons[one_batch] = "single_batch"
        out[~one_batch], reasons[~one_batch] = _combat_validated(
            data[~one_batch],
            batch_int,
            par_prior=par_prior,
            mean_only=mean_only,
            ref_batch=ref_batch,
        )
        return out, reasons

    if ref_batch is not None:
        no_reference = np.isnan(data[:, batch_int == ref_batch]).all(axis=1)
        if no_reference.any():
            logger.info(
                "%d feature(s) with no value in the reference batch left uncorrected",
                int(no_reference.sum()),
            )
            return set_aside(no_reference, "no_reference")

    unestimable = _unestimable_rows(
        data, batch_int, mean_only=mean_only or not par_prior, ref_batch=ref_batch
    )
    if unestimable.any():
        logger.info(
            "%d feature(s) with within-batch variance below the floor left uncorrected",
            int(unestimable.sum()),
        )
        return set_aside(unestimable, "zero_variance")

    if data.shape[0] < 2:
        logger.debug("Single feature input, returning copy")
        reasons[:] = "too_few_features"
        return data.copy(), reasons

    missing = np.isnan(data)
    if not missing.any():
        corrected = _combat_dense(
            data, batch_int, par_prior=par_prior, mean_only=mean_only, ref_batch=ref_batch
        )
    else:
        corrected = _combat_nan(
            data, batch_int, par_prior=par_prior, mean_only=mean_only, ref_batch=ref_batch
        )
    return corrected, reasons


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
    """Return each feature's batch means over its observed cells (sva's ``Beta.NA``).

    With the one-hot batch design, the least-squares coefficient of a batch is
    the mean of the feature's observed values in it; a batch with no observed
    value gets 0, the minimum-norm solution. Rows with the same pattern of
    observed cells are computed together. Shape (n_batches, n_features).
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
    """Return each row's sample variance (``ddof=1``) over its observed cells.

    Rows with the same pattern of observed cells are computed together. A row
    with at most one observed cell gets 1.0.
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
    observed: npt.NDArray[np.bool_] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Shrink the per-batch location and scale estimates with the empirical-Bayes prior.

    Parametric modes use the normal and inverse-gamma priors (the iterative
    solver, or the closed-form posterior mean in location-only mode);
    non-parametric modes use the likelihood-weighted average over features.
    The reference batch, if any, gets location 0 and scale 1.

    *observed*, shape (n_batches, n_features), marks the batches in which each
    feature has a value; ``None`` means every feature has a value in every
    batch. A batch's prior then uses only the features observed in it; the
    estimates of a feature for a batch it lacks have no cell to adjust. A batch
    observed by fewer than two features has no prior to estimate, so its
    location and scale estimates are used unshrunk.
    """
    few: npt.NDArray[np.bool_] | None = None
    if observed is None:
        gamma_bar = gamma_hat.mean(axis=1)
        t2 = gamma_hat.var(axis=1, ddof=1)
        delta_for_prior = list(delta_hat)
    else:
        few = observed.sum(axis=1) < 2
        # Placeholder priors keep the shrinkage finite for batches whose
        # estimates are restored unshrunk below.
        gamma_bar = np.array(
            [0.0 if few[i] else gamma_hat[i, observed[i]].mean() for i in range(layout.n_batch)]
        )
        t2 = np.array(
            [
                1.0 if few[i] else gamma_hat[i, observed[i]].var(ddof=1)
                for i in range(layout.n_batch)
            ]
        )
        delta_for_prior = [delta_hat[i, observed[i]] for i in range(layout.n_batch)]
    if par_prior and not layout.mean_only:
        priors = [_hyper_priors(delta_for_prior[i]) for i in range(layout.n_batch)]
    else:
        priors = [(1.0, 1.0)] * layout.n_batch

    if par_prior and not layout.mean_only:
        a_prior = np.array([prior[0] for prior in priors])
        b_prior = np.array([prior[1] for prior in priors])
        gamma_star, delta_star = _it_sol_batches(
            s_data, layout.batches_ind, gamma_hat, delta_hat, gamma_bar, t2, a_prior, b_prior
        )
    else:
        gamma_star = np.empty_like(gamma_hat)
        delta_star = np.empty_like(gamma_hat)
        for i, idx in enumerate(layout.batches_ind):
            if par_prior:
                gamma_star[i] = _postmean_unit_scale(gamma_bar[i], t2[i], gamma_hat[i])
                delta_star[i] = 1.0
            elif observed is None:
                gamma_star[i], d_star = _int_eprior(s_data[:, idx], gamma_hat[i], delta_hat[i])
                delta_star[i] = 1.0 if layout.mean_only else d_star
            else:
                rows = observed[i]
                gamma_star[i], delta_star[i] = gamma_hat[i], delta_hat[i]
                g_sub, d_sub = _int_eprior(
                    s_data[rows][:, idx], gamma_hat[i, rows], delta_hat[i, rows]
                )
                gamma_star[i, rows] = g_sub
                delta_star[i, rows] = 1.0 if layout.mean_only else d_sub

    if few is not None and few.any():
        gamma_star[few] = gamma_hat[few]
        delta_star[few] = 1.0 if layout.mean_only else delta_hat[few]
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
    logger.debug(
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
    observed = np.stack([(~np.isnan(data[:, idx])).any(axis=1) for idx in layout.batches_ind])
    partial = ~observed.all(axis=0)

    if layout.ref_idx is not None:
        grand_mean = b_hat[layout.ref_idx]
        ref_cols = layout.batches_ind[layout.ref_idx]
        fitted_ref = (layout.design[:, ref_cols].T @ b_hat).T
        var_n = _row_var_nan_grouped(data[:, ref_cols] - fitted_ref)
    else:
        grand_mean = (layout.batch_sizes / data.shape[1]) @ b_hat
        if partial.any():
            weights = observed[:, partial] * layout.batch_sizes[:, np.newaxis]
            grand_mean[partial] = (weights * b_hat[:, partial]).sum(axis=0) / weights.sum(axis=0)
        residuals = (layout.design.T @ b_hat).T
        np.subtract(data, residuals, out=residuals)
        var_n = _row_var_nan_grouped(residuals, valid_row_groups)
        del residuals

    std_pooled = np.sqrt(np.maximum(var_n, _VARIANCE_FLOOR))
    stand_mean = grand_mean[:, np.newaxis]
    s_data = data - stand_mean
    s_data /= std_pooled[:, np.newaxis]

    gamma_hat = _beta_na_grouped_batch_design(
        s_data, layout.design, valid_row_groups, grouped_layouts
    )
    delta_hat = np.ones_like(gamma_hat)
    if not layout.mean_only:
        for b, idx in enumerate(layout.batches_ind):
            delta_hat[b] = _row_var_nan_grouped(s_data[:, idx])

    gamma_star, delta_star = _estimate_batch_effects(
        s_data,
        gamma_hat,
        delta_hat,
        layout,
        par_prior=par_prior,
        observed=observed if partial.any() else None,
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
    np.square(residuals, out=residuals)
    var_pooled = residuals.mean(axis=0)
    del residuals

    std_pooled = np.sqrt(np.maximum(var_pooled, _VARIANCE_FLOOR))
    stand_mean = grand_mean[:, np.newaxis]
    s_data = data - stand_mean
    s_data /= std_pooled[:, np.newaxis]

    gamma_hat = np.linalg.solve(layout.xx_t, layout.design @ s_data.T)
    delta_hat = np.ones_like(gamma_hat)
    if not layout.mean_only:
        for i, idx in enumerate(layout.batches_ind):
            delta_hat[i] = s_data[:, idx].var(axis=1, ddof=1)

    gamma_star, delta_star = _estimate_batch_effects(
        s_data, gamma_hat, delta_hat, layout, par_prior=par_prior
    )
    return _adjust(s_data, data, gamma_star, delta_star, std_pooled, stand_mean, layout)
