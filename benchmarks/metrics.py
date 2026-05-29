"""Metrics dataclasses and collectors for benchmark runs.

Defines the measurement types used throughout the benchmark system:
single-run metrics, scenario-level aggregates.

Usage::

    from benchmarks.metrics import SingleRunMetrics, ScenarioMetrics
    from benchmarks.metrics import aggregate_metrics
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Single-run metrics
# ---------------------------------------------------------------------------


@dataclass
class SingleRunMetrics:
    """Measurements from a single Python benchmark run.

    The ``result`` field holds the corrected DataFrame (for validity /
    concordance checks).  Release it after use to free memory; the
    harness validity pass reads it from the warmup run and then drops
    the reference.
    """

    elapsed_s: float
    cpu_pct: float
    tracemalloc_peak_mb: float
    rss_before_kb: int = 0
    rss_after_kb: int = 0
    rss_post_gc_kb: int = 0
    rss_delta_kb: int = 0
    rss_delta_post_gc_kb: int = 0
    result: Any = None  # dict with result_df and feature counts from run_once

    def release_result(self) -> None:
        """Drop the result reference to free memory."""
        self.result = None


@dataclass(frozen=True)
class RssStability:
    """Interpretation of repeated post-GC RSS samples for one scenario."""

    status: str
    reason: str
    tolerance_kb: int
    total_growth_kb: int
    tail_span_kb: int
    monotonic_non_decreasing: bool


# ---------------------------------------------------------------------------
# Scenario-level aggregate
# ---------------------------------------------------------------------------


@dataclass
class ScenarioMetrics:
    """Aggregated metrics across multiple runs of the same scenario.

    The raw times list is preserved so that downstream report generation
    can compute summary statistics or percentiles as needed.

    ``features_corrected`` and ``features_passthrough`` are populated
    after the warmup run (they are the same across runs).
    """

    n_reps_actual: int
    times_s: list[float] = field(default_factory=list)
    cpu_pcts: list[float] = field(default_factory=list)
    rss_before_kbs: list[int] = field(default_factory=list)
    rss_after_kbs: list[int] = field(default_factory=list)
    rss_post_gc_kbs: list[int] = field(default_factory=list)
    rss_delta_kbs: list[int] = field(default_factory=list)
    rss_delta_post_gc_kbs: list[int] = field(default_factory=list)
    tracemalloc_peak_mbs: list[float] = field(default_factory=list)
    features_out: int = 0
    features_corrected: int = 0
    features_passthrough: int = 0

    @property
    def median_s(self) -> float:
        if not self.times_s:
            return 0.0
        return float(statistics.median(self.times_s))

    @property
    def stdev_s(self) -> float:
        if len(self.times_s) < 2:
            return 0.0
        return float(statistics.stdev(self.times_s))

    @property
    def p5_s(self) -> float:
        if not self.times_s:
            return 0.0
        sorted_t = sorted(self.times_s)
        idx = max(0, math.ceil(0.05 * len(sorted_t)) - 1)
        return sorted_t[idx]

    @property
    def p95_s(self) -> float:
        if not self.times_s:
            return 0.0
        sorted_t = sorted(self.times_s)
        idx = min(len(sorted_t) - 1, math.ceil(0.95 * len(sorted_t)) - 1)
        return sorted_t[idx]

    @property
    def rss_stability(self) -> RssStability:
        return assess_rss_stability(self.rss_post_gc_kbs)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def aggregate_metrics(runs: list[SingleRunMetrics]) -> ScenarioMetrics:
    """Aggregate a list of single-run metrics into a ScenarioMetrics.

    Parameters
    ----------
    runs : list[SingleRunMetrics]
        One or more runs of the same scenario.

    Returns
    -------
    ScenarioMetrics
        Aggregate with lists of raw measurements and computed statistics.
    """
    times = [r.elapsed_s for r in runs]
    cpus = [r.cpu_pct for r in runs]
    rss_before = [r.rss_before_kb for r in runs]
    rss_after = [r.rss_after_kb for r in runs]
    rss_post_gc = [r.rss_post_gc_kb for r in runs]
    rss = [r.rss_delta_kb for r in runs]
    rss_post_gc_delta = [r.rss_delta_post_gc_kb for r in runs]
    mems = [r.tracemalloc_peak_mb for r in runs]

    return ScenarioMetrics(
        n_reps_actual=len(runs),
        times_s=times,
        cpu_pcts=cpus,
        rss_before_kbs=rss_before,
        rss_after_kbs=rss_after,
        rss_post_gc_kbs=rss_post_gc,
        rss_delta_kbs=rss,
        rss_delta_post_gc_kbs=rss_post_gc_delta,
        tracemalloc_peak_mbs=mems,
    )


def assess_rss_stability(
    rss_post_gc_kbs: list[int],
    *,
    tolerance_kb: int = 1024,
) -> RssStability:
    """Classify repeated-run RSS behavior.

    Rules:
    - ``plateau``: the last up to 3 post-GC samples are within tolerance.
      This includes one-time warm-up growth followed by stabilization.
    - ``growing``: samples are monotonically non-decreasing and total growth
      exceeds tolerance with no stable tail.
    - ``mixed``: behavior is neither plateau nor monotonic growth.
    - ``insufficient-data``: fewer than 2 timed repetitions.
    - ``n/a``: no samples available.
    """
    if not rss_post_gc_kbs:
        return RssStability(
            status="n/a",
            reason="No post-GC RSS samples recorded.",
            tolerance_kb=tolerance_kb,
            total_growth_kb=0,
            tail_span_kb=0,
            monotonic_non_decreasing=True,
        )

    if len(rss_post_gc_kbs) < 2:
        return RssStability(
            status="insufficient-data",
            reason="Need at least two timed repetitions to assess RSS stability.",
            tolerance_kb=tolerance_kb,
            total_growth_kb=0,
            tail_span_kb=0,
            monotonic_non_decreasing=True,
        )

    total_growth_kb = rss_post_gc_kbs[-1] - rss_post_gc_kbs[0]
    tail = rss_post_gc_kbs[-min(3, len(rss_post_gc_kbs)):]
    tail_span_kb = max(tail) - min(tail)
    monotonic_non_decreasing = all(
        curr >= prev for prev, curr in zip(rss_post_gc_kbs, rss_post_gc_kbs[1:])
    )

    if tail_span_kb <= tolerance_kb:
        reason = (
            "RSS rose during warm-up and then plateaued."
            if total_growth_kb > tolerance_kb
            else "RSS stayed within plateau tolerance across timed runs."
        )
        return RssStability(
            status="plateau",
            reason=reason,
            tolerance_kb=tolerance_kb,
            total_growth_kb=total_growth_kb,
            tail_span_kb=tail_span_kb,
            monotonic_non_decreasing=monotonic_non_decreasing,
        )

    if monotonic_non_decreasing and total_growth_kb > tolerance_kb:
        return RssStability(
            status="growing",
            reason="RSS increased monotonically across timed runs without a stable tail.",
            tolerance_kb=tolerance_kb,
            total_growth_kb=total_growth_kb,
            tail_span_kb=tail_span_kb,
            monotonic_non_decreasing=monotonic_non_decreasing,
        )

    return RssStability(
        status="mixed",
        reason="RSS fluctuated beyond plateau tolerance without monotonic growth.",
        tolerance_kb=tolerance_kb,
        total_growth_kb=total_growth_kb,
        tail_span_kb=tail_span_kb,
        monotonic_non_decreasing=monotonic_non_decreasing,
    )
