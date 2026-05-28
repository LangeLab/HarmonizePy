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
    rss_delta_kb: int
    phase_times: dict[str, float] | None = None
    result: Any = None  # dict with result_df and feature counts from run_once

    def release_result(self) -> None:
        """Drop the result reference to free memory."""
        self.result = None


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
    rss_delta_kbs: list[int] = field(default_factory=list)
    tracemalloc_peak_mbs: list[float] = field(default_factory=list)
    phase_times: dict[str, list[float]] | None = None
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
    rss = [r.rss_delta_kb for r in runs]
    mems = [r.tracemalloc_peak_mb for r in runs]

    # Collect phase times across runs into dict-of-lists
    phase_agg: dict[str, list[float]] | None = None
    for r in runs:
        if r.phase_times is not None:
            if phase_agg is None:
                phase_agg = {k: [] for k in r.phase_times}
            for k, v in r.phase_times.items():
                if k in phase_agg:
                    phase_agg[k].append(v)

    return ScenarioMetrics(
        n_reps_actual=len(runs),
        times_s=times,
        cpu_pcts=cpus,
        rss_delta_kbs=rss,
        tracemalloc_peak_mbs=mems,
        phase_times=phase_agg if phase_agg else None,
    )
