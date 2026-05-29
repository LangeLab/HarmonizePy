"""In-process Python benchmark runner.

No subprocess, no CLI, no disk I/O during timing.  The runner calls
the public ``harmonize()`` API directly and measures wall-clock time,
CPU usage, memory, and RSS deltas per run.

Usage::

    from benchmarks.runners.python_runner import run_once, run_scenario
    from benchmarks.metrics import aggregate_metrics

    metrics = run_once(data_df, desc_df, scenario)
    agg = aggregate_metrics([metrics])
"""

from __future__ import annotations

import gc
import logging
import resource
import time
import tracemalloc
from pathlib import Path

import pandas as pd

from ..metrics import SingleRunMetrics
from ..scenarios import Scenario

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RSS reader (Linux /proc/self/status)
# ---------------------------------------------------------------------------


def _read_proc_rss_kb() -> int:
    """Read current RSS from ``/proc/self/status`` (Linux).  Falls back to 0."""
    try:
        with Path("/proc/self/status").open() as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def run_once(
    data_df: pd.DataFrame,
    desc_df: pd.DataFrame,
    scenario: Scenario,
    *,
    unique_removal: bool = True,
    needed_values: int | None = None,
    keep_result_df: bool = True,
) -> SingleRunMetrics:
    """Run a single benchmark iteration and return measurements.

    Calls the public ``harmonize()`` API directly (no subprocess, no
    disk I/O).  Feature counts are computed from the result DataFrame:
    all-NaN rows are passthrough features, others are corrected.

        RSS is measured three ways:

        - ``rss_before_kb``: post-GC baseline before the run
        - ``rss_after_kb``: RSS immediately after the call returns
        - ``rss_post_gc_kb``: retained RSS after dropping transient objects and
            forcing GC

        The post-GC value is the most stable repeated-run signal because it is less
        sensitive to short-lived benchmark harness objects.

    Parameters
    ----------
    data_df : DataFrame
        Features x samples matrix.
    desc_df : DataFrame
        Batch description.
    scenario : Scenario
        Scenario to run (provides ``algorithm``, ``combat_mode``,
        ``block``, ``sort``).
    unique_removal : bool
        Passed through to ``harmonize()``.
    needed_values : int or None
        Passed through to ``harmonize()``.
    keep_result_df : bool
        When ``True`` (default), retain the corrected DataFrame in the
        returned metrics for validity or concordance checks. Timed benchmark
        repetitions should set this to ``False`` so prior runs do not keep
        full result frames alive.

    Returns
    -------
    SingleRunMetrics
        Elapsed time, CPU, memory, RSS, and scalar feature counts.
        The corrected DataFrame is included in ``.result`` only when
        ``keep_result_df=True``.
    """
    from harmonizepy import harmonize

    # Suppress harmonizepy INFO logging during timing (syscall overhead per line)
    hz_logger = logging.getLogger("harmonizepy")
    _prev_level = hz_logger.level
    hz_logger.setLevel(logging.WARNING)

    gc.collect()
    rss_before_kb = _read_proc_rss_kb()

    tracemalloc.start()

    ru_before = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()

    result_df = harmonize(
        data_df, desc_df,
        algorithm=scenario.algorithm,  # type: ignore[arg-type]
        combat_mode=scenario.combat_mode or 1,
        block=scenario.block,
        sort=scenario.sort,
        unique_removal=unique_removal,
        needed_values=needed_values,
        output_file=None,
    )
    elapsed = time.perf_counter() - t0
    ru_after = resource.getrusage(resource.RUSAGE_SELF)

    hz_logger.setLevel(_prev_level)

    _, peak_alloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after_kb = _read_proc_rss_kb()

    cpu_user = ru_after.ru_utime - ru_before.ru_utime
    cpu_sys = ru_after.ru_stime - ru_before.ru_stime
    cpu_pct = 100.0 * (cpu_user + cpu_sys) / elapsed if elapsed > 0 else 0.0

    # Compute feature counts and total memory.
    # tracemalloc only tracks Python objects. NumPy allocates array data
    # natively (C malloc / mmap), invisible to tracemalloc. We add the
    # result array's nbytes to capture the dominant NumPy cost.
    n_total = len(result_df)
    n_passthrough = int(result_df.isna().all(axis=1).sum())
    n_corrected = n_total - n_passthrough

    # Peak memory: tracemalloc (Python objects) + result ndarray (NumPy data)
    peak_bytes = peak_alloc
    try:
        peak_bytes += result_df.to_numpy().nbytes
    except Exception:
        pass

    result_payload: dict[str, object] = {
        "n_corrected": n_corrected,
        "n_passthrough": n_passthrough,
        "n_total": n_total,
    }
    if keep_result_df:
        result_payload["result_df"] = result_df
    else:
        del result_df

    gc.collect()
    rss_post_gc_kb = _read_proc_rss_kb()

    return SingleRunMetrics(
        elapsed_s=elapsed,
        cpu_pct=cpu_pct,
        tracemalloc_peak_mb=peak_bytes / (1024 * 1024),
        rss_before_kb=rss_before_kb,
        rss_after_kb=rss_after_kb,
        rss_post_gc_kb=rss_post_gc_kb,
        rss_delta_kb=rss_after_kb - rss_before_kb,
        rss_delta_post_gc_kb=rss_post_gc_kb - rss_before_kb,
        result=result_payload,
    )


# ---------------------------------------------------------------------------
# Adaptive repetition loop
# ---------------------------------------------------------------------------


def run_scenario(
    data_df: pd.DataFrame,
    desc_df: pd.DataFrame,
    scenario: Scenario,
    *,
    budget_s: int = 30,
    min_reps: int = 3,
    max_reps: int = 10,
    unique_removal: bool = True,
    needed_values: int | None = None,
) -> list[SingleRunMetrics]:
    """Run timed repetitions of a scenario with adaptive scheduling.

    The harness is responsible for an untimed warmup run (for validity
    checks) before calling this function.

    Parameters
    ----------
    data_df : DataFrame
        Features x samples matrix.
    desc_df : DataFrame
        Batch description.
    scenario : Scenario
        Scenario parameters.
    budget_s : int
        Per-scenario time budget in seconds (default 30).
    min_reps : int
        Minimum timed repetitions (default 3).
    max_reps : int
        Maximum timed repetitions (default 10).
    unique_removal : bool
        Passed through to the pipeline.
    needed_values : int or None
        Passed through to the pipeline.

    Returns
    -------
    list[SingleRunMetrics]
        Timed runs.  At least *min_reps* entries unless the budget
        prevents it (logged as warning).
    """
    runs: list[SingleRunMetrics] = []

    # First rep determines adaptive schedule
    m1 = run_once(
        data_df, desc_df, scenario,
        unique_removal=unique_removal,
        needed_values=needed_values,
        keep_result_df=False,
    )
    runs.append(m1)
    accumulated = m1.elapsed_s

    # Determine how many more reps to run
    if accumulated * min_reps > budget_s:
        logger.warning(
            "Scenario %s: single rep %.3fs exceeds budget (%ds), "
            "running minimum %d reps",
            scenario.id, accumulated, budget_s, min_reps,
        )
        for _ in range(min_reps - 1):
            runs.append(
                run_once(
                    data_df, desc_df, scenario,
                    unique_removal=unique_removal,
                    needed_values=needed_values,
                    keep_result_df=False,
                )
            )
    else:
        while accumulated < budget_s and len(runs) < max_reps:
            m = run_once(
                data_df, desc_df, scenario,
                unique_removal=unique_removal,
                needed_values=needed_values,
                keep_result_df=False,
            )
            runs.append(m)
            accumulated += m.elapsed_s

    return runs
