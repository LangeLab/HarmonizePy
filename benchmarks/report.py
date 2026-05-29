"""JSON and Markdown report generation for benchmark results.

Converts harness ``ScenarioResult`` tuples into the v2.0 output JSON
schema (plan Section 13) and generates human-readable Markdown reports
comparable to the old ``benchmarks/RESULTS.md``.

Usage::

    from benchmarks.report import build_report_json, write_report_json, generate_markdown

    results = harness.run_all()
    report = build_report_json(results, cfg)
    write_report_json(report, "results/benchmark.json")
    md = generate_markdown(report)
    Path("results/report.md").write_text(md)
"""

from __future__ import annotations

import json
import platform
import statistics
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from .metrics import ScenarioMetrics
from .scenarios import Config
from .validity import ValidityResult

# ---------------------------------------------------------------------------
# JSON report builder
# ---------------------------------------------------------------------------


def build_report_json(
    results: list[tuple[Any, ScenarioMetrics, ValidityResult | None, dict[str, Any] | None]],
    config: Config,
    *,
    r_startup_s: float | None = None,
    r_version: str | None = None,
    harmonizr_version: str | None = None,
    r_cores: int | None = None,
) -> dict[str, Any]:
    """Build a v2.0 benchmark results JSON object from harness results.

    Parameters
    ----------
    results : list[ScenarioResult]
        Output from ``BenchmarkHarness.run_all``.
    config : Config
        Loaded benchmark config.
    r_startup_s : float or None
        R startup time in seconds (from ``cache-r`` if available).
    r_version : str or None
        R version string.
    harmonizr_version : str or None
        HarmonizR version string.
    r_cores : int or None
        Number of cores used for R runs.

    Returns
    -------
    dict
        Serializable JSON object matching the v2.0 schema.
    """
    r_cores = r_cores or config.r_cache_default_cores

    # System info
    system = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "unknown",
        "python": platform.python_version(),
        "harmonizepy": version("harmonizepy"),
        "r": r_version,
        "r_available": r_version is not None,
        "harmonizr": harmonizr_version,
        "r_startup_s": r_startup_s,
    }

    entries = []
    seen_datasets: set[str] = set()
    for scenario, s_metrics, validity, r_entry in results:
        entry = _build_scenario_entry(scenario, s_metrics, validity, r_entry, r_cores, r_startup_s)
        entries.append(entry)
        if scenario.dataset not in seen_datasets:
            seen_datasets.add(scenario.dataset)

    # Build dataset metadata from config
    datasets_meta: dict[str, dict[str, Any]] = {}
    for ds_name in sorted(seen_datasets):
        try:
            ds_spec = config.datasets[ds_name]
            # Derive type label from tags
            tags = ds_spec.tags
            if "real" in tags:
                extra = " ".join(t for t in tags if t not in ("real", "bulk", "scp"))
                ds_type = f"Real {extra}".strip() if extra else "Real"
            elif "scp" in tags:
                ds_type = "SCP cohort"
            else:
                ds_type = "Bulk proteomics"
            # Calculate file sizes
            from .datasets import resolve_dataset_paths
            paths = resolve_dataset_paths(config, ds_name)
            input_size = paths.input_path.stat().st_size if paths.input_path.is_file() else 0
            desc_size = paths.desc_path.stat().st_size if paths.desc_path.is_file() else 0
            total_kb = (input_size + desc_size) / 1024.0
            size_str = f"{total_kb:.0f} KB" if total_kb < 1024 else f"{total_kb / 1024.0:.1f} MB"
            datasets_meta[ds_name] = {
                "type": ds_type,
                "features": ds_spec.features,
                "samples": ds_spec.samples,
                "batches": ds_spec.batches,
                "file_size": size_str,
                "missing_frac": ds_spec.missing_frac,
            }
        except (KeyError, OSError):
            datasets_meta[ds_name] = {
                "type": "Unknown", "features": 0, "samples": 0, "batches": 0,
                "file_size": "N/A", "missing_frac": None,
            }

    return {
        "schema_version": "2.0",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "system": system,
        "datasets": datasets_meta,
        "results": entries,
    }


def _build_scenario_entry(
    scenario: Any,
    s_metrics: ScenarioMetrics,
    validity: ValidityResult | None,
    r_entry: dict[str, Any] | None,
    r_cores: int,
    r_startup_s: float | None,
) -> dict[str, Any]:
    """Build a single scenario entry for the results JSON."""
    # Scenario metadata
    scenario_obj = {
        "id": scenario.id,
        "dataset": scenario.dataset,
        "algorithm": scenario.algorithm,
        "combat_mode": scenario.combat_mode,
        "block": scenario.block,
        "sort": scenario.sort,
        "tags": list(scenario.tags),
    }

    # Python metrics
    python_obj: dict[str, Any] = {
        "n_reps_actual": s_metrics.n_reps_actual,
        "times_s": s_metrics.times_s,
        "median_s": s_metrics.median_s,
        "stdev_s": s_metrics.stdev_s,
        "p5_s": s_metrics.p5_s,
        "p95_s": s_metrics.p95_s,
        "cpu_pct": statistics.median(s_metrics.cpu_pcts) if s_metrics.cpu_pcts else None,
        "tracemalloc_peak_mb": statistics.median(s_metrics.tracemalloc_peak_mbs) if s_metrics.tracemalloc_peak_mbs else None,
        "rss_post_gc_kbs": s_metrics.rss_post_gc_kbs,
        "rss_before_kb": statistics.median(s_metrics.rss_before_kbs) if s_metrics.rss_before_kbs else None,
        "rss_after_kb": statistics.median(s_metrics.rss_after_kbs) if s_metrics.rss_after_kbs else None,
        "rss_post_gc_kb": statistics.median(s_metrics.rss_post_gc_kbs) if s_metrics.rss_post_gc_kbs else None,
        "rss_delta_kb": statistics.median(s_metrics.rss_delta_kbs) if s_metrics.rss_delta_kbs else None,
        "rss_delta_post_gc_kb": statistics.median(s_metrics.rss_delta_post_gc_kbs) if s_metrics.rss_delta_post_gc_kbs else None,
        "rss_stability": {
            "status": s_metrics.rss_stability.status,
            "reason": s_metrics.rss_stability.reason,
            "tolerance_kb": s_metrics.rss_stability.tolerance_kb,
            "total_growth_kb": s_metrics.rss_stability.total_growth_kb,
            "tail_span_kb": s_metrics.rss_stability.tail_span_kb,
            "monotonic_non_decreasing": s_metrics.rss_stability.monotonic_non_decreasing,
        },
        "features_out": s_metrics.features_out,
        "features_corrected": s_metrics.features_corrected,
        "features_passthrough": s_metrics.features_passthrough,
    }

    # R metrics (from cache)
    r_obj: dict[str, Any] | None = None
    comparison_meta: dict[str, Any] | None = None
    if r_entry is not None and isinstance(r_entry, dict):
        r_data = r_entry
        r_obj = {
            "n_reps_actual": 1,
            "times_s": [r_data.get("times_s")],
            "median_s": r_data.get("times_s"),
            "stdev_s": None,
            "cpu_user_s": [r_data.get("cpu_user_s")],
            "cpu_sys_s": [r_data.get("cpu_sys_s")],
            "rss_delta_kb": r_data.get("rss_delta_kb"),
            "rss_kb": r_data.get("rss_kb"),
            "rss_peak_kb": r_data.get("rss_peak_kb"),
            "r_heap_mb": r_data.get("r_heap_mb"),
            "r_heap_delta_mb": r_data.get("r_heap_delta_mb"),
            "output_tsv": r_data.get("output_tsv"),
            "error": r_data.get("error"),
        }
        comp_type = "realistic" if r_cores and r_cores > 1 else "single_core_fair"
        note = (
            f"R={r_cores} cores (foreach/doParallel), "
            f"Python=1 thread (NumPy single-threaded)."
        ) if r_cores and r_cores > 1 else (
            "R=1 core, Python=1 thread. Equal threading, startup excluded."
        )
        comparison_meta = {
            "r_cores": r_cores,
            "r_startup_s": r_startup_s,
            "comparison_type": comp_type,
            "note": note,
        }

    # Validity
    validity_obj: dict[str, Any] | None = None
    if validity is not None:
        validity_obj = {
            "shape_preserved": validity.shape_preserved,
            "nan_preserved": validity.nan_preserved,
            "no_inf": validity.no_inf,
            "row_count_match": validity.row_count_match,
            "concordance_max_rel": validity.concordance_max_rel,
            "concordance_mean_rel": validity.concordance_mean_rel,
            "concordance_nan_match": validity.concordance_nan_match,
            "concordance_shared_features": validity.concordance_shared_features,
            "error": validity.error,
        }

    return {
        "scenario": scenario_obj,
        "python": python_obj,
        "r": r_obj,
        "comparison_meta": comparison_meta,
        "validity": validity_obj,
    }


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def write_report_json(report: dict[str, Any], path: str | Path) -> None:
    """Write a report JSON to *path*."""
    with Path(path).open("w") as fh:
        json.dump(report, fh, indent=2, default=str)


def read_report_json(path: str | Path) -> dict[str, Any]:
    """Read a v2.0 report JSON from *path*."""
    with Path(path).open() as fh:
        return cast("dict[str, Any]", json.load(fh))


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _detect_blas_threads() -> int:
    """Detect available BLAS threads from the system."""
    import os
    return os.cpu_count() or 32


def generate_markdown(report: dict[str, Any]) -> str:
    """Generate a structured Markdown report with Speed, Memory, and Concordance sections."""
    system = report.get("system", {})
    results = report.get("results", [])

    blas_threads = _detect_blas_threads()

    r_cores_str = "N/A"
    for entry in results:
        cm = entry.get("comparison_meta")
        if cm and cm.get("r_cores"):
            r_cores_str = str(cm["r_cores"])
            break

    lines: list[str] = []

    # ==================================================================
    # Header
    # ==================================================================
    lines.append("# Benchmark Results")
    lines.append("")
    lines.append(f"**Generated:** {report.get('generated_at', '')}")
    lines.append(f"**Platform:** {system.get('os', '')}")
    lines.append(f"**CPU:** {system.get('cpu', '')}")
    lines.append(f"**Threads:** Python=1 thread, R={r_cores_str} core(s), System/BLAS={blas_threads} threads")
    lines.append(f"**Python:** {system.get('python', '')} (harmonizepy v{system.get('harmonizepy', '')})")
    if system.get("r"):
        lines.append(f"**R:** {system['r']} (HarmonizR {system.get('harmonizr', 'N/A')})")
    lines.append("")

    # ==================================================================
    # Data Specifications
    # ==================================================================
    _add_data_specs(lines, report)

    # ==================================================================
    # Speed
    # ==================================================================
    _add_speed_section(lines, results, r_cores_str)

    # ==================================================================
    # Memory
    # ==================================================================
    _add_memory_section(lines, results, r_cores_str)

    # ==================================================================
    # Concordance
    # ==================================================================
    _add_concordance_section(lines, results)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data Specifications
# ---------------------------------------------------------------------------


def _add_data_specs(lines: list[str], report: dict[str, Any]) -> None:
    """Build data specifications table from metadata."""
    datasets_meta = report.get("datasets", {})
    if not datasets_meta:
        return

    lines.append("## Data Specifications")
    lines.append("")
    lines.append("| Dataset | Type | Features | Samples | Batches | Missing | File Size |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for ds_name in sorted(datasets_meta):
        meta = datasets_meta[ds_name]
        ds_type = meta.get("type", "?")
        feat = meta.get("features", "?")
        smp = meta.get("samples", "?")
        bat = meta.get("batches", "?")
        size_str = meta.get("file_size", "?")
        mf = meta.get("missing_frac")
        miss = f"{int(mf * 100)}%" if mf is not None else "N/A"
        lines.append(f"| {ds_name} | {ds_type} | {feat} | {smp} | {bat} | {miss} | {size_str} |")
    lines.append("")


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------


def _has_r_data(results: list[dict[str, Any]]) -> bool:
    return any(
        r.get("r") and r["r"].get("median_s") is not None
        for r in results
    )


def _add_speed_section(lines: list[str], results: list[dict[str, Any]], r_cores: str) -> None:
    """Python vs R timing side by side with ratio."""
    speed_rows = [r for r in results if r.get("python", {}).get("median_s") is not None]
    if not speed_rows:
        return

    lines.append("## Speed")
    lines.append("")
    lines.append(f"Wall-clock time per scenario. Python runs single-threaded. R uses {r_cores} core(s) with system BLAS threading.")
    lines.append("")

    has_r = _has_r_data(results)

    header = "| Dataset | Algorithm | Mode | Block | Sort | Py Med (s) | Py Reps"
    sep    = "| --- | --- | --- | --- | --- | --- | ---"
    if has_r:
        header += " | R Med (s) | R Reps | Ratio (R/Py)"
        sep    += " | --- | --- | ---"
    header += " |"
    sep    += " |"

    lines.append(header)
    lines.append(sep)

    for r in speed_rows:
        sc = r["scenario"]
        py = r["python"]
        ds = sc["dataset"]
        algo = sc["algorithm"]
        mode = str(sc["combat_mode"]) if sc["combat_mode"] is not None else "--"
        block = str(sc["block"]) if sc["block"] is not None else "--"
        sort = sc["sort"] if sc["sort"] is not None else "--"
        py_med = f"{py['median_s']:.4f}"
        py_reps = str(py["n_reps_actual"])
        row = f"| {ds} | {algo} | {mode} | {block} | {sort} | {py_med} | {py_reps}"

        if has_r:
            r_data = r.get("r")
            if r_data and r_data.get("median_s") is not None:
                r_med = f"{r_data['median_s']:.3f}"
                r_reps = str(r_data.get("n_reps_actual", 1))
                ratio = r_data["median_s"] / py["median_s"]
                ratio_str = f"{ratio:.2f}x" if ratio >= 0.01 else f"{ratio:.4f}x"
                row += f" | {r_med} | {r_reps} | {ratio_str}"
            else:
                row += " | -- | -- | --"

        lines.append(row)

    lines.append("")


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def _add_memory_section(lines: list[str], results: list[dict[str, Any]], r_cores: str) -> None:
    """Python vs R memory signals side by side."""
    mem_rows = [r for r in results if r.get("python", {}).get("median_s") is not None]
    if not mem_rows:
        return

    lines.append("## Memory")
    lines.append("")
    lines.append("Python RSS retained delta is measured from a post-GC baseline before the call to a post-GC RSS after transient benchmark objects are released. This is more stable across repeated in-process runs than a raw before/after delta. RSS stability summarizes the repeated timed runs: plateau means the last up to 3 post-GC samples stayed within 1024 KB, and growing means monotonic growth without a stable tail. Raw end-of-call RSS is still stored in the JSON output. Heap is Python tracemalloc peak and R gc() absolute heap. R runs all scenarios in one process, so later scenarios may reuse previously allocated memory.")
    lines.append("")

    has_r = _has_r_data(results)

    header = "| Dataset | Algorithm | Mode | Py RSS stability | Py RSS retained Δ (KB) | R RSS Δ (KB) | Py RSS post-GC (KB) | Py Heap (MB) | R Heap (MB) |"
    sep    = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    if not has_r:
        header = "| Dataset | Algorithm | Mode | Py RSS stability | Py RSS retained Δ (KB) | Py RSS post-GC (KB) | Py Heap (MB) |"
        sep    = "| --- | --- | --- | --- | --- | --- | --- |"

    lines.append(header)
    lines.append(sep)

    for r in mem_rows:
        sc = r["scenario"]
        py = r["python"]
        ds = sc["dataset"]
        algo = sc["algorithm"]
        mode = str(sc["combat_mode"]) if sc["combat_mode"] is not None else "--"
        py_rss_stability = py.get("rss_stability", {})
        py_rss_stability_status = str(py_rss_stability.get("status", "--")).upper()
        py_rss = (
            f"{py.get('rss_delta_post_gc_kb', 0):.0f}"
            if py.get("rss_delta_post_gc_kb") is not None
            else "--"
        )
        py_rss_post_gc = f"{py.get('rss_post_gc_kb', 0):.0f}" if py.get("rss_post_gc_kb") is not None else "--"
        py_heap = f"{py.get('tracemalloc_peak_mb', 0):.2f}" if py.get("tracemalloc_peak_mb") is not None else "--"
        row = f"| {ds} | {algo} | {mode} | {py_rss_stability_status} | {py_rss} |"

        if has_r:
            r_data = r.get("r")
            if r_data and r_data.get("rss_delta_kb") is not None:
                r_rss = f"{r_data['rss_delta_kb']:.0f}"
                r_heap = f"{r_data['r_heap_mb']:.1f}" if r_data.get("r_heap_mb") is not None else "--"
            else:
                r_rss = "--"
                r_heap = "--"
            row += f" {r_rss} | {py_rss_post_gc} | {py_heap} | {r_heap} |"
        else:
            row += f" {py_rss_post_gc} | {py_heap} |"

        lines.append(row)

    lines.append("")


# ---------------------------------------------------------------------------
# Concordance
# ---------------------------------------------------------------------------


def _add_concordance_section(lines: list[str], results: list[dict[str, Any]]) -> None:
    """Python vs R numerical concordance."""
    conc_rows = []
    for r in results:
        v = r.get("validity")
        if v and v.get("concordance_max_rel") is not None:
            conc_rows.append(r)

    if not conc_rows:
        return

    lines.append("## Concordance")
    lines.append("")
    lines.append("Python vs R corrected output comparison on shared features. Relative difference is |py - r| / |r| for non-NaN cells.")
    lines.append("")

    lines.extend([
        "| Dataset | Algorithm | Mode | Block | Sort | Shared Features | Py Only | R Only | NaN Match | Max Rel | Mean Rel |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for r in conc_rows:
        sc = r["scenario"]
        v = r["validity"]
        ds = sc["dataset"]
        algo = sc["algorithm"]
        mode = str(sc["combat_mode"]) if sc["combat_mode"] is not None else "--"
        block = str(sc["block"]) if sc["block"] is not None else "--"
        sort = sc["sort"] if sc["sort"] is not None else "--"
        shared = str(v.get("concordance_shared_features", ""))
        py_only = str(v.get("concordance_py_only_features", ""))
        r_only = str(v.get("concordance_r_only_features", ""))
        nm = "YES" if v.get("concordance_nan_match") else "NO"
        mr = f"{v['concordance_max_rel']:.2e}" if v.get("concordance_max_rel") is not None else "--"
        mn = f"{v['concordance_mean_rel']:.2e}" if v.get("concordance_mean_rel") is not None else "--"
        lines.append(
            f"| {ds} | {algo} | {mode} | {block} | {sort} | "
            f"{shared} | {py_only} | {r_only} | {nm} | {mr} | {mn} |"
        )
    lines.append("")
