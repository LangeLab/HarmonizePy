"""Python coordinator for R HarmonizR benchmark baseline.

Manages the R session lifecycle and the persistent cache of R results.
Cache entries are keyed by scenario parameters + dataset file hashes
+ HarmonizR version + core count, so stale entries are automatically
invalidated when any ingredient changes.

Usage::

    from benchmarks.runners.r_runner import (
        r_available, get_harmonizr_version,
        compute_cache_key, cache_hit, read_cache_entry,
        build_scenario_entry, run_r_process,
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, cast

from ..scenarios import Config, Scenario

logger = logging.getLogger(__name__)

_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
_R_SCRIPT = _BENCHMARKS_DIR / "r" / "benchmark_r.R"


# ---------------------------------------------------------------------------
# R availability
# ---------------------------------------------------------------------------


def r_available() -> bool:
    """Return ``True`` if ``Rscript`` is on ``PATH``."""
    return shutil.which("Rscript") is not None


def _run_r_capture(expr: str, timeout: int = 30) -> str:
    """Run an R expression and return the last non-warning line of stdout."""
    result = subprocess.run(
        ["Rscript", "-e", expr],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"R expression failed: {result.stderr.strip()}")
    # Strip renv status messages and other R startup noise
    lines = [ln for ln in result.stdout.splitlines() if ln.strip() and "out-of-sync" not in ln]
    return lines[-1].strip() if lines else ""


def get_r_version() -> str:
    """Get the installed R version string."""
    return _run_r_capture("cat(as.character(getRversion()))")


def get_harmonizr_version() -> str:
    """Get the installed HarmonizR package version."""
    return _run_r_capture(
        "suppressMessages(library(HarmonizR)); cat(as.character(packageVersion('HarmonizR')))",
        timeout=30,
    )


# ---------------------------------------------------------------------------
# Cache key computation
# ---------------------------------------------------------------------------


def _sha256_file(path: str | Path) -> str:
    """Return hex SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_cache_key(
    scenario: Scenario,
    input_path: str | Path,
    desc_path: str | Path,
    harmonizr_version: str,
    cores: int,
) -> str:
    """Compute a stable 12-char cache key for an R scenario.

    The key incorporates:
    - scenario ID (dataset + algorithm + mode + block + sort)
    - SHA-256 of the input data file
    - SHA-256 of the description file
    - HarmonizR version string
    - core count

    If any ingredient changes, a new key is produced and the old cache
    entry becomes stale.
    """
    h = hashlib.sha256()
    h.update(scenario.id.encode())
    h.update(_sha256_file(input_path).encode())
    h.update(_sha256_file(desc_path).encode())
    h.update(harmonizr_version.encode())
    h.update(str(cores).encode())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def _cache_entry_dir(cache_dir: str | Path, cache_key: str) -> Path:
    return Path(cache_dir) / cache_key


def cache_hit(cache_dir: str | Path, cache_key: str) -> bool:
    """Return ``True`` if a valid cache entry exists for *cache_key*."""
    entry_dir = _cache_entry_dir(cache_dir, cache_key)
    return (
        entry_dir.is_dir()
        and (entry_dir / "results.json").is_file()
    )


def read_cache_entry(
    cache_dir: str | Path,
    cache_key: str,
) -> dict[str, Any] | None:
    """Read a cache entry and return the result dict.

    Returns ``None`` if the entry does not exist or is incomplete.
    """
    entry_dir = _cache_entry_dir(cache_dir, cache_key)
    results_path = entry_dir / "results.json"
    output_tsv = entry_dir / "output.tsv"

    if not results_path.is_file():
        return None

    with results_path.open() as fh:
        entry: dict[str, object] = json.load(fh)

    if output_tsv.is_file():
        entry["output_tsv"] = str(output_tsv)

    return entry


def _write_cache_entry(
    cache_dir: str | Path,
    cache_key: str,
    result: dict[str, Any],
    harmonizr_version: str,
    scenario_id: str,
    input_path: str | Path,
    desc_path: str | Path,
    cores: int,
    r_version: str,
    startup_s: float,
) -> None:
    """Write a single scenario result to the cache.

    Args:
        result: The scenario result dict from the R output JSON.
    """
    entry_dir = _cache_entry_dir(cache_dir, cache_key)
    entry_dir.mkdir(parents=True, exist_ok=True)

    # meta.json: key ingredients for debugging
    meta = {
        "scenario_id": scenario_id,
        "core_count": cores,
        "harmonizr_version": harmonizr_version,
        "r_version": r_version,
        "startup_s": startup_s,
        "input_sha256": _sha256_file(input_path),
        "desc_sha256": _sha256_file(desc_path),
        "cache_key": cache_key,
    }
    with (entry_dir / "meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2)

    # Copy output TSV to cache before updating the result path
    output_tsv_cache = entry_dir / "output.tsv"
    output_tsv_src = cast("str | None", result.get("output_tsv"))
    if output_tsv_src and Path(output_tsv_src).is_file():
        shutil.copy2(output_tsv_src, output_tsv_cache)

    # results.json: update output_tsv to the cache copy, then write
    result["output_tsv"] = str(output_tsv_cache)
    with (entry_dir / "results.json").open("w") as fh:
        json.dump(result, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# CSV to TSV conversion (HarmonizR only reads TSV)
# ---------------------------------------------------------------------------


def _ensure_tsv(path: str | Path, tmp_dir: str | Path, dataset_name: str) -> str:
    """If *path* is a CSV, convert to TSV in *tmp_dir* and return the TSV path.

    HarmonizR's ``read.table(sep="\\t")`` cannot parse CSV files, so
    datasets with ``input_format: csv`` (like DIA) must be converted
    before passing to R.  The conversion happens before the R process
    starts, so no I/O is included in timing.
    """
    p = Path(path)
    if p.suffix.lower() == ".csv":
        tsv_path = Path(tmp_dir) / f"{dataset_name}_input.tsv"
        if not tsv_path.is_file():
            import pandas as pd
            df = pd.read_csv(path, index_col=0)
            df.to_csv(tsv_path, sep="\t")
        return str(tsv_path)
    return str(path)


# ---------------------------------------------------------------------------
# Build scenario JSON for R handoff
# ---------------------------------------------------------------------------


def build_scenario_entry(
    scenario: Scenario,
    input_path: str | Path,
    desc_path: str | Path,
    output_tsv: str | Path,
    cores: int,
    timeout_s: int,
    *,
    tmp_dir: str | Path | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Build a scenario dict for the R handoff JSON.

    This is one element of the scenarios array that gets written to
    the ``--scenarios`` file consumed by ``benchmark_r.R``.
    """
    # Only convert data file (HarmonizR reads data with read.table(sep="\t"))
    data_path = _ensure_tsv(input_path, tmp_dir or Path(input_path).parent, dataset_name or str(input_path))
    # Description is always read with read.csv(sep=",") by HarmonizR, so keep original
    desc_path_str = str(desc_path)
    return {
        "id": scenario.id,
        "data": data_path,
        "desc": desc_path_str,
        "output_tsv": str(output_tsv),
        "algorithm": scenario.algorithm,
        "combat_mode": scenario.combat_mode,
        "block": scenario.block,
        "sort": scenario.sort,
        "n_reps": 1,
        "cores": cores,
        "timeout_s": timeout_s,
    }


# ---------------------------------------------------------------------------
# R process management
# ---------------------------------------------------------------------------


def run_r_process(
    scenarios_json_path: str | Path,
    output_json_path: str | Path,
    timeout: int = 600,
) -> dict[str, Any]:
    """Launch ``benchmark_r.R`` and return the parsed results JSON.

    On timeout, attempts to read partial results from the output file.
    The R script writes JSON after each scenario completes, so partial
    data is available even if the full batch didn't finish.

    Args:
        scenarios_json_path: Path to the handoff JSON file.
        output_json_path: Path where R will write results.
        timeout: Global wall-clock timeout for the entire R process.

    Returns:
        Parsed results dict with keys ``r_version``, ``harmonizr_version``,
        ``startup_s``, ``results`` (list of per-scenario dicts).
        On timeout, ``results`` contains only completed scenarios.

    Raises:
        RuntimeError: If Rscript is not available or the process fails.
    """
    if not r_available():
        raise RuntimeError("Rscript not found on PATH")

    if not _R_SCRIPT.is_file():
        raise FileNotFoundError(f"R script not found: {_R_SCRIPT}")

    cmd = [
        "Rscript", str(_R_SCRIPT),
        "--scenarios", str(scenarios_json_path),
        "--output", str(output_json_path),
    ]

    logger.info("Launching R process: %s", " ".join(cmd))

    # Use Popen with process_group=0 so R and all its doParallel workers
    # are in one process group that we can kill atomically.
    timed_out = False
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        process_group=0,
    )

    try:
        _, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        logger.warning(
            "R process timed out after %ds. Killing process group...",
            timeout,
        )
        # Kill the entire process group (R + all doParallel workers)
        try:
            os.killpg(process.pid, signal.SIGTERM)
            time.sleep(2)
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        _stdout, stderr = process.communicate()

    # Try to read output even on timeout (partial JSON)
    output_path = Path(output_json_path)
    if output_path.is_file():
        with output_path.open() as fh:
            partial = cast("dict[str, Any]", json.load(fh))
        if timed_out:
            n_done = len(partial.get("results", []))
            n_total = 0
            try:
                with Path(scenarios_json_path).open() as sfh:
                    n_total = len(json.load(sfh))
            except Exception:
                pass
            logger.warning(
                "Partial R results: %d/%d scenarios completed.",
                n_done, n_total,
            )
        return partial

    # No output file at all: real failure
    if timed_out:
        raise RuntimeError(
            f"R process timed out after {timeout}s and produced no output."
        )
    if process.returncode != 0:
        stderr_tail = (stderr or "").strip().split("\n")[-5:]
        raise RuntimeError(
            f"R process failed (exit {process.returncode}): "
            f"{' | '.join(stderr_tail)}"
        )

    # Output file missing despite clean exit
    raise RuntimeError(
        f"R process completed (exit {process.returncode}) but "
        f"no output found at {output_json_path}; "
        f"stderr: {(stderr or '')[-200:]}"
    )


# ---------------------------------------------------------------------------
# High-level operations
# ---------------------------------------------------------------------------


def cache_r_scenarios(
    scenarios: list[Scenario],
    config: Config,
    cache_dir: str | Path,
    tmp_dir: str | Path,
    cores: int | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Run R on scenarios and populate the cache.

    This is the core of the ``bench.py cache-r`` command.

    Args:
        scenarios: List of scenarios to cache.
        config: Loaded benchmark config.
        cache_dir: Cache directory path.
        tmp_dir: Temp directory for intermediate files.
        cores: Number of R cores to use.  Defaults to config's
            ``r_cache.default_cores``.
        force: If ``True``, re-run even if cache entry exists.

    Returns:
        Dict mapping scenario id to status string:
        ``"cached"``, ``"hit"``, or ``"error: <msg>"``.
    """
    if cores is None:
        cores = config.r_cache_default_cores

    harmonizr_version = get_harmonizr_version()
    r_version = get_r_version()
    cache_dir = Path(cache_dir)
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Separate scenarios that need R runs from cache hits
    to_run: list[Scenario] = []
    statuses: dict[str, str] = {}

    for s in scenarios:
        from ..datasets import resolve_dataset_paths
        paths = resolve_dataset_paths(config, s.dataset)
        if not paths.r_eligible:
            statuses[s.id] = "skipped (r_eligible=false)"
            continue

        key = compute_cache_key(
            s, paths.input_path, paths.desc_path,
            harmonizr_version, cores,
        )
        if cache_hit(cache_dir, key) and not force:
            statuses[s.id] = "hit"
        else:
            to_run.append(s)

    if not to_run:
        logger.info("All %d scenarios already cached.", len(scenarios))
        return statuses

    # Build handoff JSON for R
    from ..datasets import resolve_dataset_paths as resolve_paths
    scenario_entries = []
    for s in to_run:
        paths = resolve_paths(config, s.dataset)
        output_tsv = tmp_dir / f"{s.id}_c{cores}_r.tsv"
        entry = build_scenario_entry(
            s, paths.input_path, paths.desc_path,
            output_tsv, cores, config.r_cache_timeout_s,
            tmp_dir=tmp_dir, dataset_name=s.dataset,
        )
        scenario_entries.append(entry)

    scenarios_json = tmp_dir / "r_scenarios.json"
    with scenarios_json.open("w") as fh:
        json.dump(scenario_entries, fh, indent=2)

    results_json = tmp_dir / "r_results.json"
    if results_json.exists():
        results_json.unlink()

    # Run R with a global timeout proportional to scenario count
    global_timeout = max(600, len(to_run) * config.r_cache_timeout_s + 120)
    r_result = run_r_process(scenarios_json, results_json, timeout=global_timeout)

    # Populate cache from results
    startup_s = cast("float", r_result.get("startup_s", 0.0))
    session_rss_kb = cast("int | None", r_result.get("rss_kb"))
    session_rss_peak_kb = cast("int | None", r_result.get("rss_peak_kb"))
    results_list = cast("list[dict[str, Any]]", r_result.get("results", []))
    for entry in results_list:
        sid = cast("str", entry["id"])
        # Propagate session-level metrics into each per-scenario entry
        if session_rss_kb is not None:
            entry["rss_kb"] = session_rss_kb
        if session_rss_peak_kb is not None:
            entry["rss_peak_kb"] = session_rss_peak_kb
        # Find the matching scenario
        matching = [s for s in to_run if s.id == sid]
        if not matching:
            statuses[sid] = "error: no matching scenario"
            continue
        scenario = matching[0]

        paths = resolve_paths(config, scenario.dataset)
        key = compute_cache_key(
            scenario, paths.input_path, paths.desc_path,
            harmonizr_version, cores,
        )

        err = cast("str | None", entry.get("error"))
        if err and err != "NA":
            statuses[sid] = f"error: {err}"
            # Still cache the error result to avoid re-runs
            _write_cache_entry(
                cache_dir, key, entry,
                harmonizr_version, sid,
                paths.input_path, paths.desc_path,
                cores, r_version, startup_s,
            )
            continue

        _write_cache_entry(
            cache_dir, key, entry,
            harmonizr_version, sid,
            paths.input_path, paths.desc_path,
            cores, r_version, startup_s,
        )
        statuses[sid] = "cached"

    return statuses


def resolve_r_results(
    scenarios: list[Scenario],
    config: Config,
    cache_dir: str | Path,
    cores: int | None = None,
) -> dict[str, dict[str, Any] | None]:
    """Resolve R results for a list of scenarios from cache.

    This is the core of the ``bench.py run --with-r`` path.

    Returns a dict mapping scenario id to either the cached result dict
    or ``None`` if no cache entry exists.
    """
    if cores is None:
        cores = config.r_cache_default_cores

    harmonizr_version = get_harmonizr_version()
    cache_dir = Path(cache_dir)

    results: dict[str, dict[str, Any] | None] = {}
    for s in scenarios:
        from ..datasets import resolve_dataset_paths
        paths = resolve_dataset_paths(config, s.dataset)
        if not paths.r_eligible:
            results[s.id] = None
            continue

        key = compute_cache_key(
            s, paths.input_path, paths.desc_path,
            harmonizr_version, cores,
        )
        entry = read_cache_entry(cache_dir, key)
        results[s.id] = entry

    return results
