"""Benchmark runner for HarmonizePy.

Orchestrates the full benchmark matrix: generates datasets, runs
HarmonizePy for each combination, optionally runs R HarmonizR
for comparison, collects timings, peak memory, and feature retention,
and produces a JSON results file and benchmarks/RESULTS.md.

Usage::

    # Full benchmark (Python only)
    python benchmarks/run_benchmarks.py

    # Full benchmark with R comparison
    python benchmarks/run_benchmarks.py --with-r

    # Single dataset
    python benchmarks/run_benchmarks.py --dataset medium
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
_HARMONIZEPY_CLI: list[str] = [str(_REPO_ROOT / ".venv" / "bin" / "harmonizepy")]
if not Path(_HARMONIZEPY_CLI[0]).exists():
    _HARMONIZEPY_CLI = [sys.executable, "-m", "harmonizepy"]

_GENERATE_SCRIPT = Path(__file__).parent / "generate_data.py"
_TEMPLATE_R = Path(__file__).parent / "template_run.R"
_DATA_DIR = Path(__file__).parent / "data"
_RESULTS_DIR = Path(__file__).parent / "results"
_RESULTS_MD = Path(__file__).parent / "RESULTS.md"

_DATASETS = ["small", "medium", "large"]
_R_AVAILABLE = shutil.which("Rscript") is not None

_BENCHMARKS: list[tuple[str, str, int | None, int | None, str | None]] = [
    *[(ds, "ComBat", m, None, None) for ds in _DATASETS for m in [1, 2, 3, 4]],
    *[(ds, "limma", None, None, None) for ds in _DATASETS],
    *[(ds, "ComBat", 1, 2, None) for ds in _DATASETS],
    *[(ds, "ComBat", 1, 2, "sparsity") for ds in ["medium", "large"]],
]

_DS_FEATURES: dict[str, int] = {"small": 1000, "medium": 5000, "large": 10000}


@dataclass
class RunResult:
    dataset: str
    algorithm: str
    combat_mode: int | None
    block: int | None
    sort: str | None
    python_time_s: float | None
    python_memory_mb: float | None
    features_out: int | None
    features_pt: int | None  # passed through without correction
    features_corrected: int | None
    r_time_s: float | None
    max_rel_diff: float | None


def _generate_datasets(datasets: list[str]) -> None:
    for ds in datasets:
        data_file = _DATA_DIR / f"{ds}_input.tsv"
        if not data_file.exists():
            subprocess.run(
                [sys.executable, str(_GENERATE_SCRIPT), "--dataset", ds],
                check=True,
            )


def _run_python_benchmark(
    data_path: str,
    desc_path: str,
    dataset: str,
    output_path: str,
    algo: str,
    mode: int | None,
    block: int | None,
    sort: str | None,
) -> tuple[float, float, int, int]:
    """Run harmonizepy. Returns (time_s, memory_mb, features_out).

    Memory is measured via ``/usr/bin/time -v`` max resident set size
    when available; falls back to 0.0.
    """
    cmd = [
        *_HARMONIZEPY_CLI,
        data_path,
        desc_path,
        "-o",
        output_path,
        "--algorithm",
        algo,
        "--no-log",
    ]
    if algo == "ComBat" and mode is not None:
        cmd.extend(["--combat-mode", str(mode)])
    if block is not None:
        cmd.extend(["--block", str(block)])
    if sort is not None:
        cmd.extend(["--sort", sort])

    _time_cmd = shutil.which("time")
    use_time_cmd = _time_cmd and "/usr/bin/time" in _time_cmd

    if use_time_cmd:
        wrapped = ["/usr/bin/time", "-v", *cmd]
    else:
        wrapped = cmd

    start = time.monotonic()
    result = subprocess.run(wrapped, capture_output=True, text=True)
    py_time = time.monotonic() - start

    if result.returncode != 0:
        raise RuntimeError(f"harmonizepy failed (exit {result.returncode}): {result.stderr}")

    memory_mb = 0.0
    if use_time_cmd:
        for line in (result.stderr or "").split("\n"):
            if "Maximum resident set size" in line:
                try:
                    memory_mb = int(line.split(":")[-1].strip()) / 1024.0
                except (ValueError, IndexError):
                    pass

    # Parse pipeline stats from stderr
    features_in = _DS_FEATURES.get(dataset, 0)
    features_pt = 0
    for line in (result.stderr or "").split("\n"):
        if "Input:" in line:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    features_in = int(parts[3])
                except (ValueError, IndexError):
                    pass
        if "passed through without correction" in line:
            # "INFO [harmonizepy] N feature(s) passed through..."
            parts = line.split()
            if len(parts) >= 4:
                try:
                    features_pt = int(parts[2])
                except (ValueError, IndexError):
                    pass

    features_out = features_in
    return py_time, memory_mb, features_out, features_pt


def _run_r_benchmark(
    data_path: str,
    desc_path: str,
    output_path: str,
    algo: str,
    mode: int | None,
    block: int | None,
    sort: str | None,
) -> float:
    """Run R HarmonizR. Returns wall-clock time in seconds."""
    r_sort = f"{sort}_sort" if sort else "NA"
    r_block = str(block) if block is not None else "NA"
    r_mode = str(mode) if mode is not None else "NA"

    cmd = [
        "Rscript",
        str(_TEMPLATE_R),
        data_path,
        desc_path,
        output_path,
        algo,
        r_mode,
        r_block,
        r_sort,
    ]

    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    r_time = time.monotonic() - start

    if proc.returncode != 0 or not Path(output_path).exists():
        err_tail = (proc.stderr or "").strip().split("\n")[-1] if proc.stderr else ""
        raise RuntimeError(f"R HarmonizR failed (exit {proc.returncode}): {err_tail}")

    for line in proc.stdout.split("\n"):
        if line.startswith("R_TIME:"):
            r_time = float(line.split()[1])
            break

    return r_time


def _compare_outputs(py_tsv: str, r_tsv: str) -> float:
    """Compare Python and R corrected outputs. Returns max relative difference."""
    py_df = pd.read_csv(py_tsv, sep="\t", index_col=0)
    r_df = pd.read_csv(r_tsv, sep="\t", index_col=0)

    common_idx = py_df.index.intersection(r_df.index)
    common_cols = py_df.columns.intersection(r_df.columns)
    p = py_df.loc[common_idx, common_cols].to_numpy(dtype=np.float64)
    r = r_df.loc[common_idx, common_cols].to_numpy(dtype=np.float64)

    mask = ~(np.isnan(p) | np.isnan(r))
    if not mask.any():
        return 0.0

    rel_diff = np.abs(p[mask] - r[mask]) / np.maximum(np.abs(r[mask]), 1e-12)
    return float(rel_diff.max())


def _generate_results_md(results: list[RunResult]) -> str:
    """Generate benchmarks/RESULTS.md."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    sys_info = f"{platform.system()} {platform.release()}, Python {platform.python_version()}"
    cpu_info = platform.processor() or "unknown"

    lines = [
        "# Benchmark Results",
        "",
        f"**Generated:** {now}",
        f"**Platform:** {sys_info}",
        f"**CPU:** {cpu_info}",
        f"**R available:** {_R_AVAILABLE}",
        "",
    ]

    py_results = [r for r in results if r.python_time_s is not None]
    if py_results:
        lines.extend(
            [
                "## Python Performance",
                "",
                "| Dataset | Algorithm | Mode | Block | Sort | Time (s) | Memory (MB) | Features out | Corrected | Pass-through |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for r in py_results:
            mode_s = str(r.combat_mode) if r.combat_mode is not None else "--"
            block_s = str(r.block) if r.block is not None else "--"
            sort_s = r.sort if r.sort is not None else "--"
            pt_str = str(r.features_pt) if r.features_pt is not None else "--"
            cor_str = str(r.features_corrected) if r.features_corrected is not None else "--"
            lines.append(
                f"| {r.dataset} | {r.algorithm} | {mode_s} | {block_s} | {sort_s} | "
                f"{r.python_time_s:.3f} | {r.python_memory_mb:.1f} | {r.features_out} | "
                f"{cor_str} | {pt_str} |"
            )
        lines.append("")

    r_results = [r for r in results if r.r_time_s is not None]
    if r_results:
        lines.extend(
            [
                "## R HarmonizR Performance",
                "",
                "| Dataset | Algorithm | Mode | Block | Sort | R Time (s) |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for r in r_results:
            mode_s = str(r.combat_mode) if r.combat_mode is not None else "--"
            block_s = str(r.block) if r.block is not None else "--"
            sort_s = r.sort if r.sort is not None else "--"
            lines.append(
                f"| {r.dataset} | {r.algorithm} | {mode_s} | {block_s} | {sort_s} | "
                f"{r.r_time_s:.3f} |"
            )
        lines.append("")

    concordant = [r for r in results if r.max_rel_diff is not None]
    if concordant:
        lines.extend(
            [
                "## Python vs R Concordance",
                "",
                "| Dataset | Algorithm | Mode | Max relative diff |",
                "| --- | --- | --- | --- |",
            ]
        )
        for r in concordant:
            mode_s = str(r.combat_mode) if r.combat_mode is not None else "--"
            diff_str = f"{r.max_rel_diff:.2e}" if r.max_rel_diff is not None else "N/A"
            lines.append(f"| {r.dataset} | {r.algorithm} | {mode_s} | {diff_str} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmark suite.")
    parser.add_argument(
        "--dataset", default=None, choices=[*_DATASETS, None], help="Run a single dataset only."
    )
    parser.add_argument(
        "--with-r",
        action="store_true",
        help="Run R HarmonizR benchmarks for comparison (requires R).",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else _DATASETS

    if args.with_r and not _R_AVAILABLE:
        print("ERROR: --with-r requested but Rscript not found.", file=sys.stderr)
        sys.exit(1)

    if args.with_r:
        print("R available: running R comparisons.")
    else:
        print("R not available or not requested. Python only.")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating datasets...")
    _generate_datasets(datasets)

    benchmarks = [b for b in _BENCHMARKS if b[0] in datasets]

    results: list[RunResult] = []

    for ds, algo, mode, block, sort in benchmarks:
        data_path = str(_DATA_DIR / f"{ds}_input.tsv")
        desc_path = str(_DATA_DIR / f"{ds}_batch.csv")

        label = f"{ds}/{algo}"
        if mode:
            label += f"/mode{mode}"
        if block:
            label += f"/block{block}"
        if sort:
            label += f"/{sort}"

        print(f"  {label}...", end=" ", flush=True)

        tag = f"{ds}_{algo}"
        if mode is not None:
            tag += f"_m{mode}"
        if block is not None:
            tag += f"_b{block}"
        if sort is not None:
            tag += f"_{sort}"
        py_out = str(_RESULTS_DIR / f"{tag}_py.tsv")
        try:
            py_time, memory_mb, features_out, features_pt = _run_python_benchmark(
                data_path,
                desc_path,
                ds,
                py_out,
                algo,
                mode,
                block,
                sort,
            )
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"PY ERR ({exc})", flush=True)
            results.append(
                RunResult(
                    ds,
                    algo,
                    mode,
                    block,
                    sort,
                    python_time_s=None,
                    python_memory_mb=None,
                    features_out=None,
                    features_pt=None,
                    features_corrected=None,
                    r_time_s=None,
                    max_rel_diff=None,
                )
            )
            continue

        r_time = None
        max_rel_diff = None

        if args.with_r:
            r_out = str(_RESULTS_DIR / f"{tag}_r.tsv")
            try:
                r_time = _run_r_benchmark(
                    data_path,
                    desc_path,
                    r_out,
                    algo,
                    mode,
                    block,
                    sort,
                )
                max_rel_diff = _compare_outputs(py_out, r_out)
            except (RuntimeError, FileNotFoundError) as exc:
                print(f"R ERR ({exc})", end=" ", flush=True)

        print(f"OK ({py_time:.3f}s, {memory_mb:.1f}MB)", flush=True)
        results.append(
            RunResult(
                ds,
                algo,
                mode,
                block,
                sort,
                python_time_s=py_time,
                python_memory_mb=memory_mb,
                features_out=features_out,
                features_pt=features_pt,
                features_corrected=features_out - features_pt,
                r_time_s=r_time,
                max_rel_diff=max_rel_diff,
            )
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    results_file = _RESULTS_DIR / f"benchmark_{timestamp}.json"
    with Path(results_file).open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    md = _generate_results_md(results)
    with _RESULTS_MD.open("w") as f:
        f.write(md)
    print(f"Summary written to {_RESULTS_MD}")


if __name__ == "__main__":
    main()
