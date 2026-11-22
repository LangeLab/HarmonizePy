"""Peak memory of HarmonizePy and HarmonizR, each in a fresh process, on the same files.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python -m benchmarks.memory_peak

Every run starts a new process that reads the data matrix and the batch
description from disk, corrects it once, and exits; the measurement is that
process's peak resident set size (``ru_maxrss`` from ``os.wait4``), the memory
the operating system had to give it. Both tools read the same files, so both
peaks include parsing them. A baseline run per tool loads the interpreter and
the package and stops, which gives the fixed cost of starting the tool.

HarmonizR runs through ``Rscript`` in the project's renv library, on one
core, with BiocParallel set to serial so that sva's per-batch work stays in the
measured process. Results go to ``benchmarks/results/memory_peak.json``; a run
keeps the measurements already there and adds the missing ones.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .datasets import resolve_dataset_paths
from .runners.r_runner import _ensure_tsv
from .scenarios import load_config

DATASETS = ("small", "murine", "medium", "dia", "scp_small")
ENGINES: tuple[tuple[str, int | None], ...] = (("limma", None), ("ComBat", 1), ("ComBat", 3))

_PY = """
import sys
from harmonizepy import harmonize
if sys.argv[1] != "baseline":
    harmonize(sys.argv[2], sys.argv[3], algorithm=sys.argv[1], combat_mode=int(sys.argv[4]))
"""

_R = """
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages({library(HarmonizR); library(BiocParallel)})
# sva::ComBat would otherwise take BiocParallel's default multicore workers, whose memory
# the parent process does not report; serial keeps the whole run in one process.
register(SerialParam())
if (args[1] != "baseline") {
  invisible(harmonizR(args[2], args[3], algorithm = args[1], ComBat_mode = as.integer(args[4]),
                      ur = TRUE, plot = FALSE, output_file = FALSE, verbosity = 0, cores = 1))
}
"""


def peak_rss_mb(cmd: list[str], timeout_s: float) -> tuple[float, float]:
    """Run *cmd* to completion and return its peak RSS in MB and its wall time in seconds.

    The child's stderr goes to a temporary file, not a pipe: R and sva write
    progress there, and an unread pipe would fill and block the child.
    """
    with tempfile.TemporaryFile() as err:
        start = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err)
        deadline = start + timeout_s
        while True:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                break
            if time.perf_counter() > deadline:
                proc.kill()
                os.wait4(proc.pid, 0)
                raise TimeoutError(" ".join(cmd[:2]))
            time.sleep(0.05)
        wall = time.perf_counter() - start
        if os.waitstatus_to_exitcode(status) != 0:
            err.seek(0)
            raise RuntimeError(f"{cmd[0]} failed: {err.read().decode()[-400:]}")
    # ru_maxrss is in kilobytes on Linux.
    return usage.ru_maxrss / 1024, wall


def run(config_path: str, timeout_s: float, out: Path) -> dict[str, Any]:
    """Measure every dataset and engine, saving after each one; measurements already in *out* are kept."""
    cfg = load_config(config_path)
    tools = {
        "harmonizepy": lambda alg, mode, data, desc: [
            sys.executable,
            "-c",
            _PY,
            alg,
            data,
            desc,
            str(mode),
        ],
        "harmonizr": lambda alg, mode, data, desc: [
            "Rscript",
            "-e",
            _R,
            alg,
            data,
            desc,
            str(mode),
        ],
    }
    report: dict[str, Any] = json.loads(out.read_text()) if out.is_file() else {}
    if "baseline_mb" not in report:
        report["baseline_mb"] = {
            name: peak_rss_mb(make("baseline", 1, "", ""), timeout_s)[0]
            for name, make in tools.items()
        }
    rows: list[dict[str, Any]] = report.setdefault("results", [])
    done = {(r["dataset"], r["algorithm"], r["mode"]) for r in rows}
    for ds in DATASETS:
        paths = resolve_dataset_paths(cfg, ds)
        # HarmonizR reads tab-separated text only; both tools get the same file.
        data = _ensure_tsv(paths.input_path, cfg.paths_tmp_dir, ds)
        for algorithm, mode in ENGINES:
            if (ds, algorithm, mode) in done:
                continue
            row: dict[str, Any] = {"dataset": ds, "algorithm": algorithm, "mode": mode}
            for name, make in tools.items():
                peak, wall = peak_rss_mb(
                    make(algorithm, mode or 1, data, str(paths.desc_path)), timeout_s
                )
                row[name] = {"peak_mb": peak, "wall_s": wall}
            rows.append(row)
            report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            out.write_text(json.dumps(report, indent=2))
            print(
                f"{ds} {algorithm} {mode or ''}: py {row['harmonizepy']['peak_mb']:.0f} MB, "
                f"R {row['harmonizr']['peak_mb']:.0f} MB",
                flush=True,
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmarks.memory_peak", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--config", default="benchmarks/config.yaml", help="Benchmark config.")
    parser.add_argument("--timeout", type=float, default=1800, help="Seconds allowed per run.")
    parser.add_argument(
        "--out", default="benchmarks/results/memory_peak.json", help="Output JSON path."
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run(args.config, args.timeout, out)
    print(f"Results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
