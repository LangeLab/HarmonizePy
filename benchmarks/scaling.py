"""Scaling sweep: wall time and peak memory of ``harmonize()`` against matrix size.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python -m benchmarks.scaling

Synthetic matrices from ``generate_dataset`` (per-batch missingness, 20 percent
of features absent from each batch, ten batches) are corrected with limma and
the four ComBat modes:

- a feature sweep at 60 samples;
- a sample sweep at 5000 features;
- a blocking pair, without and with ``block=2``, which also records the share
  of observed input values that reach the output.

Each point is timed with the harness's ``run_once``, so wall time and peak
memory are measured as in the main benchmark; the reported time is the median
of the repetitions. Only Python runs here: R baselines come from the parity
pass of ``benchmarks.bench``. Results go to ``benchmarks/results/scaling.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import harmonizepy
from harmonizepy.core import plan_affiliations, prepare_inputs

from .datasets import generate_dataset
from .runners.python_runner import run_once
from .scenarios import Scenario

FEATURES = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000)
SAMPLES = (40, 100, 200, 600, 2_000, 6_000)
QUICK_FEATURES = (1_000, 2_000)
QUICK_SAMPLES = (40, 100)
ENGINES: tuple[tuple[str, int | None], ...] = (
    ("limma", None),
    ("ComBat", 1),
    ("ComBat", 2),
    ("ComBat", 3),
    ("ComBat", 4),
)
N_BATCHES = 10
MISSING_FRAC = 0.20
# Points slower than this get one repetition instead of three.
SLOW_S = 20.0


def _measure(
    data: pd.DataFrame, desc: pd.DataFrame, algorithm: str, mode: int | None, block: int | None
) -> dict[str, Any]:
    """Time one configuration and return its record, keeping the output for one count."""
    scenario = Scenario(dataset="sweep", algorithm=algorithm, combat_mode=mode, block=block)
    first = run_once(data, desc, scenario, keep_result_df=True)
    result = first.result["result_df"]
    observed_out = int(result.notna().to_numpy().sum())
    first.release_result()
    runs = [first]
    if first.elapsed_s < SLOW_S:
        runs += [run_once(data, desc, scenario, keep_result_df=False) for _ in range(2)]
    return {
        **_groups(data, desc, algorithm, mode, block),
        "algorithm": algorithm,
        "mode": mode,
        "block": block,
        "n_features": data.shape[0],
        "n_samples": data.shape[1],
        "n_batches": N_BATCHES,
        "reps": len(runs),
        "median_s": statistics.median(r.elapsed_s for r in runs),
        "peak_mb": max(r.tracemalloc_peak_mb for r in runs),
        "input_mb": data.to_numpy().nbytes / 2**20,
        "observed_in": int(data.notna().to_numpy().sum()),
        "observed_out": observed_out,
    }


def _groups(
    data: pd.DataFrame, desc: pd.DataFrame, algorithm: str, mode: int | None, block: int | None
) -> dict[str, int]:
    """Count the affiliation groups a run forms, and those big enough for an engine."""
    inputs = prepare_inputs(
        data,
        desc,
        algorithm=algorithm,
        combat_mode=mode or 1,
        needed_values=None,
        sort=None,
        block=block,
        unique_removal=True,
    )
    plan = plan_affiliations(
        inputs.data,
        inputs.batch_list,
        sort=None,
        block=block,
        needed_values=inputs.needed_values,
        unique_removal=True,
    )
    sizes = pd.Series([a for a in plan.affiliation_list if a]).value_counts()
    return {"n_groups": int(sizes.size), "n_groups_fitted": int((sizes >= 2).sum())}


def _matrix(n_features: int, n_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate_dataset(
        "sweep", n_features, n_samples, N_BATCHES, MISSING_FRAC, missing_mode="per_batch"
    )


def run_sweep(*, quick: bool = False) -> dict[str, Any]:
    """Run the three sweeps and return them with the environment they ran in."""
    records: dict[str, list[dict[str, Any]]] = {"features": [], "samples": [], "blocking": []}
    for n_features in QUICK_FEATURES if quick else FEATURES:
        data, desc = _matrix(n_features, 60)
        for algorithm, mode in ENGINES:
            records["features"].append(_measure(data, desc, algorithm, mode, None))
            print(f"features {n_features}: {algorithm} {mode or ''} done", flush=True)
    for n_samples in QUICK_SAMPLES if quick else SAMPLES:
        data, desc = _matrix(5_000, n_samples)
        for algorithm, mode in ENGINES:
            records["samples"].append(_measure(data, desc, algorithm, mode, None))
            print(f"samples {n_samples}: {algorithm} {mode or ''} done", flush=True)
    for n_features in QUICK_FEATURES[:1] if quick else (5_000, 20_000):
        data, desc = _matrix(n_features, 60)
        for algorithm, mode in (("limma", None), ("ComBat", 1), ("ComBat", 3)):
            for block in (None, 2):
                records["blocking"].append(_measure(data, desc, algorithm, mode, block))
        print(f"blocking {n_features} done", flush=True)
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": {
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "harmonizepy": harmonizepy.__version__,
            "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS")
            or os.environ.get("OMP_NUM_THREADS")
            or "unset",
        },
        "design": {
            "n_batches": N_BATCHES,
            "missing_frac": MISSING_FRAC,
            "missing_mode": "per_batch",
            "feature_sweep_samples": 60,
            "sample_sweep_features": 5_000,
        },
        **records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.scaling", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", default="benchmarks/results/scaling.json", help="Output JSON path."
    )
    parser.add_argument(
        "--quick", action="store_true", help="Two small sizes per sweep, for a smoke run."
    )
    args = parser.parse_args(argv)
    report = run_sweep(quick=args.quick)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
