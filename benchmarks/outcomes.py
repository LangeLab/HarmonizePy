"""What each tool returns for every input feature, on the datasets with an R baseline.

    uv run python -m benchmarks.outcomes

reads the parity pass at ``benchmarks/results/benchmark_r_parity.json`` and
writes ``benchmarks/results/outcomes.json``.

For ComBat mode 1, the default, each input feature is put in one class per tool:

- ``corrected``: returned with values that differ from the input;
- ``raw``: returned with its input values unchanged (HarmonizePy reports why);
- ``empty``: returned as a row with no value;
- ``dropped``: not in the output at all.

HarmonizePy's classes come from its correction report. HarmonizR's come from
the cached output of the parity pass: a returned row is ``raw`` when every value
it has equals the input's in the same cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from harmonizepy import harmonize

from .datasets import load_dataset, resolve_dataset_paths
from .scenarios import load_config

CLASSES = ("corrected", "raw", "empty", "dropped")


def _harmonizr_classes(data: pd.DataFrame, r_out: pd.DataFrame) -> dict[str, int]:
    shared = data.index.intersection(r_out.index)
    cols = data.columns.intersection(r_out.columns)
    given = data.loc[shared, cols].to_numpy(dtype=np.float64)
    got = r_out.loc[shared, cols].to_numpy(dtype=np.float64)
    empty = np.isnan(got).all(axis=1)
    both = ~np.isnan(got) & ~np.isnan(given)
    diff = np.where(both, np.abs(got - given), 0.0)
    tolerance = 1e-9 * np.maximum(np.abs(np.nan_to_num(given)), 1.0)
    unchanged = (diff <= tolerance).all(axis=1)
    raw = ~empty & unchanged
    return {
        "corrected": int((~empty & ~unchanged).sum()),
        "raw": int(raw.sum()),
        "empty": int(empty.sum()),
        "dropped": len(data.index.difference(r_out.index)),
    }


def run(parity_path: str, config_path: str) -> dict[str, Any]:
    parity = json.loads(Path(parity_path).read_text())
    cfg = load_config(config_path)
    rows = []
    for e in parity["results"]:
        s = e["scenario"]
        if (
            s["algorithm"] != "ComBat"
            or s.get("combat_mode") != 1
            or s.get("block")
            or not e.get("r")
        ):
            continue
        data, desc = load_dataset(resolve_dataset_paths(cfg, s["dataset"]))
        _, report = harmonize(data, desc, return_report=True)
        status = report["status"]
        py = {
            "corrected": int((status == "corrected").sum()),
            "raw": int(status.str.startswith("raw_").sum()),
            "empty": int((status == "missing_no_usable_batch").sum()),
            "dropped": 0,
        }
        raw_reasons = {
            k: int(v) for k, v in status[status.str.startswith("raw_")].value_counts().items()
        }
        r_out = pd.read_csv(e["r"]["output_tsv"], sep="\t", index_col=0)
        rows.append(
            {
                "dataset": s["dataset"],
                "n_features": len(data),
                "harmonizepy": py,
                "harmonizepy_raw_reasons": raw_reasons,
                "harmonizr": _harmonizr_classes(data, r_out),
            }
        )
        print(f"{s['dataset']}: done", flush=True)
    return {"combat_mode": 1, "classes": list(CLASSES), "datasets": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmarks.outcomes", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--parity",
        default="benchmarks/results/benchmark_r_parity.json",
        help="JSON of the R-backed parity pass.",
    )
    parser.add_argument("--config", default="benchmarks/config.yaml", help="Benchmark config.")
    parser.add_argument(
        "--out", default="benchmarks/results/outcomes.json", help="Output JSON path."
    )
    args = parser.parse_args(argv)
    report = run(args.parity, args.config)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"Results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
