#!/usr/bin/env python3
"""HarmonizePy benchmark suite: CLI entry point.

Usage::

    python benchmarks/bench.py run
    python benchmarks/bench.py cache-r
    python benchmarks/bench.py validity
    python benchmarks/bench.py profile --dataset medium --algorithm ComBat --combat-mode 3
    python benchmarks/bench.py report --input results/benchmark.json --out report.md
    python benchmarks/bench.py generate-data
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .harness import BenchmarkHarness
from .report import build_report_json, generate_markdown, read_report_json, write_report_json
from .scenarios import build_registry, filter_registry, load_config

logger = logging.getLogger(__name__)

_BENCHMARKS_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _BENCHMARKS_DIR / "config.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _resolve_config(config_path: str | None) -> Path:
    if config_path:
        return Path(config_path)
    return _DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s [bench] %(message)s",
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------


def _add_run_parser(subparsers: Any) -> None:
    p = subparsers.add_parser("run", help="Run benchmark scenarios with timing.")
    p.add_argument("--datasets", nargs="*", default=None, help="Datasets to run (default: all).")
    p.add_argument("--algorithms", nargs="*", default=None, choices=["ComBat", "limma"], help="Filter by algorithm.")
    p.add_argument("--combat-modes", nargs="*", type=int, default=None, choices=[1, 2, 3, 4], help="Filter by ComBat mode.")
    p.add_argument("--tags", nargs="*", default=None, help="Filter by scenario tags (e.g., real scp).")
    p.add_argument("--budget-s", type=int, default=None, help="Per-scenario time budget (default: 30).")
    p.add_argument("--n-reps", type=int, default=None, metavar="N", help="Fixed repetition count (overrides adaptive).")
    p.add_argument("--min-reps", type=int, default=None, help="Minimum repetitions (default: 3, ignored with --n-reps).")
    p.add_argument("--max-reps", type=int, default=None, help="Maximum repetitions (default: 10, ignored with --n-reps).")
    p.add_argument("--no-warmup", action="store_true", help="Skip warmup validity run.")
    p.add_argument("--with-r", action="store_true", help="Attach R baseline from cache.")
    p.add_argument("--with-r-cores", type=int, default=None, help="R cores for cache lookup (default: 16).")
    p.add_argument("--force-r", action="store_true", help="Bypass cache, run R live and update cache.")
    p.add_argument("--out", default=None, metavar="PATH", help="Output JSON path.")
    p.add_argument("--md", default=None, metavar="PATH", help="Output Markdown path (default: next to JSON).")
    p.add_argument("--config", default=None, help="Config file path.")


def _cmd_run(args: argparse.Namespace) -> int:
    _setup_logging(verbose=False)
    cfg_path = _resolve_config(args.config)
    cfg = load_config(cfg_path)

    registry = build_registry(cfg)
    scenarios = filter_registry(
        registry,
        datasets=args.datasets,
        algorithms=args.algorithms,
        combat_modes=args.combat_modes,
        tags=args.tags,
    )

    if not scenarios:
        print("No scenarios match the given filters.", file=sys.stderr)
        return 1

    print(f"Running {len(scenarios)} scenario(s)...")

    # Clear tmp at start
    tmp_dir = Path(cfg.paths_tmp_dir)
    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Resolve repetition and warmup parameters
    min_reps = cfg.python_min_reps
    max_reps = cfg.python_max_reps
    use_warmup = cfg.python_warmup
    if args.n_reps is not None:
        min_reps = max_reps = args.n_reps
    else:
        if args.min_reps is not None:
            min_reps = args.min_reps
        if args.max_reps is not None:
            max_reps = args.max_reps
    if args.no_warmup:
        use_warmup = False

    harness = BenchmarkHarness(
        cfg, scenarios,
        budget_s=args.budget_s,
        min_reps=min_reps,
        max_reps=max_reps,
        warmup=use_warmup,
    )
    results = harness.run_all()

    # Build report JSON
    r_startup: float | None = None
    r_ver: str | None = None
    h_ver: str | None = None
    if args.with_r or args.force_r:
        try:
            from .runners.r_runner import (
                cache_r_scenarios,
                get_harmonizr_version,
                get_r_version,
                resolve_r_results,
            )
            r_ver = get_r_version()
            h_ver = get_harmonizr_version()

            if args.force_r:
                print("Running R scenarios live (--force-r)...")
                statuses = cache_r_scenarios(
                    scenarios, cfg,
                    cache_dir=cfg.paths_cache_dir,
                    tmp_dir=cfg.paths_tmp_dir,
                    cores=args.with_r_cores,
                    force=True,
                )
                misses = sum(1 for s in statuses.values() if s.startswith("error"))
                if misses:
                    print(f"  {misses} R scenario(s) failed.", file=sys.stderr)

            r_entries = resolve_r_results(
                scenarios, cfg, cfg.paths_cache_dir,
                cores=args.with_r_cores,
            )
            # Merge resolved results back into harness results
            merged = []
            miss_count = 0
            for scenario, s_metrics, validity, _ in results:
                r_entry = r_entries.get(scenario.id)
                if r_entry is None:
                    miss_count += 1
                merged.append((scenario, s_metrics, validity, r_entry))
            results = merged
            if miss_count:
                print(
                    f"Warning: R cache miss for {miss_count}/{len(scenarios)} "
                    f"scenario(s). Use --force-r to populate cache.",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"Warning: R resolution failed: {exc}", file=sys.stderr)

    report = build_report_json(
        results, cfg,
        r_startup_s=r_startup,
        r_version=r_ver,
        harmonizr_version=h_ver,
        r_cores=args.with_r_cores,
    )

    # Write output
    out_path = args.out
    if out_path is None:
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_path = str(Path(cfg.paths_results_dir) / f"benchmark_{ts}.json")

    write_report_json(report, out_path)
    print(f"Results written to {out_path}")

    md_path = args.md
    if md_path is None:
        md_path = str(Path(out_path).with_suffix(".md"))

    md = generate_markdown(report)
    Path(md_path).write_text(md)
    print(f"Report written to {md_path}")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: cache-r
# ---------------------------------------------------------------------------


def _add_cacher_parser(subparsers: Any) -> None:
    p = subparsers.add_parser("cache-r", help="Run R harmonizR on scenarios and populate cache.")
    p.add_argument("--datasets", nargs="*", default=None, help="Datasets to cache (default: all R-eligible).")
    p.add_argument("--algorithms", nargs="*", default=None, choices=["ComBat", "limma"], help="Filter by algorithm.")
    p.add_argument("--combat-modes", nargs="*", type=int, default=None, choices=[1, 2, 3, 4], help="Filter by ComBat mode.")
    p.add_argument("--tags", nargs="*", default=None, help="Filter by scenario tags.")
    p.add_argument("--cores", nargs="*", type=int, default=None, help="Core counts to cache (default: config default).")
    p.add_argument("--force", action="store_true", help="Re-run even if cache entry exists.")
    p.add_argument("--clear", action="store_true", help="Clear all cache entries and exit.")
    p.add_argument("--config", default=None, help="Config file path.")


def _cmd_cache_r(args: argparse.Namespace) -> int:
    _setup_logging(verbose=False)
    cfg_path = _resolve_config(args.config)
    cfg = load_config(cfg_path)

    if args.clear:
        cache_dir = Path(cfg.paths_cache_dir)
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
        print(f"Cleared cache at {cache_dir}")
        return 0

    registry = build_registry(cfg)
    scenarios = filter_registry(
        registry,
        datasets=args.datasets,
        algorithms=args.algorithms,
        combat_modes=args.combat_modes,
        tags=args.tags,
    )

    if not scenarios:
        print("No scenarios match the given filters.", file=sys.stderr)
        return 1

    from .runners.r_runner import cache_r_scenarios, r_available

    if not r_available():
        print("Error: Rscript not found on PATH.", file=sys.stderr)
        return 1

    cores_list = args.cores or [cfg.r_cache_default_cores]
    for cores in cores_list:
        print(f"Caching with {cores} core(s)...")
        statuses = cache_r_scenarios(
            scenarios, cfg,
            cache_dir=cfg.paths_cache_dir,
            tmp_dir=cfg.paths_tmp_dir,
            cores=cores,
            force=args.force,
        )
        cached = sum(1 for s in statuses.values() if s in ("cached", "hit"))
        errors = sum(1 for s in statuses.values() if s.startswith("error"))
        skipped = sum(1 for s in statuses.values() if s.startswith("skipped"))
        print(f"  {cached} cached, {errors} errors, {skipped} skipped")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: validity
# ---------------------------------------------------------------------------


def _add_validity_parser(subparsers: Any) -> None:
    p = subparsers.add_parser("validity", help="Run validity checks only (no timing).")
    p.add_argument("--datasets", nargs="*", default=None, help="Datasets to check.")
    p.add_argument("--algorithms", nargs="*", default=None, choices=["ComBat", "limma"], help="Filter by algorithm.")
    p.add_argument("--combat-modes", nargs="*", type=int, default=None, choices=[1, 2, 3, 4], help="Filter by ComBat mode.")
    p.add_argument("--tags", nargs="*", default=None, help="Filter by scenario tags.")
    p.add_argument("--with-r", action="store_true", help="Include R concordance checks (requires cache).")
    p.add_argument("--config", default=None, help="Config file path.")


def _cmd_validity(args: argparse.Namespace) -> int:
    _setup_logging(verbose=False)
    cfg_path = _resolve_config(args.config)
    cfg = load_config(cfg_path)

    registry = build_registry(cfg)
    scenarios = filter_registry(
        registry,
        datasets=args.datasets,
        algorithms=args.algorithms,
        combat_modes=args.combat_modes,
        tags=args.tags,
    )

    if not scenarios:
        print("No scenarios match filters.", file=sys.stderr)
        return 1

    # Resolve R results if --with-r
    r_results: dict[str, dict[str, Any] | None] = {}
    if args.with_r:
        try:
            from .runners.r_runner import resolve_r_results
            r_results = resolve_r_results(scenarios, cfg, cfg.paths_cache_dir)
        except Exception as exc:
            print(f"Warning: R resolution failed: {exc}", file=sys.stderr)

    from .datasets import load_dataset, resolve_dataset_paths
    from .runners.python_runner import run_once
    from .validity import compute_concordance, validate_result

    all_pass = True
    for scenario in scenarios:
        paths = resolve_dataset_paths(cfg, scenario.dataset)
        data_df, desc_df = load_dataset(paths)
        result = run_once(data_df, desc_df, scenario)
        result_dict = result.result
        if result_dict is not None and isinstance(result_dict, dict) and "result_df" in result_dict:
            result_df = result_dict["result_df"]
            vr = validate_result(data_df, result_df, scenario.id)
            # R concordance
            r_entry = r_results.get(scenario.id)
            if r_entry is not None and r_entry.get("output_tsv"):
                cv = compute_concordance(result_df, r_entry["output_tsv"])
                vr.concordance_max_rel = cv.concordance_max_rel
                vr.concordance_mean_rel = cv.concordance_mean_rel
                vr.concordance_nan_match = cv.concordance_nan_match
                vr.concordance_shared_features = cv.concordance_shared_features
                vr.concordance_py_only_features = cv.concordance_py_only_features
                vr.concordance_r_only_features = cv.concordance_r_only_features
                vr.concordance_shared_nonnan_cells = cv.concordance_shared_nonnan_cells
                vr.concordance_p95_rel = cv.concordance_p95_rel
            status = "PASS" if vr.error is None else f"FAIL ({vr.error})"
        else:
            vr = None
            status = "FAIL (no result)"
        print(f"  {scenario.id}: {status}")
        if vr and vr.error:
            all_pass = False
        result.release_result()

    return 0 if all_pass else 1


# ---------------------------------------------------------------------------
# Subcommand: profile
# ---------------------------------------------------------------------------


def _add_profile_parser(subparsers: Any) -> None:
    p = subparsers.add_parser("profile", help="cProfile a single scenario.")
    p.add_argument("--dataset", required=True, help="Dataset name.")
    p.add_argument("--algorithm", required=True, choices=["ComBat", "limma"], help="Algorithm.")
    p.add_argument("--combat-mode", type=int, default=1, choices=[1, 2, 3, 4], help="ComBat mode.")
    p.add_argument("--block", type=int, default=None, help="Block size.")
    p.add_argument("--sort", default=None, choices=["sparsity", "jaccard", "seriation"], help="Sort strategy.")
    p.add_argument("--top-n", type=int, default=20, help="Number of top functions to show.")
    p.add_argument("--config", default=None, help="Config file path.")


def _cmd_profile(args: argparse.Namespace) -> int:
    _setup_logging(verbose=True)
    cfg_path = _resolve_config(args.config)
    cfg = load_config(cfg_path)
    import cProfile
    import pstats

    from .datasets import load_dataset, resolve_dataset_paths
    from .runners.python_runner import run_once
    from .scenarios import Scenario

    scenario = Scenario(
        dataset=args.dataset,
        algorithm=args.algorithm,
        combat_mode=args.combat_mode,
        block=args.block,
        sort=args.sort,
    )

    paths = resolve_dataset_paths(cfg, args.dataset)
    data_df, desc_df = load_dataset(paths)

    profiler = cProfile.Profile()
    profiler.enable()
    run_once(data_df, desc_df, scenario)
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("cumtime")
    print(f"\nTop {args.top_n} functions by cumulative time:")
    stats.print_stats(args.top_n)

    return 0


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------


def _add_report_parser(subparsers: Any) -> None:
    p = subparsers.add_parser("report", help="Regenerate markdown report from saved JSON.")
    p.add_argument("--input", required=True, help="Path to v2.0 results JSON.")
    p.add_argument("--out", default=None, help="Output markdown path (default: input path with .md extension).")


def _cmd_report(args: argparse.Namespace) -> int:
    report = read_report_json(args.input)
    md = generate_markdown(report)
    out_path = args.out or str(Path(args.input).with_suffix(".md"))
    Path(out_path).write_text(md)
    print(f"Report written to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: generate-data
# ---------------------------------------------------------------------------


def _add_generate_data_parser(subparsers: Any) -> None:
    p = subparsers.add_parser("generate-data", help="Generate synthetic benchmark datasets.")
    p.add_argument("--datasets", nargs="*", default=None, help="Datasets to generate (default: all synthetic).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--config", default=None, help="Config file path.")


def _cmd_generate_data(args: argparse.Namespace) -> int:
    _setup_logging(verbose=False)
    cfg_path = _resolve_config(args.config)
    cfg = load_config(cfg_path)

    from .datasets import generate_dataset, write_dataset

    datasets_to_gen = args.datasets or ["small", "medium", "large", "scp_small", "scp_large"]
    output_dir = _BENCHMARKS_DIR / cfg.paths_data_dir

    for ds_name in datasets_to_gen:
        if ds_name not in cfg.datasets:
            print(f"Unknown dataset '{ds_name}', skipping.", file=sys.stderr)
            continue
        ds = cfg.datasets[ds_name]
        print(f"Generating '{ds_name}' ({ds.features} x {ds.samples}, {ds.batches} batches)...")
        if ds.missing_frac is None:
            print(f"  Skipping '{ds_name}' (no missing_frac configured).", file=sys.stderr)
            continue
        data_df, desc_df = generate_dataset(
            name=ds_name,
            n_features=ds.features,
            n_samples=ds.samples,
            n_batches=ds.batches,
            missing_frac=ds.missing_frac,
            missing_mode=ds.missing_mode,
            seed=args.seed,
        )
        write_dataset(data_df, desc_df, ds_name, output_dir)
        print(f"  Written to {output_dir}/")

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="HarmonizePy benchmark suite.",
    )
    parser.add_argument("--config", default=None, help="Config file path.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    _add_cacher_parser(subparsers)
    _add_validity_parser(subparsers)
    _add_profile_parser(subparsers)
    _add_report_parser(subparsers)
    _add_generate_data_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cmd_map = {
        "run": _cmd_run,
        "cache-r": _cmd_cache_r,
        "validity": _cmd_validity,
        "profile": _cmd_profile,
        "report": _cmd_report,
        "generate-data": _cmd_generate_data,
    }

    handler = cmd_map.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
