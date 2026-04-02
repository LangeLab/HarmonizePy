"""Command-line interface for HarmonizePy.

Entry points::

    harmonizepy data.tsv batch.csv -o corrected.tsv
    python -m harmonizepy data.tsv batch.csv -o corrected.tsv

Run ``harmonizepy --help`` for full flag documentation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .affiliation import build_affiliation_list
from .blocking import build_block_list
from .core import harmonize
from .io import read_description, read_main_data
from .validation import validate_data_matrix, validate_description, validate_harmonize_args

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(levelname)s [harmonizepy] %(message)s"

_FORMATS = ("tsv", "csv", "feather")

# Map file extensions to output formats — anything unrecognised falls back to tsv
_EXT_TO_FMT: dict[str, str] = {
    ".tsv": "tsv",
    ".txt": "tsv",
    ".csv": "csv",
    ".feather": "feather",
    ".ftr": "feather",
}

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harmonizepy",
        description=(
            "Batch-effect harmonization for mass-spectrometry proteomics data.\n"
            "Wraps ComBat (parametric / non-parametric) and limma::removeBatchEffect.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # Minimal — ComBat mode 1, auto settings\n"
            "  harmonizepy data.tsv batch.csv -o corrected.tsv\n\n"
            "  # Sort by sparsity then block into pairs before correction\n"
            "  harmonizepy data.tsv batch.csv --sort sparsity --block 2 -o corrected.tsv\n\n"
            "  # limma, no unique-removal, silent run\n"
            "  harmonizepy data.tsv batch.csv --algorithm limma --no-unique-removal -q -o out.tsv\n\n"
            "  # Validate inputs and print run plan without computing\n"
            "  harmonizepy data.tsv batch.csv --dry-run\n\n"
            "  # Save a reproducible JSON summary alongside the result\n"
            "  harmonizepy data.tsv batch.csv -o corrected.tsv --summary run.json\n"
        ),
    )

    # -- Positional inputs --------------------------------------------------
    parser.add_argument(
        "data",
        help="Features x samples matrix (TSV, first column = feature identifiers).",
    )
    parser.add_argument(
        "description",
        help="Batch description (CSV with columns: ID, sample, batch).",
    )

    # -- Output -------------------------------------------------------------
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Output file path. Default: <data_stem>_corrected.tsv placed next to the "
            "input data file. Format is inferred from the extension (.tsv, .csv, "
            ".feather/.ftr) unless --output-format is given."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=_FORMATS,
        default=None,
        metavar="{tsv,csv,feather}",
        help=(
            "Force a specific output format regardless of file extension. "
            "Choices: tsv (default), csv, feather. "
            "Feather requires pyarrow: pip install pyarrow."
        ),
    )

    # -- Algorithm ----------------------------------------------------------
    parser.add_argument(
        "--algorithm",
        choices=("ComBat", "limma"),
        default="ComBat",
        help=(
            "Batch correction algorithm. "
            "ComBat (default): empirical Bayes shrinkage, robust to small batches. "
            "limma: linear-model residuals, faster, less aggressive correction."
        ),
    )
    parser.add_argument(
        "--combat-mode",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        metavar="{1,2,3,4}",
        help=(
            "ComBat variant (ignored when --algorithm limma). "
            "1 = parametric, full correction (default — best for most datasets). "
            "2 = non-parametric, full correction (use for very small batches, n < 10). "
            "3 = parametric, location-only (no variance correction). "
            "4 = non-parametric, location-only."
        ),
    )
    parser.add_argument(
        "--needed-values",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Minimum non-missing values a feature must have in every batch it enters. "
            "Default: auto — 2 for ComBat modes 1/3 and limma (parametric estimators "
            "need ≥ 2 observations); 1 for modes 2/4 (non-parametric can handle "
            "single observations). Override only if you have specific quality requirements."
        ),
    )

    # -- Sorting and blocking -----------------------------------------------
    parser.add_argument(
        "--sort",
        choices=("sparsity", "jaccard", "seriation"),
        default=None,
        metavar="{sparsity,jaccard,seriation}",
        help=(
            "Sort batches before blocking so dissimilar batches are not grouped together. "
            "sparsity: order by missing-value fraction (fastest, good default when sorting). "
            "jaccard: pairwise feature-overlap similarity (more accurate). "
            "seriation: optimal leaf ordering via hierarchical clustering (slowest, most "
            "accurate for many batches). "
            "No effect when --block is omitted."
        ),
    )
    parser.add_argument(
        "--block",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Group N consecutive (optionally sorted) batches into one sub-matrix block. "
            "Must be ≥ 2 and < total number of batches. "
            "Reduces memory usage and improves correction for datasets with many batches "
            "(≥ 5). Omit (default) to process all batches together."
        ),
    )

    # -- Feature handling ---------------------------------------------------
    parser.add_argument(
        "--unique-removal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rescue features whose batch-presence pattern is unique (a singleton) by "
            "cropping to the nearest shared pattern before correction. "
            "Enabled by default — keeps more features in the output. "
            "Disable only if you want strict pattern matching (--no-unique-removal)."
        ),
    )

    # -- Workflow helpers ---------------------------------------------------
    parser.add_argument(
        "--summary",
        default=None,
        metavar="PATH",
        help=(
            "Write a JSON run summary to PATH after completion. "
            "Contains all resolved parameters, input/output dimensions, and the "
            "package version — useful for reproducibility and lab notebooks."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load and validate inputs, resolve all parameters, print a run plan, "
            "then exit without running correction. "
            "Exit code 0 if inputs are valid, 1 on validation error. "
            "Use in pipeline pre-checks to catch problems before committing compute time."
        ),
    )

    # -- Verbosity ----------------------------------------------------------
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging: print sub-matrix dimensions, affiliation counts, timing.",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all progress messages; only warnings and errors are shown.",
    )

    # -- Version ------------------------------------------------------------
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('harmonizepy')}",
    )

    return parser


# ---------------------------------------------------------------------------
# Path / format helpers
# ---------------------------------------------------------------------------


def _resolve_output_path(data_path: str, output: str | None) -> str:
    """Return an output path: explicit *output* arg or ``<stem>_corrected.tsv`` next to data."""
    if output is not None:
        return output
    p = Path(data_path)
    return str(p.parent / f"{p.stem}_corrected.tsv")


def _infer_format(path: str, fmt_arg: str | None) -> str:
    """Determine write format from the explicit flag, then from file extension, then tsv."""
    if fmt_arg is not None:
        return fmt_arg
    return _EXT_TO_FMT.get(Path(path).suffix.lower(), "tsv")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _write_result(df: pd.DataFrame, path: str, fmt: str) -> None:
    """Write *df* in the requested format."""
    if fmt == "csv":
        df.to_csv(path)
    elif fmt == "feather":
        # to_feather() requires pyarrow; index is preserved automatically
        df.reset_index().to_feather(path)
    else:
        df.to_csv(path, sep="\t")


def _write_summary(
    path: str,
    *,
    data_file: str,
    description_file: str,
    output_file: str,
    output_format: str,
    algorithm: str,
    combat_mode: int,
    needed_values: int | None,
    sort: str | None,
    block: int | None,
    unique_removal: bool,
    n_features_input: int,
    n_features_output: int,
    n_samples: int,
    n_batches: int,
) -> None:
    """Write a JSON run-summary file to *path*."""
    summary: dict[str, object] = {
        "harmonizepy_version": version("harmonizepy"),
        "data_file": str(Path(data_file).resolve()),
        "description_file": str(Path(description_file).resolve()),
        "output_file": str(Path(output_file).resolve()),
        "output_format": output_format,
        "algorithm": algorithm,
        "combat_mode": combat_mode,
        "needed_values": needed_values,  # None = auto-selected inside harmonize
        "sort_strategy": sort,
        "block_size": block,
        "unique_removal": unique_removal,
        "n_features_input": n_features_input,
        "n_features_output": n_features_output,
        "n_samples": n_samples,
        "n_batches": n_batches,
    }
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logging.getLogger("harmonizepy").info("Run summary written to %s", path)


# ---------------------------------------------------------------------------
# Dry-run display
# ---------------------------------------------------------------------------


def _print_dry_run(
    *,
    data_file: str,
    description_file: str,
    output_file: str,
    output_format: str,
    algorithm: str,
    combat_mode: int,
    needed_values: int | None,
    sort: str | None,
    block: int | None,
    unique_removal: bool,
    n_features: int,
    n_samples: int,
    n_batches: int,
    n_submatrices: int,
) -> None:
    """Print a human-readable dry-run summary to stdout."""
    nv_str = "auto" if needed_values is None else str(needed_values)
    sort_str = sort or "none"
    block_str = str(block) if block is not None else "none (all batches in one group)"
    ur_str = "enabled" if unique_removal else "disabled"
    algo_str = f"{algorithm} mode {combat_mode}" if algorithm == "ComBat" else algorithm
    rule = "─" * 52

    lines = [
        "",
        f"  HarmonizePy {version('harmonizepy')} — dry run",
        f"  {rule}",
        f"  Input data:      {data_file}",
        f"  Description:     {description_file}",
        f"  Output:          {output_file}  [{output_format}]",
        f"  {rule}",
        f"  Features:        {n_features}",
        f"  Samples:         {n_samples}",
        f"  Batches:         {n_batches}",
        f"  Sub-matrices:    {n_submatrices}  (unique affiliation groups)",
        f"  {rule}",
        f"  Algorithm:       {algo_str}",
        f"  needed_values:   {nv_str}",
        f"  Sort strategy:   {sort_str}",
        f"  Block size:      {block_str}",
        f"  Unique removal:  {ur_str}",
        f"  {rule}",
        "  Inputs valid. Use without --dry-run to run correction.",
        "",
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments and run the harmonization pipeline."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # -- Logging setup ------------------------------------------------------
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(format=_LOG_FORMAT, level=level, force=True)

    # -- Resolve output path and format -------------------------------------
    output_path = _resolve_output_path(args.data, args.output)
    output_fmt = _infer_format(output_path, args.output_format)

    # -- File existence pre-check (fast fail before any loading) ------------
    if not Path(args.data).is_file():
        parser.error(f"data file not found: '{args.data}'")
    if not Path(args.description).is_file():
        parser.error(f"description file not found: '{args.description}'")

    # -- Dry-run path -------------------------------------------------------
    if args.dry_run:
        try:
            data = read_main_data(args.data)
            description = read_description(args.description)
            validate_data_matrix(data)
            validate_description(description, data)
            # Resolve n_batches so block_size can be validated properly
            sample_to_batch = dict(
                zip(
                    description.iloc[:, 0].astype(str),
                    description.iloc[:, 2].astype(int),
                    strict=True,
                )
            )
            batch_arr = np.array([sample_to_batch[col] for col in data.columns], dtype=np.int64)
            n_batches = int(np.unique(batch_arr).size)
            validate_harmonize_args(
                args.algorithm,
                args.combat_mode,
                args.needed_values if args.needed_values is not None else 2,
                sort_strategy=args.sort,
                block_size=args.block,
                unique_removal=args.unique_removal,
                n_batches=n_batches,
            )
            # Resolve effective needed_values for affiliation spotting
            if args.needed_values is not None:
                nv_eff = args.needed_values
            elif args.algorithm == "limma" or args.combat_mode in (1, 3):
                nv_eff = 2
            else:
                nv_eff = 1
            # Build block list
            if args.block is not None:
                block_list = build_block_list(batch_arr, block_size=args.block)
            else:
                block_list = batch_arr.copy()
            # Spot sub-matrices (no adjustment)
            affiliation_list = build_affiliation_list(data, batch_arr, block_list, nv_eff)
            n_submatrices = len({a for a in affiliation_list if len(a) > 0})
        except (ValueError, TypeError, KeyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        n_features, n_samples = data.shape
        _print_dry_run(
            data_file=args.data,
            description_file=args.description,
            output_file=output_path,
            output_format=output_fmt,
            algorithm=args.algorithm,
            combat_mode=args.combat_mode,
            needed_values=args.needed_values,
            sort=args.sort,
            block=args.block,
            unique_removal=args.unique_removal,
            n_features=n_features,
            n_samples=n_samples,
            n_batches=n_batches,
            n_submatrices=n_submatrices,
        )
        return  # exit 0 implicitly — no correction performed

    # -- Full pipeline run --------------------------------------------------
    try:
        data_df = read_main_data(args.data)
        description_df = read_description(args.description)

        n_features_input, n_samples = data_df.shape
        n_batches = int(description_df.iloc[:, 2].nunique())

        result = harmonize(
            data_df,
            description_df,
            algorithm=args.algorithm,
            combat_mode=args.combat_mode,
            needed_values=args.needed_values,
            sort=args.sort,
            block=args.block,
            unique_removal=args.unique_removal,
        )
    except (ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # -- Write output -------------------------------------------------------
    try:
        _write_result(result, output_path, output_fmt)
    except Exception as exc:
        print(f"ERROR writing output to '{output_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    logging.getLogger("harmonizepy").info("Output written to %s", output_path)

    # -- Optional JSON summary ----------------------------------------------
    if args.summary:
        try:
            _write_summary(
                args.summary,
                data_file=args.data,
                description_file=args.description,
                output_file=output_path,
                output_format=output_fmt,
                algorithm=args.algorithm,
                combat_mode=args.combat_mode,
                needed_values=args.needed_values,
                sort=args.sort,
                block=args.block,
                unique_removal=args.unique_removal,
                n_features_input=n_features_input,
                n_features_output=result.shape[0],
                n_samples=n_samples,
                n_batches=n_batches,
            )
        except OSError as exc:
            print(f"WARNING: could not write summary to '{args.summary}': {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
