"""Command-line interface for HarmonizePy.

Entry points::

    harmonizepy data.tsv batch.csv -o corrected.tsv
    python -m harmonizepy data.tsv batch.csv -o corrected.tsv

Run ``harmonizepy --help`` for full flag documentation.
"""

# PYTHON_ARGCOMPLETE_OK

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
_FILE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

_FORMATS = ("tsv", "csv", "feather")

# Map file extensions to output formats. Anything unrecognised falls back to tsv.
_EXT_TO_FMT: dict[str, str] = {
    ".tsv": "tsv",
    ".txt": "tsv",
    ".csv": "csv",
    ".feather": "feather",
    ".ftr": "feather",
}

# Config keys that map directly to CLI flag destinations
_VALID_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "algorithm",
        "combat_mode",
        "needed_values",
        "sort",
        "block",
        "unique_removal",
        "output",
        "output_format",
        "summary",
    }
)

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
            "  # Minimal: ComBat mode 1, auto settings\n"
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
            "1 = parametric, full correction (default, best for most datasets). "
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
            "Default: auto, 2 for ComBat modes 1/3 and limma (parametric estimators "
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
            "Enabled by default; keeps more features in the output. "
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
            "package version, useful for reproducibility and lab notebooks."
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
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "Path to a TOML, JSON, or YAML (.yaml/.yml) config file. "
            "Keys map 1:1 to CLI flag names (algorithm, combat_mode, needed_values, "
            "sort, block, unique_removal, output, output_format, summary). "
            "CLI flags override config file values; config file overrides built-in defaults. "
            "YAML requires pyyaml; TOML requires Python ≥ 3.11 or tomli on older versions."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help=(
            "Print the run summary as JSON to stdout after completion. "
            "For programmatic consumption: pipe to jq or capture in a shell script. "
            "Suppresses INFO log messages so stdout contains only the JSON object."
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

    # -- Log file (optional override / disable) -----------------------------
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help=(
            "Write a detailed execution log to PATH (default: <output_stem>.log). "
            "Includes timestamps and module-level information. "
            "Use --no-log to disable file logging entirely."
        ),
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable automatic log file creation. By default a .log file is "
        "written alongside the output with full DEBUG information.",
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


def _resolve_log_path(output_path: str) -> str:
    """Return the default log path: same directory and stem as *output_path* with ``.log``.

    Examples
    --------
    >>> _resolve_log_path("/out/corrected.tsv")
    '/out/corrected.log'
    >>> _resolve_log_path("result.csv")
    'result.log'
    """
    p = Path(output_path)
    return str(p.parent / f"{p.stem}.log")


def _load_config(path: str) -> dict[str, object]:
    """Load a TOML, JSON, or YAML config file and return the parsed mapping.

    Supported formats
    -----------------
    ``.json``         (stdlib, always available)
    ``.toml``         (``tomllib`` on Python >= 3.11, or ``tomli`` package)
    ``.yaml``/``.yml`` (requires ``pyyaml``: ``pip install pyyaml``)
    """
    p = Path(path)
    ext = p.suffix.lower()
    raw: object

    if ext == ".json":
        with p.open("rb") as fh:
            raw = json.load(fh)

    elif ext == ".toml":
        try:
            import tomllib  # stdlib >= 3.11
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef,import-not-found]
            except ImportError:
                raise ImportError(
                    "TOML config requires 'tomllib' (Python ≥ 3.11) "
                    "or 'tomli': pip install harmonizepy[config]"
                ) from None
        with p.open("rb") as fh:
            raw = tomllib.load(fh)

    elif ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "YAML config requires 'pyyaml': pip install harmonizepy[config]"
            ) from None
        with p.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

    else:
        raise ValueError(f"Unsupported config format '{ext}'. Use .toml, .json, .yaml, or .yml.")

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a TOML/JSON/YAML mapping, got {type(raw).__name__}.")
    return raw  # type narrowed to dict[Any, Any] by isinstance check above


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


def _build_summary_dict(
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
) -> dict[str, object]:
    """Build the run-summary dict (shared by --summary and --json)."""
    return {
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


def _write_summary(path: str, summary: dict[str, object]) -> None:
    """Write a JSON run-summary file to *path*."""
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
        f"  HarmonizePy {version('harmonizepy')} dry run",
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

    # Shell completion: silently skip when argcomplete is not installed.
    # This is the standard optional-dependency pattern, not a bug swallow.
    try:
        import argcomplete  # type: ignore[import-not-found]

        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    # -- Config file pre-parse: apply config values as parser defaults ------
    # Use a lightweight pre-parser so we can call parser.set_defaults() before
    # the full parse.  CLI flags always win because set_defaults() only affects
    # the default value, not an explicitly supplied argument.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default=None)
    _pre_ns, _ = _pre.parse_known_args(argv)
    if _pre_ns.config is not None:
        if not Path(_pre_ns.config).is_file():
            parser.error(f"config file not found: '{_pre_ns.config}'")
        try:
            _cfg = _load_config(_pre_ns.config)
        except (ValueError, ImportError, OSError) as exc:
            parser.error(str(exc))
        _unknown = set(_cfg) - _VALID_CONFIG_KEYS
        if _unknown:
            parser.error(f"Unknown config key(s): {', '.join(sorted(_unknown))}")
        parser.set_defaults(**{k: v for k, v in _cfg.items() if k in _VALID_CONFIG_KEYS})

    args = parser.parse_args(argv)

    # -- Resolve output path and format -------------------------------------
    output_path = _resolve_output_path(args.data, args.output)
    output_fmt = _infer_format(output_path, args.output_format)

    # -- Logging setup ------------------------------------------------------
    # Configure the root harmonizepy logger so all child modules
    # (harmonizepy.combat, harmonizepy.core, etc.) inherit the settings.
    # --json suppresses INFO so stdout contains only the JSON object.
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet or args.json_output:
        level = logging.WARNING
    else:
        level = logging.INFO

    root_logger = logging.getLogger("harmonizepy")
    root_logger.setLevel(logging.DEBUG)
    root_logger.propagate = False

    # Terminal handler: clean format, no timestamps, respects --verbose/--quiet.
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT))
    root_logger.addHandler(console)

    # File handler: always-on by default, writes DEBUG with timestamps.
    # Log path is derived from the output path (e.g. corrected.tsv -> corrected.log)
    # unless overridden via --log-file, or disabled via --no-log.
    if not args.no_log:
        log_path = args.log_file if args.log_file else _resolve_log_path(output_path)
        try:
            fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(_FILE_LOG_FORMAT))
            root_logger.addHandler(fh)
        except OSError:
            # If the log file cannot be created, warn and continue without it.
            logging.getLogger("harmonizepy").warning(
                "Could not create log file at %s; proceeding without file logging.",
                log_path,
            )

    # -- File existence pre-check (fast fail before any loading) ------------
    if not Path(args.data).is_file():
        parser.error(f"data file not found: '{args.data}'")
    if not Path(args.description).is_file():
        parser.error(f"description file not found: '{args.description}'")

    # -- Dry-run path -------------------------------------------------------
    # Suppress pipeline logging during dry-run so the plan output is clean.
    if args.dry_run:
        _prev_level = console.level
        console.setLevel(logging.WARNING)
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
            # Resolve effective needed_values once for both validation and spotting
            if args.needed_values is not None:
                nv_eff = args.needed_values
            elif args.algorithm == "limma" or args.combat_mode in (1, 3):
                nv_eff = 2
            else:
                nv_eff = 1
            validate_harmonize_args(
                args.algorithm,
                args.combat_mode,
                nv_eff,
                sort_strategy=args.sort,
                block_size=args.block,
                unique_removal=args.unique_removal,
                n_batches=n_batches,
            )
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
        finally:
            console.setLevel(_prev_level)

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
        return  # exit 0 implicitly, no correction performed

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

    # -- Run summary (--summary file and/or --json stdout) ------------------
    if args.summary or args.json_output:
        _summary = _build_summary_dict(
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
        if args.summary:
            try:
                _write_summary(args.summary, _summary)
            except OSError as exc:
                print(
                    f"WARNING: could not write summary to '{args.summary}': {exc}",
                    file=sys.stderr,
                )
        if args.json_output:
            print(json.dumps(_summary, indent=2))


if __name__ == "__main__":
    main()
