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
import re
import sys
import tomllib
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path, PurePosixPath, PureWindowsPath

import pandas as pd

from .core import harmonize, plan_affiliations, prepare_inputs, resolve_needed_values
from .io import read_description, read_main_data
from .validation import sample_batch_map

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(levelname)s [harmonizepy] %(message)s"
_FILE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Errors from reading, validating, correcting, or writing that end a run with
# exit code 1 and one ERROR line instead of a traceback.
_RUN_ERRORS = (ValueError, TypeError, KeyError, ImportError, OSError)

_FORMATS = ("tsv", "csv", "parquet")

# Map file extensions to output formats. Anything unrecognised falls back to tsv.
_EXT_TO_FMT: dict[str, str] = {
    ".tsv": "tsv",
    ".txt": "tsv",
    ".csv": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
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
            "  harmonizepy data.tsv batch.csv\n\n"
            "  # Explicit output path\n"
            "  harmonizepy data.tsv batch.csv -o corrected.tsv\n\n"
            "  # Sort by sparsity then block into pairs before correction\n"
            "  harmonizepy data.tsv batch.csv --sort sparsity --block 2 -o corrected.parquet\n\n"
            "  # limma, no unique-removal, silent run\n"
            "  harmonizepy data.tsv batch.csv --algorithm limma --no-unique-removal -q -o out.parquet\n\n"
            "  # Validate inputs and print run plan without computing\n"
            "  harmonizepy data.tsv batch.csv --dry-run\n\n"
            "  # Save a reproducible JSON summary alongside the result\n"
            "  harmonizepy data.tsv batch.csv -o corrected.parquet --summary run.json\n"
        ),
    )

    # -- Positional inputs --------------------------------------------------
    parser.add_argument(
        "data",
        help="Features x samples matrix (TSV, first column = feature identifiers).",
    )
    parser.add_argument(
        "description",
        help=(
            "Batch description (CSV with columns ID, sample, batch). Without the names "
            "ID and batch, the first column holds the sample IDs and the third the batches."
        ),
    )

    # -- Output -------------------------------------------------------------
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Output file path. Default: <data_stem>_corrected placed next to the input "
            "data file, with the extension of --output-format (.tsv by default, .csv, or "
            ".parquet). Format is inferred from the extension (.tsv, .csv, .parquet/.pq) "
            "unless --output-format is given."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=_FORMATS,
        default=None,
        metavar="{tsv,csv,parquet}",
        help=(
            "Force a specific output format regardless of file extension. "
            "Choices: tsv (default), csv, parquet. "
            "Parquet requires pyarrow: pip install harmonizepy[io]."
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
            "1 = parametric, location + scale (default, best for most datasets). "
            "2 = parametric, location only (no variance correction). "
            "3 = non-parametric, location + scale (use for very small batches, n < 10). "
            "4 = non-parametric, location only."
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
            "Exit code 0 if inputs are valid, 1 on a validation error, 2 on a usage error "
            "or a missing input file. Writes no files; cannot be combined with --log-file, "
            "--summary, or --json. "
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


def _resolve_output_path(
    data_path: str, output: str | None, output_format: str | None = None
) -> str:
    """Return an output path based on explicit args or CLI defaults.

    Without ``-o``, the default path is ``<stem>_corrected`` next to the input
    data file, with the extension of ``--output-format`` (``.tsv`` by default,
    ``.csv``, or ``.parquet``).
    """
    if output is not None:
        return output
    # Preserve the path style supplied by the caller. This keeps POSIX-style
    # paths stable when the CLI is exercised on Windows, while retaining
    # native drive/backslash formatting for regular Windows paths.
    path_type = (
        PureWindowsPath
        if "\\" in data_path or re.match(r"[A-Za-z]:([\\/]|$)", data_path)
        else PurePosixPath
    )
    p = path_type(data_path)
    ext = {"csv": ".csv", "parquet": ".parquet"}.get(output_format or "tsv", ".tsv")
    return str(p.parent / f"{p.stem}_corrected{ext}")


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
    ``.toml``         (stdlib ``tomllib``)
    ``.yaml``/``.yml`` (requires ``pyyaml``: ``pip install pyyaml``)
    """
    p = Path(path)
    ext = p.suffix.lower()
    raw: object

    if ext == ".json":
        with p.open("rb") as fh:
            raw = json.load(fh)

    elif ext == ".toml":
        with p.open("rb") as fh:
            raw = tomllib.load(fh)

    elif ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as error:
            raise ImportError(
                "YAML config requires 'pyyaml': pip install harmonizepy[config]"
            ) from error
        with p.open(encoding="utf-8") as fh:
            try:
                raw = yaml.safe_load(fh)
            except yaml.YAMLError as error:
                raise ValueError(f"invalid YAML in config file '{path}': {error}") from error

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
    elif fmt == "parquet":
        from .io import _HAVE_PYARROW

        if not _HAVE_PYARROW:
            raise ImportError("Parquet output requires pyarrow: pip install harmonizepy[io]")
        df.to_parquet(path, index=True, engine="pyarrow")
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
    needed_values: int,
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
        "needed_values": needed_values,
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
    needed_values: int,
    sort: str | None,
    block: int | None,
    unique_removal: bool,
    n_features: int,
    n_samples: int,
    n_batches: int,
    n_submatrices: int,
) -> None:
    """Print a human-readable dry-run summary to stdout."""
    nv_str = str(needed_values)
    sort_str = sort or "none"
    block_str = str(block) if block is not None else "none (all batches in one group)"
    ur_str = "enabled" if unique_removal else "disabled"
    algo_str = f"{algorithm} mode {combat_mode}" if algorithm == "ComBat" else algorithm
    rule = "-" * 52

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
        import argcomplete

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
    output_path = _resolve_output_path(args.data, args.output, args.output_format)
    output_fmt = _infer_format(output_path, args.output_format)

    # A dry run prints its plan and writes nothing, so options that only shape the
    # files or the summary of a correction run would be ignored without a word.
    if args.dry_run:
        for flag, value in (
            ("--log-file", args.log_file),
            ("--summary", args.summary),
            ("--json", args.json_output),
        ):
            if value:
                parser.error(f"{flag} cannot be used with --dry-run")

    # -- File existence pre-check (fast fail before any loading) ------------
    if not Path(args.data).is_file():
        parser.error(f"data file not found: '{args.data}'")
    if not Path(args.description).is_file():
        parser.error(f"description file not found: '{args.description}'")

    package_logger = logging.getLogger("harmonizepy")
    saved_level, saved_propagate = package_logger.level, package_logger.propagate
    handlers = _configure_logging(args, output_path)
    try:
        if args.dry_run:
            _run_dry_run(args, output_path, output_fmt, console=handlers[0])
        else:
            _run_correction(args, output_path, output_fmt)
    finally:
        _detach_logging(handlers)
        package_logger.setLevel(saved_level)
        package_logger.propagate = saved_propagate


def _configure_logging(args: argparse.Namespace, output_path: str) -> list[logging.Handler]:
    """Attach the console handler and, for a correction run, the log file handler.

    The console shows INFO by default, DEBUG with ``--verbose``, and WARNING with
    ``--quiet`` or ``--json`` (so that stdout holds only the JSON object). The log
    file, next to the output unless ``--log-file`` or ``--no-log`` is given,
    records DEBUG with timestamps. A dry run writes no log file. The console
    handler is first in the returned list.
    """
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet or args.json_output:
        level = logging.WARNING
    else:
        level = logging.INFO

    package_logger = logging.getLogger("harmonizepy")
    package_logger.setLevel(logging.DEBUG)
    package_logger.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT))
    package_logger.addHandler(console)
    handlers: list[logging.Handler] = [console]

    if not args.no_log and not args.dry_run:
        log_path = args.log_file if args.log_file else _resolve_log_path(output_path)
        try:
            file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        except OSError:
            package_logger.warning(
                "Could not create log file at %s; proceeding without file logging.", log_path
            )
        else:
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(_FILE_LOG_FORMAT))
            package_logger.addHandler(file_handler)
            handlers.append(file_handler)
    return handlers


def _detach_logging(handlers: list[logging.Handler]) -> None:
    """Remove and close the handlers ``_configure_logging`` attached."""
    package_logger = logging.getLogger("harmonizepy")
    for handler in handlers:
        package_logger.removeHandler(handler)
        handler.close()


def _run_dry_run(
    args: argparse.Namespace, output_path: str, output_fmt: str, *, console: logging.Handler
) -> None:
    """Validate the inputs, plan the groups as a correction run would, and print the plan.

    Pipeline log messages are held back to WARNING so that the plan is the only
    output. Exits with code 1 and an ERROR line on an input error.
    """
    previous_level = console.level
    console.setLevel(logging.WARNING)
    try:
        inputs = prepare_inputs(
            args.data,
            args.description,
            algorithm=args.algorithm,
            combat_mode=args.combat_mode,
            needed_values=args.needed_values,
            sort=args.sort,
            block=args.block,
            unique_removal=args.unique_removal,
        )
        plan = plan_affiliations(
            inputs.data,
            inputs.batch_list,
            sort=args.sort,
            block=args.block,
            needed_values=inputs.needed_values,
            unique_removal=args.unique_removal,
        )
    except _RUN_ERRORS as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        console.setLevel(previous_level)

    n_features, n_samples = inputs.data.shape
    _print_dry_run(
        data_file=args.data,
        description_file=args.description,
        output_file=output_path,
        output_format=output_fmt,
        algorithm=args.algorithm,
        combat_mode=args.combat_mode,
        needed_values=inputs.needed_values,
        sort=args.sort,
        block=args.block,
        unique_removal=args.unique_removal,
        n_features=n_features,
        n_samples=n_samples,
        n_batches=inputs.n_batches,
        n_submatrices=plan.n_groups,
    )


def _run_correction(args: argparse.Namespace, output_path: str, output_fmt: str) -> None:
    """Correct the data, write the output, and write or print the run summary.

    Exits with code 1 and an ERROR line when reading, validating, correcting, or
    writing the output fails. A summary file that cannot be written gives a
    WARNING line and does not change the exit code.
    """
    try:
        data_df = read_main_data(args.data)
        description_df = read_description(args.description)
        n_features_input, n_samples = data_df.shape
        n_batches = len(set(sample_batch_map(description_df).values()))
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
    except _RUN_ERRORS as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        _write_result(result, output_path, output_fmt)
    except _RUN_ERRORS as exc:
        print(f"ERROR writing output to '{output_path}': {exc}", file=sys.stderr)
        sys.exit(1)
    logging.getLogger("harmonizepy").info("Output written to %s", output_path)

    if not (args.summary or args.json_output):
        return
    summary = _build_summary_dict(
        data_file=args.data,
        description_file=args.description,
        output_file=output_path,
        output_format=output_fmt,
        algorithm=args.algorithm,
        combat_mode=args.combat_mode,
        needed_values=resolve_needed_values(args.algorithm, args.combat_mode, args.needed_values),
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
            _write_summary(args.summary, summary)
        except OSError as exc:
            print(f"WARNING: could not write summary to '{args.summary}': {exc}", file=sys.stderr)
    if args.json_output:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
