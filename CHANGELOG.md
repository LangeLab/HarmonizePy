<!-- markdownlint-disable MD024 -->

# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Module-level logging**: `logger.info`/`logger.debug` messages at pipeline milestones in `combat.py`, `combat_wrapper.py`, `limma_wrapper.py`, `sorting.py`, `blocking.py`, `affiliation.py`, `splitting.py`, and `io.py`. Covers convergence, sub-matrix adjustment, feature affiliation counts, sorting strategies, block construction, and I/O operations.
- **CLI logging flags**: `--log-file PATH` overrides the default log path (auto-derived from output path as `<output_stem>.log`). `--no-log` disables file logging. By default a `.log` file is always written alongside the output with full DEBUG detail and timestamps.
- **Logging warnings**: non-convergence after 1M iterations, all features dropped by `needed_values`, log file write failure.
- **Pipeline timing**: total wall-clock duration reported at INFO level on completion.
- **CI workflow** (`.github/workflows/ci.yml`): runs `ruff check`, `mypy src/`, `pytest tests/` on Python 3.12 via `uv`. Triggers on push/PR to main.
- **STYLE_GUIDE.md**: writing conventions for docstrings, imports, type annotations, naming, commenting, module responsibilities, and table usage policy.
- **TESTING.md**: test standards covering chain-of-thought docstrings, invariant/contract/failure testing, array assertion rules, and mock boundaries.
- **FEATURE_PARITY.md**: R HarmonizR v1.10.0 vs HarmonizePy v0.2.0 comparison across core algorithms, pipeline parameters, I/O, sorting, and compute model.
- **Invariant tests**: output isolation, determinism, and input non-mutation added to `test_api.py`.

### Changed

- **Console logging**: now writes to stderr instead of stdout, preventing log messages from polluting JSON or dry-run output.
- **Root logger**: `harmonizepy` logger configured with `propagate=False` to prevent double-logging when external code configures the root handler.
- **LICENSE** (`LICENSE`): GPL-3.0 with individual copyright holder and acknowledgments for original R HarmonizR.
- **plan.md**: phase markers updated to reflect completed and partial items.

### Fixed

- `combat.py`: removed stale `# type: ignore[no-any-return]` no longer needed after mypy inference improvements.
- `io.py` and `combat_wrapper.py`: removed redundant `cast()` calls.
- `splitting.py`: removed duplicate import block introduced during logging refactor.
- Dry-run mode: pipeline DEBUG messages are suppressed during dry-run computation so the plan output is clean.

## [0.2.0] - 2026-04-01

### Added

- **Batch sorting**: three strategies matching R HarmonizR v1.8.0: sparsity (completeness-based), jaccard (greedy nearest-neighbour), seriation (PCA-based optimal leaf ordering). See `harmonizepy.sorting`.
- **Batch blocking**: group consecutive batches into superblocks for dissection via `block=N`. See `harmonizepy.blocking`.
- **Unique-combination removal**: rescue singleton features by cropping to the nearest shared batch-presence pattern (`unique_removal=True`, matching R's `ur=TRUE` default). See `harmonizepy.affiliation`.
- **CLI** (`__main__.py`): full command-line interface with positional args, `--algorithm`, `--combat-mode`, `--sort`, `--block`, `--unique-removal`, `--needed-values`, `--dry-run`, `--summary`, `--json`, `--config`, `--verbose`/`--quiet`, `--version`, `--help`. Output formats: TSV, CSV, Feather.
- **`HarmonizeConfig`** frozen dataclass for reproducible run configuration.
- **`validate_harmonize_args`** centralised validation of all pipeline parameters.

### Changed

- **`harmonize()` pipeline** (`core.py`): extended from direct ComBat/limma adjustment to the full HarmonizR workflow: sort, block, spot missing, unique removal, split, adjust, concat, re-sort columns.
- **Splitting** (`splitting.py`): pre-allocates a single output array instead of one per affiliation group, reducing peak memory from ~3x input to ~1x input.
- **Input validation** (`validation.py`): extracted from individual modules into a single file with consistent error messages.
- **IO** (`io.py`): now reads TSV, CSV, and Feather formats; drops all-NaN rows on read; preserves all-NaN columns (structural missingness).

### Removed

- Redundant validation spread across `combat.py` and `limma_wrapper.py`, consolidated into `validation.py`.
- `np.matrix` usage from ComBat engine, now fully vectorised NumPy.

### Fixed

- `_int_eprior` in `combat.py`: guard against zero/negative `d_hat` causing `-inf` in log-likelihood computation (clamp to `1e-12`).
- `_aprior` / `_bprior`: return flat prior (`1.0`) when variance cannot be estimated (zero-variance features).
- `combat()`: force `mean_only=True` when any batch has fewer than 2 samples (mirrors R `sva::ComBat` behaviour).
- `combat()`: negative `delta_star` from `_postvar` clamped before `sqrt`.

### R Concordance

- Full test suite validated against R `sva::ComBat`, `limma::removeBatchEffect`, and `HarmonizR` (v1.8.0 pipeline with sort, block, and unique-removal toggles). Covers small, medium, large, unbalanced, minimal, high-var, near-constant, negative, wide, sparse, high-missingness, and singleton-batch edge cases.

## [0.1.0] - 2025-12-15

### Added

- **ComBat engine** (`combat.py`): pure NumPy reimplementation of Johnson, Li & Rabinovic (2007). All four modes: parametric and non-parametric, with and without scale correction. Supports `ref_batch` and automatic `mean_only` fallback for singleton batches.
- **limma engine** (`limma_wrapper.py`): pure NumPy `removeBatchEffect` using sum-to-zero contrasts and OLS (Ritchie et al. 2015).
- **HarmonizR pipeline** (`core.py`): structural missingness handling via per-feature batch presence detection, grouping by affiliation, NaN-free sub-matrix extraction, adjustment, and reassembly.
- **`harmonize()` entry point**: accepts DataFrames, file paths (TSV/CSV), and Path objects.
- **Data IO** (`io.py`): `read_main_data`, `read_description`, `write_output` for TSV/CSV.
- **Input validation**: `validate_data_matrix`, `validate_description`, `validate_combat_input`, `validate_limma_input`.
- **CLI** (`__main__.py`): minimal command-line wrapper with `-o` output flag.
- **Test suite**: R-concordance tests for all ComBat modes and limma, plus unit tests for all internal helpers.
- **R fixture generation**: scripts under `tests/fixtures/` generating reference outputs from R `sva`, `limma`, and `HarmonizR`.
