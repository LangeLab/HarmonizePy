<!-- markdownlint-disable MD024 -->

# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- NaN handling: per-feature Beta.NA computation in ComBat and limma engines, matching R sva v3.60.0. NaN-shaped sub-matrices pass through to engines; each feature independently omits NaN from OLS, variance, and EB computations.
- NaN-safe iterative (`_it_sol`) and non-parametric (`_int_eprior`) EB solvers using per-gene non-NA counts.
- Per-cell NaN R concordance fixtures and murine medulloblastoma real-data verification. All modes concordant at max_rel 0.0003.
- Benchmark suite: synthetic data generator, timing/memory harness, R comparison template, RESULTS.md with data specs and per-scenario tables.
- Benchmark datasets: small/medium/large (bulk proteomics), scp_small/scp_large (SCP with abundance-dependent missingness), murine_medulloblastoma (real).
- CLI logging flags (`--log-file`, `--no-log`), module-level logging, pipeline timing.
- CI workflow (ruff, mypy, pytest via uv).
- FEATURE_PARITY.md and MASTER_REFERENCE.md with complete R vs Python difference inventory.
- STYLE_GUIDE.md and TESTING.md documentation.

### Changed

- Engine NaN handling: replaced row-dropping with per-feature Beta.NA path. Dense path unchanged.
- `splitting.py`: removed column-dropping logic. Per-cell NaN flows through to engines.
- `validation.py`: NaN allowed in engine inputs; validation checks only structural properties.
- Performance: `_int_eprior` binomial formula optimization (mode 3: 2.2x faster), `build_affiliation_list` vectorization (7x faster), `splitting.py` numpy-only hot loop (1.4-2.2x faster).
- IO: parquet default output (12x faster writes, 2x smaller), pyarrow CSV engine auto-detect, removed feather support.
- Test NaN invariants: now verify correct NaN handling rather than asserting NaN-free input.
- Console logging to stderr. Logger propagation disabled.

### Fixed

- `_it_sol`: `t2_n_g_hat` uses original `g_hat` per iteration (not updated `g_new`), matching R's `postmean` formula. Prevents divergence with NaN data.
- `_int_eprior`: likelihood normalization uses per-gene `n_i` instead of fixed batch size. Enables correct non-parametric EB with per-cell NaN.
- Benchmark data generator: minimum 4 samples per batch to satisfy needed_values=2.
- Benchmark runner: unique tagged filenames per scenario.
- Dry-run mode: pipeline DEBUG suppressed during plan output.

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
