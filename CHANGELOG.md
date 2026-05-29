<!-- markdownlint-disable MD024 -->

# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] - 2026-05-29

### Added

- Fixture-backed CI workflow with a representative test subset for push and pull request runs.
- Top-level benchmark results documentation now summarizes dataset design, benchmark environment, R-vs-Python representative runtimes, and the mixed R-backed versus Python-only benchmark policy.

### Changed

- R-generated TSV and CSV fixtures are now tracked in git so CI can run concordance coverage without regenerating them.
- Package metadata, smoke checks, README examples, parity docs, and workspace reference docs are aligned on version 0.3.2.
- CLI default output now writes `<data_stem>_corrected.tsv` unless parquet is explicitly requested.

### Fixed

- Repo-wide Ruff and mypy failures cleared on the current CI target.
- Install docs and dependency notes now document `harmonizepy[io]` for parquet support and pyarrow-accelerated IO.

## [0.3.1] - 2026-05-28

### Added

- New benchmark system: in-process runner, YAML-driven scenario registry, stable R cache keys, 6-command CLI (`run`/`cache-r`/`validity`/`profile`/`report`/`generate-data`), and JSON/Markdown report generation.
- Per-scenario validity checks, R concordance in the harness, and scenario tags (`parametric`, `nonparametric`, `blocking`, `sorted`, `py_only`) for filtering.
- Comparable memory metrics: R heap and RSS delta alongside Python tracemalloc including output array `nbytes`.
- R wrapper recovers from `sva::ComBat` singular matrix crashes by removing problematic features and retrying.

### Changed

- `splitting.py` and `core.py`: keep split-and-rebuild on one final assembled matrix, cache per-affiliation column indices, and for sorted runs scatter corrected submatrices directly back to original column positions instead of reordering the full result afterwards.
- Internal optimization and benchmark-status documents reorganized to separate completed work from remaining memory-focused items.
- Benchmarking now uses public `harmonize()` directly, defaults R baselines to 1 core, suppresses timed-run logging, and auto-converts DIA CSV input to TSV for R.

### Fixed

- Orphaned `doParallel` workers killed via process group on R timeout.
- R version strings cleaned of renv status messages.
- `generate_from_config` uses per-dataset `missing_frac` instead of hardcoded 0.0.
- `block` passed to HarmonizR as numeric (was silently falling back to unblocked). Historical `Block = 2` results invalidated.
- Memory/reporting fixes: tracemalloc MB divisor corrected from 1e6 to 1024^2, and R `gc()` now reads the MB column instead of raw cell counts.
- CLI/config fixes: config paths now resolve relative to the config file, and `--combat-modes` now reaches `filter_registry` in `run`, `cache-r`, and `validity`.
- Harness batch column lookup uses column name with positional fallback matching `core.py`.
- Dead code removed: `Scenario.cache_key`, `phase_times` field and aggregation.

### Performance

- `_combat_nan`: consolidated NaN-heavy ComBat work including vectorized standardization/final adjustment, grouped valid-mask reuse, one-hot grouped Beta.NA with reused reduced layouts, grouped row variance by sums/squared sums, and reusable `_it_sol` buffers. These changes reduced engine overhead while preserving concordance.
- Non-parametric ComBat: binomial residual precompute plus block-vectorized `_int_eprior` cut large mode 3 from 9.76s to 5.87s.
- limma NaN path now groups identical valid-observation masks and solves each reduced design once, improving representative full-pipeline timings from 0.48s to 0.27s on murine and from 1.13s to 0.36s on DIA while preserving concordance.
- Affiliation and sorting helpers now reuse repeated affiliation patterns and precomputed non-NaN presence masks, reducing representative sorted/block timings from 0.15s to 0.13s on medium ComBat mode 1 and from 2.80s to 2.65s on `scp_large` limma.
- Data-matrix validation now inspects `df.dtypes` directly instead of indexing every column, reducing `validate_data_matrix()` on `scp_large` from 0.90s to 0.05s and improving representative `scp_large` limma sort+block timing from 2.72s to 2.35s without changing concordance.
- `splitting()` now returns its assembled output with `DataFrame(..., copy=False)`, removing the final full-width materialization copy and improving representative `scp_large` limma sort+block timing from 2.38s to 2.07s.
- The split path now reuses one float64 ndarray across affiliation spotting and extraction, removing a duplicate whole-frame conversion and improving representative `scp_large` limma sort+block timing from 2.24s to 1.96s.
- Direct-assembly and column-cache refactors improved pipeline timings: medium limma 0.15s->0.09s, medium m1 0.69s->0.63s, scp_small limma 0.16s->0.13s, scp_small m1 0.36s->0.32s.
- Sorted block runs now avoid a full final unsort copy, improving representative `scp_large` timings from 5.87s to 5.57s for ComBat mode 1 and from 3.87s to 3.66s for limma.

## [0.3.0] - 2026-05-27

### Added

- NaN handling: per-feature Beta.NA computation in ComBat and limma engines, matching R sva v3.60.0. NaN-shaped sub-matrices pass through to engines; each feature independently omits NaN from OLS, variance, and EB computations.
- NaN-safe iterative (`_it_sol`) and non-parametric (`_int_eprior`) EB solvers using per-gene non-NA counts.
- Per-cell NaN R concordance fixtures and murine medulloblastoma real-data verification. All modes concordant at max_rel 0.0003.
- FEATURE_PARITY.md with complete R vs Python difference inventory.
- DIA real-proteomics dataset verification: 8470 proteins, 2 batches, perfect concordance at machine epsilon.

### Changed

- Engine NaN handling: replaced row-dropping with per-feature Beta.NA path. Dense path unchanged.
- `splitting.py`: removed column-dropping logic. Per-cell NaN flows through to engines.
- `validation.py`: NaN allowed in engine inputs; validation checks only structural properties.
- Test NaN invariants: now verify correct NaN handling rather than asserting NaN-free input.
- Code cleanup: removed redundant copies, consolidated duplicated helpers, vectorized block assignment.
- Benchmark runner: R memory measurement and expanded concordance metrics.

### Fixed

- `_it_sol`: `t2_n_g_hat` uses original `g_hat` per iteration (not updated `g_new`), matching R's `postmean` formula. Prevents divergence with NaN data.
- `_int_eprior`: likelihood normalization uses per-gene `n_i` instead of fixed batch size. Enables correct non-parametric EB with per-cell NaN.
- Benchmark data generator: minimum 4 samples per batch to satisfy needed_values=2.
- Benchmark runner: unique tagged filenames per scenario, R memory measurement via `/usr/bin/time -v`.
- Dry-run mode: pipeline DEBUG suppressed during plan output.

### Performance

Modes 1/2 run in ~1.3-1.7s on 8470x36 data. Modes 3/4 dominate at ~8.5s (80% spent in `_int_eprior` broadcast loop). Targets: binomial formula for `_int_eprior` (est. 2.8x on modes 3/4), grouped NaN-pattern solving for per-feature OLS/variance.

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
