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
- **Benchmark suite**: `benchmarks/generate_data.py` for synthetic data, `benchmarks/run_benchmarks.py` for timing/memory/R comparison, `benchmarks/template_run.R` for R HarmonizR integration. Generates human-readable `benchmarks/RESULTS.md` with per-scenario tables and data specifications.
- **Pass-through logging**: single-feature and single-batch groups that skip correction are now logged at INFO level (total count) and DEBUG level (individual feature names).
- **Corrected vs pass-through reporting**: benchmark results table now shows corrected and pass-through feature counts alongside timing and memory.
- **SCP benchmark datasets**: `scp_small` (3000x1000, 20 batches, 50% missing) and `scp_large` (5000x10000, 100 batches, 60% missing) added to benchmark suite. Run Python-only (no R comparison). SCP missingness uses abundance-dependent detection profiles so features form real correction groups (scp_small: 3000/3000 corrected, scp_large: 5000/5000 corrected).
- **Data specifications table**: automatically generated in RESULTS.md showing features, samples, batches, missingness, and file size per dataset.
- **Pipeline timing parsing**: benchmark runner now uses the internal "Done" timing from the CLI for more accurate algorithm performance measurement, excluding IO overhead.

### Changed

- **Console logging**: now writes to stderr instead of stdout, preventing log messages from polluting JSON or dry-run output.
- **Root logger**: `harmonizepy` logger configured with `propagate=False` to prevent double-logging when external code configures the root handler.
- **LICENSE** (`LICENSE`): GPL-3.0 with individual copyright holder and acknowledgments for original R HarmonizR.
- **plan.md**: phase markers updated to reflect completed and partial items.
- **_int_eprior (`combat.py`)**: replaced per-iteration 2-D broadcast (`np.square(x - g).sum()`) with binomial formula (`sum_x2_i - 2*g*sum_x_i + n*g^2`), pre-computed `log(d)` to eliminate redundant log operations, and reused the boolean mask instead of allocating a new one per iteration. Large mode 3: 19.2s to 8.8s (2.2x faster).
- **_it_sol (`combat.py`)**: same binomial formula optimization applied to the iterative solver's residual sum-of-squares computation.
- **build_affiliation_list (`affiliation.py`)**: replaced per-feature Python row loop with vectorized per-block column sums across all features simultaneously. Large mode 1: 0.70s to 0.10s (7x faster).
- **Benchmark result ordering**: limma rows now appear first in the Python Performance table, followed by ComBat modes in ascending order.
- **Benchmark expansion**: all four ComBat modes (1-4) are now tested with block=2 and sort+block combinations.
- **Benchmark timing**: switched from total wall-clock to pipeline-internal "Done" timing for fair comparison across dataset sizes.
- **IO layer** (`io.py`, `__main__.py`): replaced feather with parquet as the default output format. Parquet writes are 12x faster and 2x smaller than TSV on scp_large. Added `pyarrow` as optional dependency (`harmonizepy[io]`). CLI defaults to `.parquet` output when pyarrow is installed, falls back to `.tsv`. Removed feather support.
- **PyArrow CSV engine auto-detect** (`io.py`): reads the first line of a CSV/TSV file to count columns (microsecond cost). Uses pyarrow engine when columns <= 500 (tall/narrow data, 5-10x faster), falls back to C parser for wider files.
- **splitting.py refactor**: replaced per-group DataFrame creation with numpy-only hot loop. Pre-converts data to ndarray once, uses `np.ix_` for sub-matrix extraction. scp_small: 2.2x faster, scp_large: 1.4x faster.

### Fixed

- `benchmarks/generate_data.py`: Zipfian batch size distribution created batches with 1 sample, making them unable to satisfy `needed_values=2`. Increased minimum to 4 samples per batch.
- `benchmarks/run_benchmarks.py`: output filenames did not include mode/block/sort parameters, causing overwrites between different benchmark scenarios. Fixed to use unique tagged filenames.
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
