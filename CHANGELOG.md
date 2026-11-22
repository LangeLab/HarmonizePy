<!-- markdownlint-disable MD024 -->

# Changelog

All notable changes to HarmonizePy, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

How to read it:

- Each version opens with a summary of what it changes for users. The entries follow under **Added**, **Changed**, **Fixed**, and **Removed**, each led by the area it touches.
- While the major version is 0, a minor version can change corrected values. Every intended change is listed.
- `#N` is the GitHub issue with the discussion. Work on `main` that is not released yet collects under `[Unreleased]`.
- From 0.4.0, a version's section is also the text of its GitHub release.

## [Unreleased]

This release fixes the inputs found to give wrong values, hang, or fail without a clear message, and adds a report that says what happened to each feature. Corrected values change only where a fix below applies; elsewhere they differ from 0.3.2 by rounding, at most 1e-10. Large inputs run faster and need less memory.

### Added

- **Correction report.** `harmonize(..., return_report=True)` returns a `HarmonizeResult` of the corrected matrix and a table with one row per feature: whether it was corrected or kept raw and why, the batches of its group, its group size, whether unique removal cropped it, and how many repeats of it were removed. The CLI writes the same table with `--report PATH`. The matrix is the same with or without the report (#28).
- **Input-scale warning.** `harmonize()` and the dry run warn with `InputScaleWarning` when the values look untransformed (non-negative with a 95th percentile above 100) or look like proportions, and name the transform to apply. The check is a heuristic and can be silenced by category (#25).
- **Releases.** Pushing a `vX.Y.Z` tag publishes a GitHub release once `main`'s CI has passed on the tagged commit: the wheel and sdist that CI built and smoke-tested, their SHA-256 checksums with an attestation, and this file's section as the notes.

### Changed

- **Speed and memory.** ComBat solves all batches of a group together, computes the non-parametric prior in blocks sized for the CPU cache, and checks for features to leave raw in a few array operations per group. Corrected values change by rounding only.
- **One thread.** CSV and TSV files are read with pandas' C parser instead of pyarrow's reader, which started a thread pool, and Parquet is read without reader threads. The `io` extra is needed for Parquet only.
- **Logging.** The per-group ComBat line is at DEBUG level; INFO lines report the number of groups.

### Fixed

- **ComBat statistics.**
  - A feature whose variance the engine would raise to its 1e-12 floor, such as one constant inside a batch, keeps its raw values and no longer sets the empirical-Bayes prior for every other feature (#18). In modes 2 to 4, a feature constant in only some batches is still corrected; `sva::ComBat` leaves it raw.
  - Modes 3 and 4 form each likelihood from the deviations to the feature's own mean, as `sva::ComBat` does. The expanded form lost precision and could put scale estimates off by up to a factor of three.
  - `combat()` and `remove_batch_effect()` fit a feature that has no value in some batch on the batches it was measured in; the unobserved batch counted as a batch effect of zero, so a true level of 10 came back near 7. Features with values in a single batch keep their raw values, as do, in limma, features missing a batch whose values do not outnumber their observed batches. `harmonize()` never sends such a feature to an engine, so this change alone does not alter its output (#27).
- **Features left raw.** A feature alone in its batch pattern keeps its raw values with both engines. ComBat could hang on such a row with a missing cell, and limma corrected it on its own batch means (#19).
- **Unique removal.**
  - Between equally long targets, the one more features hold wins, then the smallest; the choice depended on the order of the rows, so reordering them could change the groups and the corrected values.
- **Engine input.** `combat()` keeps a feature with no value missing instead of making every value missing. Infinite values are rejected with the feature named. All four engine functions require whole-number batch labels; `combat()` merged labels such as 0.2 and 0.9 into one batch. An unknown `ref_batch`, a one-sample reference batch, an unknown algorithm, and a `combat_mode` other than an integer from 1 to 4 are rejected with one message each.
- **Input files.**
  - A whole-number column with a blank cell loads, with the blank as missing (#20).
  - A feature is removed as a repeat only when its name and all its values repeat, and the removal is logged; a name repeated with different values raises a `ValueError`. Files and DataFrames follow the same rule (#21).
  - Rows with no value are kept when a file is read, as for a DataFrame; reading a file dropped them. Repeated sample names raise a `ValueError`.
  - The batch description is read by one rule everywhere: the columns named `ID` and `batch`, otherwise the first and third. A sample listed twice, or a batch label that is not a whole number, raises a `ValueError` that names it (#22).
- **Command line.** `--output-format csv` without `-o` writes a `.csv` file. A dry run writes no files, plans the groups exactly as the run does, and prints the `needed_values` the run uses; `--dry-run` with `--log-file`, `--summary`, `--json`, or `--report` is a usage error (#22). The summary records the resolved `needed_values`; a malformed YAML config and read or write errors give a message instead of a traceback; log lines are not repeated when `main()` runs twice.
- **Pipeline.** The log counts the features unique removal moved and the blocks correctly. `build_block_list()` accepts float labels and counts them as distinct batches; `build_affiliation_list()` checks that labels match the columns.
- **Documentation.** Docstrings, CLI help, the wiki, the README, and the parity document describe what the code does, and state the sort-order and unique-removal differences from HarmonizR with how often the results agree (#24).

## [0.3.2] - 2026-05-29

This release adds a GitHub wiki, a CI workflow that runs the R concordance tests on committed fixtures, and a benchmark summary.

### Added

- **CI.** A GitHub Actions workflow runs a representative subset of the test suite on pushes and pull requests, with the R reference outputs committed so the concordance tests need no R installation.
- **Documentation.** A GitHub wiki of ten pages (installation, quick start, input format, algorithms, pipeline, API, CLI, R compatibility, benchmarks) and a benchmark summary in `benchmarks/RESULTS.md`.

### Changed

- **Command line.** Without `-o`, the output is `<data_stem>_corrected.tsv`; Parquet is written only when requested.
- **README.** The README is shorter and links to the wiki pages.

### Fixed

- **Code quality.** Ruff and mypy pass on the CI target. The install docs name the `io` extra for Parquet.

## [0.3.1] - 2026-05-28

This release adds a benchmark harness that compares against HarmonizR, and speeds up the missing-value paths of both engines.

### Added

- **Benchmarks.** A benchmark command (`run`, `cache-r`, `validity`, `profile`, `report`, `generate-data`) with a scenario registry, cached HarmonizR baselines, concordance checks, and memory measurements for Python and R.

### Changed

- **Speed.** The ComBat missing-value path, the non-parametric prior, the limma missing-value path, the affiliation and sorting helpers, data validation, and output assembly do less repeated work, with concordance unchanged.

### Fixed

- **Benchmarks.** `block` reaches HarmonizR as a number (it fell back to no blocking, so earlier blocked R results were invalid); memory units, R version strings, config paths, mode filters, and R worker clean-up on timeout are corrected.

## [0.3.0] - 2026-05-27

Per-cell missing values reach the engines instead of being dropped: each feature is corrected from its observed cells, as `sva::ComBat` 3.60.0 does, where rows and columns with missing cells were dropped before.

### Added

- **Missing values.** ComBat and limma compute each feature's batch means, variances, and empirical-Bayes estimates from its observed cells (`Beta.NA`), and both solvers use each feature's own count of observed cells.
- **Validation.** R concordance fixtures with per-cell missing values, checks on the murine medulloblastoma and DIA proteomics datasets, and `assets/FEATURE_PARITY.md`, the list of differences from HarmonizR.

### Changed

- **Pipeline.** Sub-matrices keep their missing cells; rows and columns are no longer dropped before correction, and engine input may contain missing values.

### Fixed

- **ComBat solvers.** The parametric solver uses the original batch estimate in each iteration, as R's `postmean` does, which stops it diverging on missing data; the non-parametric prior normalises each likelihood by the feature's own observed count.
- **Command line.** The dry run no longer prints pipeline debug lines.

## [0.2.0] - 2026-04-01

This release builds the HarmonizR pipeline around the engines, after HarmonizR 1.8.0: batch sorting, blocking, and unique-combination removal, with a command line.

### Added

- **Pipeline.** Batch sorting (`sparsity`, `jaccard`, `seriation`), blocking of neighbouring batches (`block=N`), and unique-combination removal (`unique_removal=True`, after R's `ur=TRUE`).
- **Command line.** The `harmonizepy` command with the pipeline options, `--needed-values`, `--dry-run`, `--summary`, `--json`, `--config`, and TSV, CSV, and Feather output.
- **Configuration.** `HarmonizeConfig`, a frozen dataclass for a reproducible run, and one validator for all pipeline arguments.

### Changed

- **Memory.** The split step fills one output array instead of one per group, so peak memory is about the input's size instead of three times it.
- **Input.** Files are read as TSV, CSV, or Feather.

### Removed

- **Engines.** The ComBat engine no longer uses `np.matrix`, and validation repeated across modules is gone.

### Fixed

- **ComBat.** A zero or negative scale estimate no longer gives an infinite log-likelihood or a negative variance, the priors stay finite for zero-variance features, and a one-sample batch forces location-only correction, as `sva::ComBat` does.

## [0.1.0] - 2025-12-15

The first release implements ComBat in its four modes and limma's `removeBatchEffect` in NumPy, with the HarmonizR approach to missing values (split by batch pattern, correct, reassemble), tested against R.

### Added

- **Engines.** ComBat (Johnson, Li & Rabinovic 2007), parametric and non-parametric, with and without scale correction, with `ref_batch`; limma's `removeBatchEffect` with sum-to-zero contrasts (Ritchie et al. 2015).
- **Pipeline.** `harmonize()` groups features by the batches they are observed in, corrects each group, and reassembles the matrix; it accepts DataFrames and TSV or CSV paths.
- **Command line.** A minimal `harmonizepy` command with `-o`.
- **Tests.** Concordance tests against R `sva`, `limma`, and HarmonizR, with the scripts that generate the reference outputs.
