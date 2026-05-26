# Feature Parity: R HarmonizR vs HarmonizePy

Reference: R HarmonizR v1.10.0 (Bioconductor 3.23), HarmonizePy v0.2.0.
R API sourced from the HarmonizR Vignette (2026-04-28).

## Core Algorithms

| Feature                                        | R HarmonizR                                  | HarmonizePy                                | Notes                   |
| ---------------------------------------------- | -------------------------------------------- | ------------------------------------------ | ----------------------- |
| ComBat mode 1 (parametric, location+scale)     | sva::ComBat par.prior=TRUE, mean.only=FALSE  | combat.py par_prior=True, mean_only=False  | R concordance validated |
| ComBat mode 2 (parametric, location only)      | sva::ComBat par.prior=TRUE, mean.only=TRUE   | combat.py par_prior=True, mean_only=True   | R concordance validated |
| ComBat mode 3 (non-parametric, location+scale) | sva::ComBat par.prior=FALSE, mean.only=FALSE | combat.py par_prior=False, mean_only=False | R concordance validated |
| ComBat mode 4 (non-parametric, location only)  | sva::ComBat par.prior=FALSE, mean.only=TRUE  | combat.py par_prior=False, mean_only=True  | R concordance validated |
| limma removeBatchEffect                        | limma::removeBatchEffect                     | limma_wrapper.py remove_batch_effect       | R concordance validated |
| Reference batch support                        | ComBat ref.batch parameter                   | combat.py ref_batch param                  | Single reference batch  |
| Automatic mean_only for singleton batches      | sva::ComBat internal                         | combat.py:306 forces mean_only=True        | Mirrors R behaviour     |

## Pipeline Parameters

| Parameter        | R name               | R values                                          | Python name             | Python values                      | Notes                               |
| ---------------- | -------------------- | ------------------------------------------------- | ----------------------- | ---------------------------------- | ----------------------------------- |
| Algorithm        | algorithm            | "ComBat" (default), "limma"                       | algorithm               | "ComBat" (default), "limma"        | Identical                           |
| ComBat mode      | ComBat_mode          | 1 (default), 2, 3, 4                              | combat_mode             | 1 (default), 2, 3, 4               | Identical                           |
| Sort strategy    | sort                 | "sparcity_sort", "jaccard_sort", "seriation_sort" | sort                    | "sparsity", "jaccard", "seriation" | R names use underscore suffix       |
| Block size       | block                | integer                                           | block                   | integer                            | Identical                           |
| Unique removal   | ur                   | TRUE (default), FALSE                             | unique_removal          | True (default), False              | Identical, default on               |
| needed_values    | Not exposed in R API | N/A                                               | needed_values           | int or None (auto)                 | Python-only; R uses fixed threshold |
| Output file      | output_file          | string, default "cured_data"                      | output_file             | str/Path or None                   | Python: no default filename         |
| Plot diagnostics | plot                 | "samplemeans", "featuremeans", "CV", or off       | Not supported           | N/A                                | Python-only feature gap             |
| Verbosity        | verbosity            | integer 0+, default 1                             | --verbose/--quiet flags | debug/info/warning                 | Different mechanism                 |
| Parallel cores   | cores                | integer, default all available                    | Not supported           | N/A                                | Python-only feature gap             |

## Missing Data Pipeline

| Feature                          | R HarmonizR                  | HarmonizePy                                | Notes                                |
| -------------------------------- | ---------------------------- | ------------------------------------------ | ------------------------------------ |
| Structural missingness detection | Internal spotting            | affiliation.py build_affiliation_list      | Same logic                           |
| Group by affiliation             | unique() on affiliation list | affiliation.py reduce_to_unique_groups     | Same logic                           |
| Sub-matrix extraction            | Internal splitting           | splitting.py splitting                     | NaN-free extraction                  |
| Adjustment per sub-matrix        | ComBat or limma              | Adjust via combat_wrapper or limma_wrapper | Same dispatch                        |
| Matrix reassembly                | Internal rebuild             | pd.concat in core.py                       | Single pre-allocated array in Python |

## Input / Output

| Feature                 | R HarmonizR                            | HarmonizePy                               | Notes                         |
| ----------------------- | -------------------------------------- | ----------------------------------------- | ----------------------------- |
| TSV data input          | read with row.names=1                  | io.py read_main_data                      | Same format                   |
| CSV description input   | read.csv                               | io.py read_description                    | Same format                   |
| Feather input           | Not supported                          | io.py lines 49-54                         | Python-only                   |
| DataFrame input         | Supported (data.frame)                 | core.py accepts pd.DataFrame              | Both accept in-memory objects |
| S4 SummarizedExperiment | Supported                              | Not applicable                            | R-specific S4 class           |
| Matrix input            | Supported                              | Not directly                              | Python requires DataFrame     |
| Path input              | Character string                       | str or Path                               | Identical                     |
| TSV output              | write.table (default "cured_data.tsv") | io.py write_output                        | Both tab-separated            |
| CSV output              | Not direct                             | __main__.py --output-format csv           | Python-only                   |
| Feather output          | Not supported                          | __main__.py --output-format feather       | Python-only                   |
| Config file             | Not supported                          | __main__.py _load_config (JSON/TOML/YAML) | Python-only                   |
| Dry-run mode            | Not supported                          | __main__.py --dry-run                     | Python-only                   |
| JSON run summary        | Not supported                          | __main__.py --summary, --json             | Python-only                   |
| Shell completion        | Not supported                          | argcomplete in __main__.py                | Python-only                   |

## Sorting Strategies

| Strategy        | R name           | R dependency      | Python name | Python dependency | Notes                    |
| --------------- | ---------------- | ----------------- | ----------- | ----------------- | ------------------------ |
| Sparsity-based  | "sparcity_sort"  | Internal          | "sparsity"  | NumPy             | Descending completeness  |
| Jaccard-based   | "jaccard_sort"   | Internal          | "jaccard"   | NumPy             | Greedy nearest-neighbour |
| Seriation-based | "seriation_sort" | seriation package | "seriation" | NumPy SVD         | PCA-based ordering       |

## Compute Model

__Language and runtime:__

- R HarmonizR requires R 4.2+ and depends on Bioconductor (sva, limma) and CRAN (seriation, doParallel, janitor, plyr).
- HarmonizePy requires Python 3.10+ with only NumPy and pandas. No R dependency at runtime. The correction algorithms are pure NumPy reimplementations, not wrappers around R.

__Parallel processing:__

- R uses `doParallel` + `foreach` for multi-core execution. The `cores` parameter controls thread count.
- HarmonizePy has no parallel processing yet. All pipeline steps run single-threaded.

__External dependencies:__

- R depends on the `seriation` package for PCA-based seriation sorting.
- HarmonizePy computes seriation via NumPy SVD directly, with no external sort library.

__Input validation:__

- R HarmonizR has minimal input validation, relying on R's native type system.
- HarmonizePy has a dedicated `validation.py` module with centralised error messages covering all public API functions.

__CLI and tooling:__

- R HarmonizR is a library-only package with no command-line interface. Usage requires writing an R script.
- HarmonizePy provides a full CLI via `harmonizepy` command (argparse): positional arguments, flags, config files, dry-run mode, JSON summary output, and shell completion.

__Testing:__

- R uses `testthat` with limited unit test coverage.
- HarmonizePy uses `pytest` with 400+ tests, including R-concordance validation against generated fixtures.

---

## Feature Retention Policy

R HarmonizR drops features that form single-feature affiliation groups (groups with exactly one feature and two or more batches). The `splitting()` function returns an empty list for these groups, removing the feature from the output entirely.

HarmonizePy passes single-feature groups through unchanged (raw values are copied to the output without adjustment). This preserves data that R loses. The trade-off is that passed-through features retain their original batch effects.

When blocking is used, single-batch blocks (blocks containing only one batch) also pass through unchanged in both implementations, but R may additionally drop samples from blocks that were excluded.

The pipeline logs at INFO level report how many features passed through without correction. A DEBUG-level log identifies individual feature names.

### Available in R HarmonizR only (continued)

- R dependencies: `sva`, `limma`, `seriation`, `doParallel`, `foreach`, `janitor`, `plyr`, `SummarizedExperiment`
- Sorting via `seriation` package
- Matrix input support directly (not just DataFrame)

### Available in HarmonizePy only

- CLI with `harmonizepy` command, flags, and positional arguments
- Config file support (JSON, TOML, YAML)
- Dry-run mode to preview pipeline configuration
- JSON run summary output (file and stdout)
- Multiple output formats: TSV, CSV, Feather
- Shell tab-completion via `argcomplete`
- Dedicated input validation module (`validation.py`)
- `needed_values` parameter exposed to users
- Feather format input/output
- Pure Python implementation, no R required
- Pathlib support for file paths
- Self-contained seriation via NumPy SVD (no external library)
- Feature retention policy: passes through single-feature groups (R drops them entirely)
- INFO/DEBUG logging for passed-through features with per-feature identification
- Automatic `.log` file written alongside output with full DEBUG trace
- Module-level logging across all pipeline stages
- Benchmark suite (`benchmarks/`) with timing, memory, feature retention, and R comparison
- 22 invariant tests covering blocked and unblocked modes for spread reduction, NaN propagation, and pass-through behavior

### Available in R HarmonizR only

- S4 `SummarizedExperiment` input/output support
- Diagnostic plotting: sample means, feature means, coefficient of variation
- Configurable verbosity level via parameter
- Parallel multi-core execution via `doParallel` and `foreach`
- Raw R matrix input support
- R-native `data.frame` integration within the Bioconductor ecosystem
