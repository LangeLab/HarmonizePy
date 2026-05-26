<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <img src="assets/logo_readme.svg" alt="HarmonizePy" width="420">
</p>

<p align="center">
  Pure-Python batch-effect harmonization toolkit validated against R <code>sva::ComBat</code>, <code>limma::removeBatchEffect</code>, and <code>HarmonizR</code>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12--3.14-2D7D46?style=flat-square&logo=python&logoColor=white" alt="Python 3.12-3.14">
  <img src="https://img.shields.io/badge/version-0.2.0-8B5CF6?style=flat-square" alt="v0.2.0">
  <img src="https://img.shields.io/badge/status-alpha-C17D10?style=flat-square" alt="Alpha">
  <img src="https://github.com/LangeLab/HarmonizePy/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/badge/576%20tests-passing-22C55E?style=flat-square" alt="576 tests">
  <img src="https://codecov.io/gh/LangeLab/HarmonizePy/branch/main/graph/badge.svg" alt="Coverage">
  <img src="https://img.shields.io/badge/license-GPL--3.0-4B9D6E?style=flat-square" alt="GPL-3.0">
</p>

<p align="center">
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-CHANGELOG-E05D44?style=flat-square" alt="Changelog"></a>
  <a href="CITATION.cff"><img src="https://img.shields.io/badge/cite-CITATION.cff-0066CC?style=flat-square" alt="Citation"></a>
  <a href="https://github.com/LangeLab/HarmonizePy/issues"><img src="https://img.shields.io/badge/issues-GitHub-8B5CF6?style=flat-square" alt="Issues"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/docs-README-0F766E?style=flat-square" alt="Docs"></a>
</p>

HarmonizePy provides batch-effect correction for omics data with structural missingness. Standard ComBat and limma require complete matrices. The high-level `harmonize()` entry point handles this automatically: it detects which features are absent in which batches, dissects the input into NaN-free sub-matrices, adjusts each independently, and reassembles the result. For dense, clean data the low-level `combat()` and `remove_batch_effect()` engines are also exposed directly. HarmonizePy reimplements the full HarmonizR v1.10.0 pipeline in pure Python with no R dependency at runtime, validated against R reference output with 576 tests.

## Capabilities

**ComBat** (Johnson et al. 2007). All four modes:

- Mode 1: parametric, location + scale
- Mode 2: parametric, location only
- Mode 3: non-parametric, location + scale
- Mode 4: non-parametric, location only

**limma** (Ritchie et al. 2015). Linear-model batch correction via sum-to-zero contrasts.

**HarmonizR-compatible pipeline (feature parity with v1.10.0):**

- Structural missingness handled automatically: the input matrix is dissected into NaN-free sub-matrices, each adjusted independently, then reassembled.
- Batch sorting: reorder batches so similar ones become neighbours. Three strategies: `"sparsity"` (completeness-based, fastest), `"jaccard"` (pairwise feature-overlap similarity), `"seriation"` (PCA-based ordering, more accurate for many batches).
- Batch blocking: group `block=N` consecutive (optionally sorted) batches into a single sub-matrix, reducing peak memory and improving adjustment quality for datasets with many batches.
- Unique-combination removal (`unique_removal=True`, default): features whose batch-presence pattern is unique across the dataset get rescued. They would otherwise form a singleton sub-matrix and be dropped. The algorithm crops their affiliation to the nearest shared pattern, trading some non-missing data for inclusion in a group adjustment.

**Validation against R HarmonizR v1.10.0:** all core algorithms are concordant at machine epsilon for unblocked modes (mode 1/3/limma). With blocking, feature retention may differ: R drops single-feature groups entirely, while HarmonizePy passes them through unchanged to preserve more data. The pipeline logs at INFO level report how many features were corrected versus passed through. See `assets/FEATURE_PARITY.md` for a detailed comparison.

## Installation

```bash
# PyPI (once published)
pip install harmonizepy

# From GitHub
pip install git+https://github.com/LangeLab/HarmonizePy.git

# Development install with uv
uv python install 3.12.13
uv sync --dev --python 3.12.13

# Optional extras
pip install harmonizepy[config]      # YAML/TOML support for --config flag
pip install harmonizepy[completion]  # Shell tab-completion via argcomplete
```

### Dependencies

- Python >= 3.12
- numpy >= 1.24, pandas >= 2.0
- pytest >= 8.0 (dev)

## Quick start

`harmonize()` returns a `pandas.DataFrame` with the same shape and column order as the input. Features are rows (DataFrame index), samples are columns.

```python
import pandas as pd
from harmonizepy import harmonize

# From file paths (returns a DataFrame)
result = harmonize("data.tsv", "description.csv")
# result is a pd.DataFrame, features as index, samples as columns

# From DataFrames
data_df = pd.read_csv("data.tsv", sep="\t", index_col=0)
desc_df = pd.read_csv("description.csv")
result = harmonize(data_df, desc_df, algorithm="limma", sort="sparsity", block=2)

# Low-level API (NaN-free ndarray in, ndarray out)
from harmonizepy import combat, remove_batch_effect

corrected = combat(data_matrix, batch_labels, par_prior=True, mean_only=False)
corrected = remove_batch_effect(data_matrix, batch_labels)
```

### Input format

- **data**: features × samples matrix. For TSV/CSV files, the first column must contain feature names (read as the DataFrame index). When passing a DataFrame directly, feature names must be in `DataFrame.index` and sample names in `DataFrame.columns`.
- **description**: CSV/DataFrame with columns `ID` (sample names matching data columns), `sample` (integer), `batch` (integer batch label).

### Parameter reference

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `algorithm` | `"ComBat"` or `"limma"` | `"ComBat"` | Batch correction algorithm |
| `combat_mode` | 1, 2, 3, 4 | 1 | ComBat variant (ignored for limma) |
| `needed_values` | int or `None` | `None` (auto) | Min non-missing values per batch for a feature to be included. Auto sets 2 for modes 1, 3 and limma; 1 for modes 2, 4. |
| `sort` | `"sparsity"`, `"jaccard"`, `"seriation"`, or `None` | `None` | Batch sorting strategy before blocking |
| `block` | int >= 2 or `None` | `None` | Group N consecutive batches into one sub-matrix block |
| `unique_removal` | bool | `True` | Rescue singleton features by cropping to nearest shared pattern |
| `config` | `HarmonizeConfig` or `None` | `None` | Reproducible run configuration (overrides individual kwargs) |
| `output_file` | str, Path, or `None` | `None` | Write corrected matrix to this TSV path |

### Choosing algorithm, mode, and strategy

Start with the default (`ComBat`, mode 1, no sort/block). It handles most datasets well.

Use `algorithm="limma"` when you need a fast, moderate correction or have a single feature. It uses a closed-form linear model with no iterative solver.

For very small batch sizes (n < 10 per batch), prefer `combat_mode=3` (non-parametric). When variance correction is unnecessary (e.g. pre-scaled data), use mode 2 or 4.

For datasets with 5 or more batches, adding `sort="sparsity"` and `block=2` improves both runtime and adjustment quality by grouping similar-completeness batches together. Use `sort="jaccard"` when batch sizes are uneven: it groups by feature-overlap similarity rather than raw completeness counts.

Leave `unique_removal=True` (the default) unless you need strict missingness-pattern matching. It keeps more features in the output by rescuing singletons at minimal cost.

## CLI usage

```bash
harmonizepy data.tsv batch.csv -o corrected.tsv

# Algorithm and mode
harmonizepy data.tsv batch.csv --algorithm limma -o corrected.tsv
harmonizepy data.tsv batch.csv --combat-mode 3 -o corrected.tsv

# Sort, block, and feature-handling flags
harmonizepy data.tsv batch.csv --sort sparsity --block 2 -o corrected.tsv
harmonizepy data.tsv batch.csv --needed-values 3 --no-unique-removal -o corrected.tsv

# Config file (TOML, JSON, or YAML with [config] extra)
harmonizepy data.tsv batch.csv --config run.toml

# Dry-run: validates inputs and prints the run plan, exits without computing
harmonizepy data.tsv batch.csv --dry-run
# Prints:
#   HarmonizePy 0.2.0 dry run
#   ────────────────────────────────────────────────────
#   Features:        1500
#   Samples:         45
#   Batches:         5
#   Sub-matrices:    3  (unique affiliation groups)
#   Algorithm:        ComBat mode 1
#   Sort strategy:    none
#   Block size:       none
#   Unique removal:   enabled
#   Inputs valid. Use without --dry-run to run correction.

# Reproducible run with JSON summary
harmonizepy data.tsv batch.csv -o corrected.tsv --summary run.json --json
```

## Running tests

```bash
uv run pytest               # all 576 tests (R concordance auto-skips if fixtures absent)
uv run pytest tests/ -v     # verbose
```

The test suite covers:

- R concordance against `sva::ComBat`, `limma::removeBatchEffect`, and `HarmonizR` (blocking, sort+block, `ur` toggle)
- Edge cases: unbalanced batches, minimal dimensions, extreme values, near-constant features, negative data, many batches, singleton batches, sparse missingness
- Failure modes: invalid inputs, NaN rejection, dimension mismatches
- Numerical stability: determinism, float32 promotion, memory isolation
- CLI integration: flag parsing, config files, dry-run, output formats (TSV, CSV, Feather)

### Regenerating R fixtures

Requires R with `sva`, `limma`, `HarmonizR`, and `seriation` (managed via `renv`):

```bash
Rscript tests/fixtures/generate_r_fixtures.R
Rscript tests/fixtures/generate_edgecase_fixtures.R
Rscript tests/fixtures/generate_blocking_fixtures.R
```

## Project layout

```text
src/harmonizepy/
    __init__.py          # Public API: harmonize, combat, remove_batch_effect
    __main__.py          # CLI entry point
    core.py              # Pipeline orchestrator (harmonize)
    types.py             # Shared data structures (HarmonizeConfig)
    validation.py        # Centralised input validation
    io.py                # TSV/CSV/Feather read/write
    affiliation.py       # Per-feature batch affiliation, UR logic
    sorting.py           # Batch sorting strategies (sparsity, jaccard, seriation)
    blocking.py          # Batch blocking (build_block_list)
    combat.py            # Pure NumPy ComBat engine
    combat_wrapper.py    # Mode dispatch (1-4) wrapper
    limma_wrapper.py     # Pure NumPy limma::removeBatchEffect
    splitting.py         # Sub-frame extraction, adjustment, reassembly
tests/
    test_smoke.py        # Version check
    test_api.py          # Public API surface tests
    test_combat.py       # ComBat unit + R concordance
    test_limma.py        # limma unit + R concordance
    test_pipeline.py     # Full pipeline integration tests
    test_sorting.py      # Sorting strategies unit tests
    test_blocking.py     # Blocking unit tests
    test_unique_removal.py  # Unique-combination removal unit tests
    test_splitting.py    # Splitting unit + NaN audit tests
    test_io.py           # I/O unit tests
    test_validation.py   # Validation unit tests
    test_cli.py          # CLI end-to-end tests
    test_comprehensive.py   # Edge-case, failure-mode, extended R concordance
    fixtures/            # R-generated reference outputs (88 TSV + 12 CSV)
data/                    # Small showcase datasets (user-provided)
```

## License

HarmonizePy is licensed under [GPL-3.0](LICENSE).

## References

### Manuscripts

- Johnson WE, Li C, Rabinovic A. "Adjusting batch effects in microarray expression data using empirical Bayes methods." *Biostatistics* 8(1):118-127, 2007.
- Ritchie ME et al. "limma powers differential expression analyses for RNA-sequencing and microarray studies." *Nucleic Acids Research* 43(7):e47, 2015.
- Voß H et al. "HarmonizR enables data harmonization across independent proteomic datasets with appropriate handling of missing values." *Nature Communications* 13:3523, 2022. <https://doi.org/10.1038/s41467-022-31007-x>
- Schlumbohm S, Neumann JE, Neumann P. "HarmonizR: blocking and singular feature data adjustment improve runtime efficiency and data preservation." *BMC Bioinformatics*, 2025. <https://doi.org/10.1186/s12859-025-06073-9>

### Code and packages

- HarmonizR R package (Bioconductor): <https://www.bioconductor.org/packages/release/bioc/html/HarmonizR.html>
- HarmonizR source (GitHub): <https://github.com/HSU-HPC/HarmonizR>

The ComBat algorithm was introduced by Johnson, Li & Rabinovic (2007). `limma::removeBatchEffect` was published by Ritchie et al. (2015). The HarmonizR pipeline (structural-missingness dissection, batch sorting, blocking, unique-combination removal) was developed by Voß et al. (2022) and Schlumbohm, Neumann & Neumann (2025). HarmonizePy is not a line-by-line port of the R sources: the engine logic was reimplemented from the published manuscripts and cross-validated against R reference output.
