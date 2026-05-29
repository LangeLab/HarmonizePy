<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <img src="https://raw.githubusercontent.com/LangeLab/HarmonizePy/main/assets/logo_readme.svg" alt="HarmonizePy" width="420">
</p>

<p align="center">
  Pure-Python batch-effect harmonization toolkit validated against R <code>sva::ComBat</code>, <code>limma::removeBatchEffect</code>, and <code>HarmonizR</code>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12--3.14-2D7D46?style=flat-square&logo=python&logoColor=white" alt="Python 3.12-3.14">
  <img src="https://img.shields.io/badge/version-0.3.2-8B5CF6?style=flat-square" alt="v0.3.2">
  <img src="https://img.shields.io/badge/status-alpha-C17D10?style=flat-square" alt="Alpha">
  <img src="https://github.com/LangeLab/HarmonizePy/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/badge/628%20tests-collected-22C55E?style=flat-square" alt="628 tests collected">
  <img src="https://codecov.io/gh/LangeLab/HarmonizePy/branch/main/graph/badge.svg" alt="Coverage">
  <img src="https://img.shields.io/badge/license-GPL--3.0-4B9D6E?style=flat-square" alt="GPL-3.0">
</p>

<p align="center">
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-CHANGELOG-E05D44?style=flat-square" alt="Changelog"></a>
  <a href="CITATION.cff"><img src="https://img.shields.io/badge/cite-CITATION.cff-0066CC?style=flat-square" alt="Citation"></a>
  <a href="https://github.com/LangeLab/HarmonizePy/issues"><img src="https://img.shields.io/badge/issues-GitHub-8B5CF6?style=flat-square" alt="Issues"></a>
  <a href="https://github.com/LangeLab/HarmonizePy/wiki"><img src="https://img.shields.io/badge/docs-Wiki-0F766E?style=flat-square" alt="Docs"></a>
</p>

HarmonizePy is a pure-Python batch-effect correction package for omics data. The high-level `harmonize()` entry point handles structural missingness automatically: features are grouped by observed batch support, corrected with ComBat or limma within compatible subsets, and reassembled without imputation. No R installation is required at runtime.

## Documentation

- [Installation](https://github.com/LangeLab/HarmonizePy/wiki/Installation) - install options, optional extras
- [Quick Start](https://github.com/LangeLab/HarmonizePy/wiki/Quick-Start) - minimal working examples
- [Input Format](https://github.com/LangeLab/HarmonizePy/wiki/Input-Format) - data matrix and batch description spec
- [Algorithms](https://github.com/LangeLab/HarmonizePy/wiki/Algorithms) - ComBat modes 1-4 and limma, when to use each
- [Pipeline](https://github.com/LangeLab/HarmonizePy/wiki/Pipeline) - structural missingness, sorting, blocking, unique removal
- [API Reference](https://github.com/LangeLab/HarmonizePy/wiki/API-Reference) - full function signatures
- [CLI Reference](https://github.com/LangeLab/HarmonizePy/wiki/CLI-Reference) - all flags, config files, output formats
- [R Compatibility](https://github.com/LangeLab/HarmonizePy/wiki/R-Compatibility) - validation scope and numerical tolerances
- [Benchmarks](https://github.com/LangeLab/HarmonizePy/wiki/Benchmarks) - dataset catalog and runtime results

## Capabilities

**ComBat** (Johnson et al. 2007). All four modes:

- Mode 1: parametric, location + scale
- Mode 2: parametric, location only
- Mode 3: non-parametric, location + scale
- Mode 4: non-parametric, location only

**limma** (Ritchie et al. 2015). Linear-model batch correction via sum-to-zero contrasts.

**HarmonizR-compatible pipeline (feature parity with v1.10.0):**

- Structural missingness handled automatically: features are grouped by observed batch support, corrected in compatible subsets, and reassembled into the original matrix shape.
- Batch sorting: reorder batches so similar ones become neighbours. Three strategies: `"sparsity"` (completeness-based, fastest), `"jaccard"` (pairwise feature-overlap similarity), `"seriation"` (PCA-based ordering, more accurate for many batches).
- Batch blocking: group `block=N` consecutive (optionally sorted) batches into a single sub-matrix, reducing peak memory and improving adjustment quality for datasets with many batches.
- Unique-combination removal (`unique_removal=True`, default): features whose batch-presence pattern is unique across the dataset get rescued. They would otherwise form a singleton sub-matrix and be dropped. The algorithm crops their affiliation to the nearest shared pattern, trading some non-missing data for inclusion in a group adjustment.

**Validation against R reference workflows:** the package is tested against `sva::ComBat`, `limma::removeBatchEffect`, and HarmonizR v1.10.0 reference outputs. Known edge-case differences in retention policy are documented in the [R Compatibility](https://github.com/LangeLab/HarmonizePy/wiki/R-Compatibility) wiki page. Runtime benchmarks across six datasets are in [Benchmarks](https://github.com/LangeLab/HarmonizePy/wiki/Benchmarks).

## Installation

```bash
# Package install (if published for this release)
pip install harmonizepy

# From GitHub
pip install git+https://github.com/LangeLab/HarmonizePy.git

# Development install with uv
uv python install 3.12.13
uv sync --dev --python 3.12.13

# Optional extras
pip install harmonizepy[config]      # YAML/TOML support for --config flag
pip install harmonizepy[completion]  # Shell tab-completion via argcomplete
pip install harmonizepy[io]          # Parquet IO and pyarrow-accelerated CSV/TSV paths
```

See [Installation](https://github.com/LangeLab/HarmonizePy/wiki/Installation) for the full dependency list and optional extras detail.

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

# Low-level API (ndarray in, ndarray out)
from harmonizepy import combat, remove_batch_effect

corrected = combat(data_matrix, batch_labels, par_prior=True, mean_only=False)
corrected = remove_batch_effect(data_matrix, batch_labels)
```

The low-level engines handle missing observations per feature and preserve NaN positions in the returned array.

### Input format

- **data**: features × samples matrix. For TSV/CSV files, the first column must contain feature names (read as the DataFrame index). When passing a DataFrame directly, feature names must be in `DataFrame.index` and sample names in `DataFrame.columns`.
- **description**: CSV/DataFrame with columns `ID` (sample names matching data columns), `sample` (integer), `batch` (integer batch label).

### Parameter reference

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `algorithm` | `"ComBat"` or `"limma"` | `"ComBat"` | Batch correction algorithm |
| `combat_mode` | 1, 2, 3, 4 | 1 | ComBat variant (ignored for limma) |
| `sort` | `"sparsity"`, `"jaccard"`, `"seriation"`, or `None` | `None` | Batch sorting strategy before blocking |
| `block` | int >= 2 or `None` | `None` | Group N consecutive batches into one sub-matrix block |
| `unique_removal` | bool | `True` | Rescue singleton features by cropping to nearest shared pattern |

Full signature with all parameters is in the [API Reference](https://github.com/LangeLab/HarmonizePy/wiki/API-Reference).

### Choosing algorithm and strategy

Start with the default (`ComBat`, mode 1). For very small batch sizes (n < 10 per batch), prefer `combat_mode=3` (non-parametric). For datasets with 5 or more batches, `sort="sparsity"` and `block=2` group similar batches together before correction. Use `sort="jaccard"` when batch sizes are uneven. See [Algorithms](https://github.com/LangeLab/HarmonizePy/wiki/Algorithms) and [Pipeline](https://github.com/LangeLab/HarmonizePy/wiki/Pipeline) for full guidance.

## CLI usage

```bash
# Basic
harmonizepy data.tsv batch.csv -o corrected.tsv

# Algorithm, sort, and block
harmonizepy data.tsv batch.csv --algorithm limma -o corrected.tsv
harmonizepy data.tsv batch.csv --combat-mode 3 --sort sparsity --block 2 -o corrected.tsv

# Validate inputs without running correction
harmonizepy data.tsv batch.csv --dry-run

# Config file (TOML, JSON, or YAML with [config] extra)
harmonizepy data.tsv batch.csv --config run.toml
```

When `-o` is omitted, the output is written as `<data_stem>_corrected.tsv` next to the input file. See [CLI Reference](https://github.com/LangeLab/HarmonizePy/wiki/CLI-Reference) for all flags, output formats, and config file syntax.

## Running tests

```bash
uv run pytest               # full test suite
uv run pytest tests/ -v     # verbose
```

The suite covers R concordance, edge cases, failure modes, numerical stability, CLI integration, and the benchmark harness. R fixture outputs are committed to the repository, so the full concordance suite runs on a plain checkout without a live R environment. Regenerating fixtures from source requires R with `sva`, `limma`, `HarmonizR`, and `seriation` managed via `renv`.

## Project layout

```text
src/harmonizepy/    # Package source: pipeline, engines, CLI, I/O
tests/              # 628 tests: unit, integration, R concordance, CLI
benchmarks/         # Benchmark CLI, dataset catalog, results
data/               # Small showcase datasets
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

The ComBat algorithm was introduced by Johnson, Li & Rabinovic (2007). `limma::removeBatchEffect` was published by Ritchie et al. (2015). The HarmonizR pipeline (structural-missingness handling, batch sorting, blocking, unique-combination removal) was developed by Voß et al. (2022) and Schlumbohm, Neumann & Neumann (2025). HarmonizePy is not a line-by-line port of the R sources: the engine logic was reimplemented from the published manuscripts and validated against R reference output.
