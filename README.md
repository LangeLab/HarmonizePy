<!-- markdownlint-disable MD033 MD041 -->

<div align="center">
  <img src="assets/logo_readme.svg" alt="HarmonizePy" width="420" />
  <br />
  Pure-Python batch-effect harmonization toolkit validated against R `sva::ComBat`, `limma::removeBatchEffect`, and `HarmonizR`.
  <br />
  Version <strong>v0.2.0</strong>. Adds batch sorting, blocking, and unique-combination removal matching full HarmonizR v1.8.0 pipeline.
  <br />
  <br />
  <a href="#running-tests"><img src="https://img.shields.io/badge/tests-411%20passing-22c55e?style=for-the-badge" alt="Tests" /></a>
  <a href="#installation"><img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-3776AB?style=for-the-badge" alt="Python versions" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-f59e0b?style=for-the-badge" alt="License: GPL-3.0" /></a>
  <a href="https://codecov.io/gh/LangeLab/HarmonizePy"><img src="https://img.shields.io/badge/coverage-tracked-6366f1?style=for-the-badge" alt="codecov" /></a>
</div>

## Features

- **ComBat** - Empirical Bayes batch correction (Johnson et al. 2007), all 4 modes:
    - Mode 1: parametric, location + scale
    - Mode 2: parametric, location only
    - Mode 3: non-parametric, location + scale
    - Mode 4: non-parametric, location only
- **limma** - Linear-model batch correction via sum-to-zero contrasts (Ritchie et al. 2015)
- **HarmonizR-compatible pipeline** - full v1.8.0 feature parity:
    - Structural missingness via automatic splitting, adjustment, and reassembly
    - Batch sorting: `"sparsity"`, `"jaccard"`, `"seriation"` strategies
    - Batch blocking: group neighbouring batches to reduce sub-matrix space (`block=2`, etc.)
    - Unique-combination removal: rescue singleton features by cropping to nearest shared pattern (`unique_removal=True`)
- Pure NumPy/SciPy — no R dependency at runtime
- Validated against R reference implementations with 411 tests

## Quick start

```python
from harmonizepy import harmonize

# From file paths
result = harmonize("data.tsv", "description.csv", algorithm="ComBat", combat_mode=1)

# From DataFrames
result = harmonize(data_df, desc_df, algorithm="limma")

# With sorting + blocking + unique removal (mirrors HarmonizR v1.8.0 defaults)
result = harmonize(data_df, desc_df, sort="sparsity", block=2, unique_removal=True)

# Low-level API
from harmonizepy import combat, remove_batch_effect

corrected = combat(data_matrix, batch_labels, par_prior=True, mean_only=False)
corrected = remove_batch_effect(data_matrix, batch_labels)
```

### Input format

- **data**: features × samples matrix (TSV with feature names as first column, or a DataFrame)
- **description**: CSV/DataFrame with columns `ID` (sample names matching data columns), `sample` (integer), `batch` (integer batch label)

## Installation

```bash
# Development install with uv
uv python install 3.12.13
uv sync --dev --python 3.12.13
```

## Running tests

```bash
uv run pytest               # all tests
uv run pytest tests/ -v     # verbose
```

The test suite includes:

- R concordance tests against `sva::ComBat`, `limma::removeBatchEffect`, and `HarmonizR` (blocking, sort+block, `ur` toggle)
- Edge cases: unbalanced batches, minimal dimensions, extreme values, near-constant features, negative data, many batches, singleton batches, sparse missingness
- Failure modes: invalid inputs, NaN rejection, dimension mismatches
- Numerical stability: determinism, float32 promotion, memory isolation

### Regenerating R fixtures

Requires R with `sva`, `limma`, `HarmonizR`, and `seriation` installed (managed via `renv`):

```bash
Rscript tests/fixtures/generate_r_fixtures.R
Rscript tests/fixtures/generate_edgecase_fixtures.R
Rscript tests/fixtures/generate_blocking_fixtures.R
```

## Project layout

```text
src/harmonizepy/
    __init__.py          # Public API: harmonize, combat, remove_batch_effect
    core.py              # Pipeline entry point (harmonize)
    types.py             # Shared data structures (HarmonizeConfig, etc.)
    validation.py        # Centralised input validation
    affiliation.py       # Per-feature batch affiliation, group reduction, unique removal
    sorting.py           # Batch sorting strategies (sparsity, jaccard, seriation)
    blocking.py          # Batch blocking (build_block_list)
    combat.py            # Pure NumPy ComBat engine
    combat_wrapper.py    # Mode dispatch (1-4) wrapper
    limma_wrapper.py     # Pure NumPy limma::removeBatchEffect
    spotting.py          # Missingness classification
    splitting.py         # Group-by-affiliation sub-frame adjustment
    rebuild.py           # Reassemble corrected sub-frames
    io.py                # TSV/CSV read/write
tests/
    test_smoke.py        # Version check
    test_combat.py       # ComBat unit + R concordance tests
    test_limma.py        # limma unit + R concordance tests
    test_api.py          # Public API surface tests
    test_pipeline.py     # Full pipeline + spotting/splitting tests
    test_sorting.py      # Sorting strategies unit tests
    test_blocking.py     # Blocking unit tests
    test_unique_removal.py  # Unique-combination removal unit tests
    test_comprehensive.py   # Edge-case, failure-mode, and extended R concordance
    fixtures/            # R-generated reference outputs (88 TSV + 12 CSV files)
data/                    # Small showcase datasets (user-provided)
temp/                    # Full-scale datasets (gitignored)
plan/                    # Development planning notes (gitignored)
```

## Dependencies

- Python >= 3.10
- numpy, pandas, scipy
- pytest >= 8.0 (dev)

## References

- Johnson WE, Li C, Rabinovic A. "Adjusting batch effects in microarray expression data using empirical Bayes methods." *Biostatistics* 8(1):118-127, 2007.
- Ritchie ME et al. "limma powers differential expression analyses for RNA-sequencing and microarray studies." *Nucleic Acids Research* 43(7):e47, 2015.
- Voß H et al. "HarmonizR enables data harmonization across independent proteomic datasets with appropriate handling of missing values." *Nature Communications* 13:3523, 2022.
- Schlumbohm S et al. "HarmonizR v2 enables advanced data harmonization with improved handling of missing values for multi-batch omics data." *bioRxiv*, 2025.
