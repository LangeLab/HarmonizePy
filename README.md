# HarmonizePy

Pure-Python batch-effect harmonization toolkit, validated against R `sva::ComBat`, `limma::removeBatchEffect`, and `HarmonizR`.

## Features

- **ComBat** - Empirical Bayes batch correction (Johnson et al. 2007), all 4 modes:
    - Mode 1: parametric, location + scale
    - Mode 2: parametric, location only
    - Mode 3: non-parametric, location + scale
    - Mode 4: non-parametric, location only
- **limma** - Linear-model batch correction via sum-to-zero contrasts (Ritchie et al. 2015)
- **HarmonizR-compatible pipeline** - handles structural missingness (features absent in some batches) via automatic splitting, adjustment, and reassembly
- Pure NumPy/SciPy - no R dependency at runtime
- Validated against R reference implementations with 226+ tests

## Quick start

```python
from harmonizepy import harmonize

# From file paths
result = harmonize("data.tsv", "description.csv", algorithm="ComBat", combat_mode=1)

# From DataFrames
result = harmonize(data_df, desc_df, algorithm="limma")

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

- R concordance tests against `sva::ComBat`, `limma::removeBatchEffect`, and `HarmonizR`
- Edge cases: unbalanced batches, minimal dimensions, extreme values, near-constant features, negative data, many batches, singleton batches, sparse missingness
- Failure modes: invalid inputs, NaN rejection, dimension mismatches
- Numerical stability: determinism, float32 promotion, memory isolation

### Regenerating R fixtures

Requires R with `sva`, `limma`, and `HarmonizR` installed (managed via `renv`):

```bash
Rscript tests/fixtures/generate_r_fixtures.R
Rscript tests/fixtures/generate_edgecase_fixtures.R
```

## Project layout

```text
src/harmonizepy/
    __init__.py          # Public API: harmonize, combat, remove_batch_effect
    core.py              # Pipeline entry point (harmonize)
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
    test_pipeline.py     # Full pipeline + spotting/splitting tests
    test_comprehensive.py # Edge-case, failure-mode, and extended R concordance
    fixtures/            # R-generated reference outputs (70+ TSV/CSV files)
data/                    # Small showcase datasets (user-provided)
temp/                    # Full-scale datasets (gitignored)
plan/                    # Development planning notes (gitignored)
```

## Dependencies

- Python >= 3.12
- numpy, pandas, scipy
- pytest >= 8.0 (dev)

## References

- Johnson WE, Li C, Rabinovic A. "Adjusting batch effects in microarray expression data using empirical Bayes methods." *Biostatistics* 8(1):118-127, 2007.
- Ritchie ME et al. "limma powers differential expression analyses for RNA-sequencing and microarray studies." *Nucleic Acids Research* 43(7):e47, 2015.
- Voß H et al. "HarmonizR enables data harmonization across independent proteomic datasets with appropriate handling of missing values." *Nature Communications* 13:3523, 2022.
