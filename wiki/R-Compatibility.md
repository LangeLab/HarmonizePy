# R Compatibility

HarmonizePy is validated against three R reference implementations:

- `sva::ComBat` (Bioconductor sva v3.60.0)
- `limma::removeBatchEffect` (Bioconductor limma)
- `HarmonizR` v1.10.0 (Bioconductor)

This page describes what is validated, the numerical tolerances achieved, and the known differences between HarmonizePy and the R implementations.

---

## Validation scope

The parity validation covers:

- All four ComBat modes and limma on dense synthetic data
- Per-cell NaN handling on real murine medulloblastoma data (4753 features, 25 samples, 4 batches, 49% missing)
- DIA proteomics data (8470 features, 36 samples, 2 batches)
- Full pipeline: blocking, sort+block, `unique_removal` toggle, and the combined stress case (sort + block + unique_removal + per-cell NaN)
- Synthetic datasets at small, medium, and SCP-scale

---

## Concordance table

| Method | Dense tolerance | Murine real data (max_rel) | NaN positions match |
| --- | --- | --- | --- |
| ComBat mode 1 (parametric, loc+scale) | 2e-5 | 0.0003 | Yes |
| ComBat mode 2 (parametric, loc only) | 1e-9 | 0.0000 | Note 1 |
| ComBat mode 3 (non-parametric, loc+scale) | 5e-4 | 0.0000 | Yes |
| ComBat mode 4 (non-parametric, loc only) | 1e-9 | 0.0000 | Note 1 |
| limma removeBatchEffect | 1e-9 | 0.0000 | Yes |

**Note 1:** Modes 2 and 4 with R: R drops 229 additional features due to an interaction between `mean_only=TRUE` and single-feature groups. HarmonizePy retains those features as all-NaN rows. On the shared subset of features, values match perfectly (0.0000).

---

## Known behavioral differences

The following differences are intentional. None affect the corrected values for features that both implementations process.

### Single-feature groups

A feature with a batch-presence pattern that appears only once (singleton group) has no other features to share the ComBat EB prior with. R drops the feature entirely. HarmonizePy keeps it with its original (uncorrected) values, with ComBat and with limma.

**When this occurs:** only when exactly 1 feature has a given batch-presence pattern and that pattern involves 2 or more batches. On the murine dataset (4753 features), this never occurs. On a medium synthetic dataset (5000 features), it affects roughly 32 of 4968 features.

**Practical effect:** the affected features retain their original batch effects. They appear in the output and can be filtered downstream if needed. The correction of all other features is unaffected.

### Constant features in ComBat

A feature whose values do not vary inside a batch has zero variance there, and in the parametric modes it would set the empirical-Bayes prior for every other feature. `sva::ComBat` 3.60.0 leaves such rows out of the fit and returns them unadjusted, and so does HarmonizePy. The other features are corrected as if the constant row were absent.

The two rules differ:

- sva leaves out a row whose variance is exactly zero in any batch with two or more samples. HarmonizePy leaves out a row whose variance its engine would raise to the 1e-12 floor: a pooled within-batch variance below 1e-12 (with `ref_batch`, the variance in the reference batch), or a batch variance below 1e-12 of the pooled variance. Near-constant rows are therefore left out too.
- sva's test ignores rows with a missing cell; with such a row, `sva::ComBat` stops with an error. HarmonizePy applies its rule to rows with missing cells.
- In location-only mode (modes 2 and 4, or any batch with a single sample), HarmonizePy does not apply the batch rule, because that mode never divides by one batch's variance. A row that is constant in one batch only is corrected.
- sva applies its rule in every mode. HarmonizePy applies it in the parametric modes only.

### All-NaN rows for unqualified features

Features with no qualifying batch (below the `needed_values` threshold in every batch) appear as all-NaN rows in the HarmonizePy output. R omits these features from the output entirely, producing a matrix with fewer rows.

**Practical effect:** the output matrix shape differs by the number of such features. The HarmonizePy convention produces a predictable output shape that matches the input regardless of data quality.

### Input validation strictness

R silently coerces non-numeric cells to NA and silently removes rows whose values repeat another row. HarmonizePy raises a `ValueError` for non-numeric cells. A row is removed only when both its name and its values repeat another row, and the removal is logged; a repeated name with different values raises a `ValueError`. Rows with identical values under different names are both kept. This is intentional: silent coercion or removal can mask data preparation errors.

### Sorting algorithm differences

All three sorting strategies can order batches differently from R. Sparsity counts the features present in each batch where R counts missing cells; jaccard builds a nearest-neighbour chain where R pairs the most similar batches first; seriation orders batches by the first principal component of their presence vectors where R uses hierarchical clustering with optimal leaf ordering (`seriation::seriate`). On generated test cases with contiguous batch labels, the orders agreed with R in 11 of 200 cases (sparsity), 54 of 200 (jaccard), and 24 of 400 (seriation). Sparsity agreed in all 200 when R's count of missing cells was used, so its difference comes from counting present features instead.

Sorting changes the output only together with blocking. The strategies are not required to match R.

### Default output filename

R writes `cured_data.tsv`. HarmonizePy writes `<data_stem>_corrected.tsv` next to the input file.

---

## Verifying against your own R results

If you want to compare HarmonizePy output against R on your own data, the simplest approach is to run the same data through R `HarmonizR::harmonizR()` and compare the outputs feature by feature.

A typical comparison in Python:

```python
import pandas as pd
import numpy as np

py_result = pd.read_csv("harmonizepy_output.tsv", sep="\t", index_col=0)
r_result  = pd.read_csv("harmonizr_output.tsv",   sep="\t", index_col=0)

# Align on shared features and samples
shared = py_result.index.intersection(r_result.index)
py = py_result.loc[shared]
r  = r_result.loc[shared]

# Max relative difference on non-NaN cells
rel_diff = np.abs(py.values - r.values) / (np.abs(r.values) + 1e-10)
print(f"Max relative difference: {np.nanmax(rel_diff):.2e}")
print(f"Features only in Python output: {len(py_result) - len(shared)}")
print(f"Features only in R output: {len(r_result) - len(shared)}")
```

A `max_rel` below `5e-4` on ComBat mode 1 (and below `1e-9` on modes 2, 3, 4 and limma) is concordant with the reference validation results. Features present only in the Python output are the single-feature pass-through and all-NaN retention cases described above.
