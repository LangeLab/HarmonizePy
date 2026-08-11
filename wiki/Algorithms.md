# Algorithms

HarmonizePy provides two batch-correction algorithms: **ComBat** and **limma**. Both are implemented in pure NumPy and handle per-cell missing values without imputation.

---

## ComBat

ComBat (Johnson, Li & Rabinovic 2007) is an empirical Bayes method. It models each feature as having a batch-specific location shift and scale change, then uses information shared across all features to estimate and remove those effects. The shared-information step makes ComBat robust even when individual batches are small.

### The four modes

| Mode | `par_prior` | `mean_only` | What it corrects | Use case |
| --- | --- | --- | --- | --- |
| 1 | `True` | `False` | Location and scale | Default. Most datasets. |
| 2 | `True` | `True` | Location only | Pre-scaled or variance-normalized data |
| 3 | `False` | `False` | Location and scale | Small batches (< ~10 samples), non-Gaussian data |
| 4 | `False` | `True` | Location only | Small batches, pre-scaled data |

**Mode 1** (parametric, location + scale) is the default and appropriate for most bulk proteomics, metabolomics, and microarray datasets where batch sizes are reasonable (10+ samples per batch).

**Mode 2** (parametric, location only) skips variance correction. Use it when the data has already been variance-normalized or when you want to preserve the measured spread between batches.

**Mode 3** (non-parametric, location + scale) makes no distributional assumption about batch effects. It is slower than mode 1 but more appropriate when batch sizes are small, when the data deviates strongly from normality, or when modes 1 and 2 produce overcorrection artifacts.

**Mode 4** (non-parametric, location only) combines the robustness of mode 3 with the location-only correction of mode 2.

### Parametric vs non-parametric

The parametric modes (1, 2) fit a gamma-inverse-gamma hyperprior to the batch effect distribution across features and use it to shrink per-feature estimates. This makes them data-efficient but sensitive to violations of the assumed distribution.

The non-parametric modes (3, 4) estimate the prior empirically by integrating over the actual observed distribution. They are more flexible but require more samples to stabilize the estimates and are notably slower on large datasets.

### The `needed_values` parameter

`needed_values` controls how many non-missing observations a feature must have in a given batch to be included in adjustment for that batch.

The default (`None`) auto-selects:

- `2` for modes 1, 3 and limma (need variance estimate: at least 2 observations)
- `1` for modes 2, 4 (location only: 1 observation suffices)

Increase `needed_values` (e.g. to `3`) if you want stricter inclusion and have enough data to afford it.

---

## limma

limma batch correction (`removeBatchEffect`, Ritchie et al. 2015) uses a linear model. It fits batch as a covariate, estimates batch coefficients by ordinary least squares with a sum-to-zero constraint, and subtracts those coefficients from the data.

limma is:

- Faster than ComBat (closed-form solution, no iterative EB solver)
- Less aggressive: it corrects location only (no scale correction)
- Fully closed-form: results are determined directly from the data, not from an iterative solver

Use limma when:

- Speed is important and the dataset is large
- A single-feature correction is needed (ComBat's EB prior requires multiple features)
- The data has already been variance-stabilized
- You want a lightweight check before running full ComBat

---

## ComBat vs limma: choosing

Start with ComBat mode 1. It is the most common choice in the literature and covers most bulk proteomics and microarray workflows.

Switch to limma when you need speed on a large matrix or when mode 1 produces overcorrection (visible as reversed batch grouping in PCA after correction).

Switch to ComBat mode 3 when:

- Batch sizes are fewer than ~10 samples
- QC plots show strongly non-Gaussian batch effects after mode 1

---

## NaN handling in both algorithms

Both ComBat and limma handle per-cell missing values (stochastic NaN within qualifying batches) using a per-feature approach. For each feature, the computation uses only the observed values in that feature's valid samples. NaN positions in the input remain NaN in the output.

This matches the behavior of R `sva::ComBat` v3.60.0 (the `Beta.NA` approach). See [[R-Compatibility]] for verification details and [[Pipeline]] for how structural missingness is handled before the engines run.
