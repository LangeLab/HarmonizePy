# Algorithms

HarmonizePy provides two batch-correction algorithms: **ComBat** and **limma**. Both are implemented in pure NumPy and handle per-cell missing values without imputation.

---

## ComBat

ComBat (Johnson, Li & Rabinovic 2007) is an empirical Bayes method. It models each feature as having a batch-specific location shift and scale change, then uses information shared across all features to estimate and remove those effects. The shared step shrinks each feature's estimates towards those of the other features, which steadies them when batches are small.

### The four modes

| Mode | `par_prior` | `mean_only` | What it corrects | Use case |
| --- | --- | --- | --- | --- |
| 1 | `True` | `False` | Location and scale | Default. Most datasets. |
| 2 | `True` | `True` | Location only | Pre-scaled or variance-normalized data |
| 3 | `False` | `False` | Location and scale | Small batches (< ~10 samples), non-Gaussian data |
| 4 | `False` | `True` | Location only | Small batches, pre-scaled data |

**Mode 1** (parametric, location + scale) is the default.

**Mode 2** (parametric, location only) skips variance correction. Use it when the data has already been variance-normalized or when you want to preserve the measured spread between batches.

**Mode 3** (non-parametric, location + scale) makes no distributional assumption about the batch effects across features: each feature's estimate is a likelihood-weighted average of the other features' estimates. Its cost grows with the square of the number of features in a group. On a known-truth simulation with heavy-tailed batch shifts it removed them less well than mode 1, and R `sva::ComBat` gives the same result, so a heavy tail is not by itself a reason to choose it.

**Mode 4** (non-parametric, location only) combines the non-parametric prior of mode 3 with the location-only correction of mode 2.

### Parametric vs non-parametric

The parametric modes (1, 2) place a normal prior on the batch location effects and an inverse-gamma prior on the batch scale effects, with hyper-parameters estimated across features, and shrink each feature's estimates towards them.

The non-parametric modes (3, 4) use the other features' estimates themselves as the prior, weighted by how well each explains the feature's observed values. Their cost grows with the square of the number of features in a group.

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

- Location only: it removes each feature's batch means and leaves the scale unchanged
- Without shrinkage: each feature is fitted on its own
- Closed-form: one least-squares fit per feature, no iterative solver

Use limma when you want each feature's batch means removed with no borrowing across features, for example as a simple comparison with ComBat.

---

## ComBat vs limma: choosing

Start with ComBat mode 1, the default. Compare with limma, which does not borrow across features, and with mode 2, which leaves the scale unchanged, when the corrected data look over-corrected, for example reversed batch grouping in a PCA after correction. No measurement here ranks the modes in general.

---

## NaN handling in both algorithms

Both ComBat and limma handle per-cell missing values (stochastic NaN within qualifying batches) using a per-feature approach. For each feature, the computation uses only the observed values in that feature's valid samples. NaN positions in the input remain NaN in the output.

This matches the behavior of R `sva::ComBat` v3.60.0 (the `Beta.NA` approach). See [[R-Compatibility]] for verification details and [[Pipeline]] for how structural missingness is handled before the engines run.
