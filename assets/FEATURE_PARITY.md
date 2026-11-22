# Feature Parity: R HarmonizR v1.10.0 vs HarmonizePy v0.4.0

References: the HarmonizR v1.10.0 R sources (13 files), sva 3.60.0, and limma, from Bioconductor 3.23.

---

## Current Validation Boundaries

Besides the dense and structural-missingness fixtures, the repository has dedicated checks for per-cell NaN handling, medium `jaccard` and `seriation` sort+block cases, chain-rescue `unique_removal` toggles, and the combined stress case (`sort + block + ur + per-cell NaN`).

Limits:

- Jaccard and seriation are fixture-backed and unit-tested, but exact algorithm matching to R is intentionally out of scope.
- Combined-stress validation is fixture-backed, but exact NaN-position parity on every feature is not a release gate: the sort order and the unique-removal rule differ from R (sections 3 and 4), which changes the groups of some features.

---

## 1. Core Algorithms

### 1.1 Dense Path and NaN Path: BOTH VERIFIED

All four ComBat modes and limma are concordant with R on both dense synthetic data and real murine data with per-cell NaN. NaN handling uses per-feature computation (Beta.NA per-feature OLS), NOT row dropping or column dropping.

| Mode                                 | Dense rtol | Murine max_rel | Murine NaN match |
| ------------------------------------ | ---------- | -------------- | ---------------- |
| ComBat 1 (parametric, loc+scale)     | 2e-5       | 0.0003         | YES              |
| ComBat 2 (parametric, loc only)      | 1e-9       | 0.0000         | \*               |
| ComBat 3 (non-parametric, loc+scale) | 5e-4       | 0.0000         | YES              |
| ComBat 4 (non-parametric, loc only)  | 1e-9       | 0.0000         | \*               |
| limma removeBatchEffect              | 1e-9       | 0.0000         | YES              |
| ref_batch, auto mean_only            | identical  | identical      | Code review      |

\*Modes 2 and 4: HarmonizR returns 870 of the shared features entirely missing although the input has values for them; HarmonizePy corrects them. HarmonizR also drops 229 features that HarmonizePy keeps: 221 with one value in a single batch, kept raw, and 8 empty rows, kept missing. On the cells both return, the values agree to 7e-09.

### 1.2 NaN Handling: Per-Feature Beta.NA (VERIFIED ON REAL DATA)

R `sva::ComBat` v3.60.0 handles per-cell NaN via per-feature computation, NOT by dropping rows. The `Beta.NA` function:

```r
Beta.NA <- function(y, X) {
    des <- X[!is.na(y), ]
    y1 <- y[!is.na(y)]
    B <- solve(crossprod(des), crossprod(des, y1))
    B
}
```

HarmonizePy implements the equivalent per-feature NaN handling across all computation steps:

- **B.hat**: R uses `apply(dat, 1, Beta.NA, design)`. Python uses `_beta_na_grouped_batch_design`, the same per-feature batch means, computed for features with the same pattern of observed cells together.
- **var.pooled**: R uses `rowVars(residuals, na.rm=TRUE)`. Python uses `_row_var_nan_grouped(residuals)` per feature.
- **gamma.hat**: R uses `apply(s.data, 1, Beta.NA, batch.design)`. Python uses `_beta_na_grouped_batch_design` on the standardised data.
- **delta.hat**: R uses `rowVars(s.data[, i], na.rm=TRUE)` per batch. Python uses `_row_var_nan_grouped(s_data[:, idx])` per batch.
- **EB solver (parametric)**: R's `postvar`/`postmean` handle NaN per-feature via `n()` in postvar. Python's `_it_sol_batches` uses `np.nansum` and each feature's observed count, solving all batches of a group in one loop.
- **EB solver (non-parametric)**: R's `int.eprior` uses `x <- sdat[i, !is.na(sdat[i, ])]` and `n <- length(x)` per gene. Python's `_int_eprior` uses each feature's observed count `n_i` in the likelihood.
- **NaN in output**: missing input cells stay missing in both.

R does not drop rows with NaN in sva 3.60.0; it handles NaN per feature throughout, and so does HarmonizePy.

**Murine dataset results** (4753 features, 25 samples, 4 batches, 49% missing):

- All 5 methods agree on the cells both return, to 3e-04 relative for ComBat mode 1, whose solver stops at a tolerance, and to 4e-11 or better for the others.
- Missing cells match on all shared features in limma and ComBat modes 1 and 3. In modes 2 and 4, HarmonizR returns 870 features entirely missing that HarmonizePy corrects.
- The 274 features present only in the Python output in modes 1 and 3 have no batch with two values; HarmonizePy keeps them as missing rows. In modes 2 and 4, which need one value per batch, the 229 such features are 221 with values in a single batch, kept raw, and 8 empty rows.

---

## 2. Pipeline Steps

### 2.1 Structural Missingness Detection (Spotting)

R uses `original_batches_existence_counter` to count non-NA per batch within each block. Python uses `notna[:, idx].sum(axis=1)` with vectorized batch indices. Equivalent. R also has a redundant `value_existence_counter` at the block level and a log-only `sum_counter`. Python omits both. Zero impact on output. `needed_values` threshold: 2 for modes 1/3/limma, 1 for modes 2/4 (R fixed, Python same defaults but user-overridable). Output: list of integer vectors (R) vs integer tuples (Python). Equivalent.

### 2.2 Sub-Matrix Extraction

R selects rows by sorted integer affiliation index and columns by `which(block_list %in% vec_in_affil)`. Python selects rows by pre-computed row index array and columns by `np.isin(block_arr, affil)`. Equivalent. Both allow per-cell NaN to pass through to the engines. `splitting.py` does NOT filter columns or rows based on NaN presence.

### 2.3 Adjustment Per Sub-Matrix

- Single-batch sub-df: both pass through raw values. Equivalent.
- Single-feature sub-df: **R drops, Python pass-through** with either engine (KNOWN DIVERGENCE).
- Constant feature (zero variance within a batch): both leave it out of the ComBat fit and return its raw values, with different rules. See section 5.
- Multi-batch with per-cell NaN: both use per-feature computation (Beta.NA approach). Murine verified at max_rel 0.0003.
- Multi-batch, no NaN: both use the vectorized path.
- **Unblocked `medium` benchmark**: 109 of 4968 shared features differ with limma and 135 with ComBat mode 1, 77 of them in their missing cells. Counted as features with a relative difference above 1e-4 or a different missing pattern. The cause is the unique-removal rule (section 4): 99 are features with a pattern of their own that HarmonizePy keeps raw and HarmonizR moves into another feature's pattern, cropping their values, and the rest are in groups whose size those moves change. With `block=2`, ComBat agrees to 7e-06.
- **Features HarmonizR returns entirely missing**: in some modes HarmonizR returns features entirely missing although the input has values for them: 1256 of 3000 on `scp_small` in modes 3 and 4, and 870 on `murine` in modes 2 and 4. HarmonizePy corrects them; on the cells both return, the values agree to 5e-15 and 7e-09.

### 2.4 Reassembly

R uses `plyr::rbind.fill` (align by row name). Python uses pre-allocated array + `pd.concat`. Equivalent for same feature sets. Dropped features: R removes from output entirely, Python keeps as all-NaN (intentional divergence).

---

## 3. Sorting Strategies

**Sparsity sort**: R `find_na()` counts the missing cells of each batch and orders ascending. Python counts, per batch, the features present with at least `needed_values` values and orders descending. The two rank batches differently: on 200 generated cases with contiguous batch labels the orders agreed in 11, and in all 200 when R's missing-cell count is used instead, so the difference comes from the count. **Known difference.**

**Jaccard sort**: R pairs the most similar batches first and then chains the pairs. Python builds a nearest-neighbour chain. The orders agreed in 54 of the same 200 cases. **Known difference.**

**Seriation sort**: R uses `seriation::seriate(binary_df, margin = 2)` (hierarchical clustering with optimal leaf ordering). Python orders batches by the first principal component of their presence vectors. The orders agreed in 24 of 400 cases. **Known difference.**

Sorting changes the output only together with blocking. The orders are not changed to match R: no measurement shows that one gives better corrections than another.

**Batch label renumbering**: R renumbers batch labels to 1,2,3... in first-appearance order after sort. Python keeps original IDs. Block structure is identical (boundaries detected by value change, not absolute label).

**Batch count method**: R uses `tail(batch_list, n=1)` (fragile for non-contiguous labels without sort). Python uses `len(np.unique(batch_list))` (always correct).

---

## 4. Unique-Removal Algorithm: MINOR DIVERGENCE

Both move a feature whose pattern is its own into the longest pattern of another feature that lies inside its own, including a pattern held by a single feature, which then keeps it. They differ in three ways:

- HarmonizR walks the features in row order and takes the first of equally long candidates. HarmonizePy takes the features longest pattern first, and among equally long candidates the one more features hold, then the smallest, so the result does not depend on the row order.
- With no candidate, HarmonizR empties the feature's pattern, and the feature comes out missing. HarmonizePy leaves it unchanged, with its raw values.
- HarmonizePy never moves a feature into a pattern that spans a single batch, whose group is left raw anyway; HarmonizR does.

For example, `(1,2,3,4), (1,2), (5,6), (5,6)` becomes `(1,2), (1,2), (5,6), (5,6)` in both. A known-truth simulation with 20 batches put the error of the corrected values at about 0.2 under either rule, against 0.44 to 0.78 for a rule that takes only patterns two features share; HarmonizR's emptying removed 40 to 130 of 2000 features per run. The `chain_rescue` fixtures cover both `ur=True` and `ur=False`.

---

## 5. Feature Retention Policy: INTENTIONAL DIVERGENCE

- **Single-feature group (1 feature, 2+ batches)**: R drops entirely. Python pass-through raw, with ComBat and with limma.
- **Feature with no value in a whole batch, engines called directly**: `sva::ComBat` 3.60.0 stops with a singular-matrix error, and `limma::removeBatchEffect` sets the coefficient it cannot estimate to zero. `combat()` and `remove_batch_effect()` fit such a feature on the batches it has: the unobserved batch is left out of its grand mean, of the batch priors, and of the limma contrasts. Because ComBat's batch priors are shared, such a feature also enters the priors of the batches it is observed in and moves the other features slightly. Both engines leave raw a feature with values in a single batch, and limma also a feature whose values do not outnumber its observed batches; with `ref_batch`, ComBat leaves raw a feature with no value in the reference batch. limma's level for such a feature is the unweighted mean of its observed batch means; R gives the same when an interior batch is missing, but a different level when the first or last batch is missing, because it sets the coefficients it cannot estimate to zero. `harmonize()` never sends such a feature to an engine, so its output is unaffected.
- **Constant feature in ComBat**: `sva::ComBat` 3.60.0 leaves out rows whose variance is exactly zero inside any batch with two or more samples, in every mode, and returns them raw. Its test uses `var()` without `na.rm`, so a row with a missing cell is never left out. Python leaves out, in every mode, a row whose pooled variance is below its 1e-12 floor (with `ref_batch`, the reference batch's variance), and in mode 1 also a row with a batch variance below 1e-12 of the pooled variance. The rule covers rows with missing cells and near-constant rows. A row that is constant in some batches only is corrected in modes 2 to 4: location-only mode never divides by one batch's variance, and in the non-parametric modes the floored batch gives the row no weight in the other features' priors.
- **Single-batch block (2+ features, 1 batch)**: Both pass-through raw.
- **Empty affiliation (no qualifying batches)**: R drops from rebuild. Python keeps as all-NaN.

Python retains more features in all cases. The trade-off: retained features are not batch-corrected.

---

## 6. Input/Output Differences

- **TSV reader**: R uses `read.table(sep="\t", row.names=1, check.names=FALSE, comment.char="", stringsAsFactors=FALSE)`. Python uses `pd.read_csv(sep="\t", index_col=0)`. Same format.
- **Empty rows and columns**: R uses `janitor::remove_empty(which=c("rows","cols"))`, which removes both. Python keeps both: an empty row stays an all-missing row and an empty column stays in the output.
- **Numeric coercion**: R uses `vapply(as.numeric)` which silently creates NA for non-numeric. Python uses pandas type inference and raises ValueError on non-numeric columns. R silently corrupts; Python rejects.
- **Duplicate rows**: R uses `unique(main_data)`, which compares values and ignores row names, so a feature whose values match another feature's is removed silently. Python removes a row only when its name and its values both repeat, logs the removal, and raises on a repeated name with different values.
- **Default output**: R writes `cured_data.tsv`. Python writes `<stem>_corrected.tsv` next to the input, or the extension of `--output-format`.
- **Visualization**: R applies `2^` before plotting (assumes log2). Not supported in Python.
- **S4 SummarizedExperiment**: R supports. Not applicable to Python.

---

## 7. Verification Status

Tolerance of the R concordance tests per scenario; `--` means no R fixture.

| Scenario                    | ComBat 1 to 4         | limma              |
| --------------------------- | --------------------- | ------------------ |
| Dense synthetic             | rtol 2e-5 to 5e-4     | rtol 1e-9          |
| Structural missingness      | rtol 5e-4             | rtol 1e-8          |
| Blocking (block 2, 4)       | rtol 5e-4             | rtol 1e-8          |
| Sparsity sort + block       | rtol 5e-4             | rtol 1e-8          |
| Jaccard sort + block        | rtol 5e-4             | --                 |
| Seriation sort + block      | rtol 5e-4             | --                 |
| Per-cell NaN synthetic      | max_rel below 1e-4    | max_rel below 1e-4 |
| Murine unblocked            | max_rel 3e-4          | max_rel 2e-10      |
| Murine block 2              | max_rel 6e-6 (mode 1) | --                 |
| Unique-removal chain rescue | rtol 1e-4             | --                 |
| Combined stress             | rtol 1e-4             | --                 |

- All scenarios pass. Missing cells must match except where noted.
- Jaccard and seriation have one `medium` block 2 fixture each, plus unit tests.
- Murine: missing cells match in modes 1 and 3 and limma; see section 1.1 for modes 2 and 4.
- Combined stress compares values on features whose missing cells match, and allows a minority to differ (sections 3 and 4).

---

## 8. Python-Only Improvements

- A per-feature correction report (`return_report=True`, `--report`): each feature's status, the batches of its group, its group size, whether unique removal cropped it, and the repeats removed
- CLI with `--dry-run`, `--config`, `--summary`, `--json`, `--report`
- `InputScaleWarning` when the data do not look log-transformed
- Input checks that name the problem: infinite values, repeated sample names, repeated feature names with different values, and batch labels that are not whole numbers
- Multiple output formats (TSV, CSV, Parquet)
- `needed_values` exposed as user parameter
- Shell tab-completion via `argcomplete`
- Centralized input validation (`validation.py`)
- Parquet input and output
- Self-contained seriation via NumPy SVD
- Module-level logging
- Rebuild into one pre-allocated output array
- Robust batch count via `len(unique())`

## 9. R-Only Features

- S4 SummarizedExperiment input/output
- Diagnostic plotting (sample means, feature means, CV) with `2^` log2 assumption
- Multi-core execution via doParallel + foreach
- Raw R matrix input

---

## 10. Complete Difference Inventory (29 Items)

### Affects Output Values

1. **Single-feature groups**: R drops entirely. Python pass-through with either engine. Causes EB prior shift.
2. **Features returned missing**: in some modes R returns features entirely missing although the input has values (section 2.3). Python corrects them.
3. **Empty affiliation**: R drops from output entirely. Python keeps as all-NaN.
4. **Unique-removal order and empty targets**: R takes features in row order and empties a feature with no target; Python takes them longest pattern first, keeps such a feature raw, and never targets a one-batch pattern; see section 4.
5. **Jaccard sort**: pair-first-then-chain vs nearest-neighbour chain. Different orderings for 3+ batches.
6. **Seriation sort**: `seriation::seriate` vs SVD/PC1 projection. Different orderings.
7. **Sparsity sort**: R counts missing cells per batch; Python counts the features present with at least `needed_values` values. Orders agreed in 11 of 200 cases.
8. **Constant feature in ComBat**: R leaves out rows with zero variance in any batch, in every mode. Python leaves out rows whose variance its engine would floor, by the rules in section 5; in modes 2 to 4 a row constant in some batches only is corrected.
9. **Flat hyper-prior**: when a batch's scale estimates have zero variance across features, R's moment estimates of the inverse-gamma prior divide by zero. Python uses a flat prior, `(1, 1)`.

### Affects I/O or Robustness

1. **Batch label renumbering**: R renumbers after sort. Python keeps original IDs. No impact on block structure.
2. **Batch count**: R uses `tail()` (fragile). Python uses `len(unique())` (robust).
   - **Description columns**: HarmonizR 1.10.0 reads the sample IDs from column 1 and the batches from column 3 by position, converting batches with `as.numeric`. Python uses the columns named `ID` and `batch` when present and falls back to those positions, and rejects batch labels that are not whole numbers.
3. **Empty rows and columns**: R `janitor::remove_empty` removes both. Python keeps both.
4. **Numeric coercion**: R `vapply(as.numeric)` silently creates NA. Python raises ValueError on non-numeric.
5. **Duplicate rows**: R `unique()` compares values only. Python requires the name and the values to repeat.
6. **Column name handling**: R `check.names=FALSE` preserves special chars. Python default may mangle them. Minor.
7. **Comment character**: R `comment.char=""` disables comment handling. Python default. No impact.
8. **String factors**: R `stringsAsFactors=FALSE`. Python no factor concept. R-only.
9. **Default output**: R `cured_data.tsv`. HarmonizePy `<stem>_corrected.tsv` next to the input, or the extension of `--output-format`.

### Zero Impact on Output

1. **Redundant block-level counter**: R has `value_existence_counter` per block. Python omits. Per-batch check is sufficient.
2. **Log-only counter**: R has `sum_counter` for console message only. Python omits. Never feeds algorithm.
3. **Visualization log2**: R applies `2^` before plotting. Python not supported. Purely diagnostic.
4. **S4 support**: R has SummarizedExperiment I/O. Not applicable to Python.
5. **Parallel execution**: R uses doParallel + foreach. HarmonizePy runs on one thread.
6. **Rebuild method**: R uses `rbind.fill`. Python uses pre-allocated array. Same output for same features.
7. **Diagnostic plotting**: R has boxplots. Not supported in Python. Purely cosmetic.
8. **var.pooled formula (dense)**: Both use `mean(residuals^2)`. Same.
9. **var.pooled formula (NaN)**: R uses `rowVars(residuals, na.rm=TRUE)`. Python uses `_row_var_nan_grouped`. Same ddof=1 formula. As in R, one missing cell anywhere switches every feature to this formula.
10. **NaN-safe `_it_sol`**: R handles per-feature NaN via `n()` in postvar. Python uses per-feature `n_per_gene`. Same approach.
11. **NaN-safe `_int_eprior`**: R uses `x <- sdat[i, !is.na(sdat[i, ])]`. Python uses per-gene non-NA count `n_i`. Same approach.
