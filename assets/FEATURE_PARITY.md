# Feature Parity: R HarmonizR v1.10.0 vs HarmonizePy v0.2.0

References: R source at `ref/HarmonizR/R/` (13 files). R sva v3.60.0, limma, HarmonizR v1.10.0 via Bioconductor 3.23.

---

## WARNING: Existing Fixture Limitations

All 185 concordance tests use **synthetic data with clean structural missingness** (whole-batch absence, no per-cell NaN within qualifying blocks). They do NOT test:

- Per-cell stochastic NaN within qualifying batches (real proteomics dropout)
- Murine-like data complexity (14.5% per-cell dropout)
- Jaccard/Seriation sort concordance on non-trivial batch count
- Unique-removal chain-rescue differences at scale
- Feature-retention policy impact on real data

**185 tests pass on these synthetic fixtures, but this does not prove concordance on real data.** The murine_medulloblastoma dataset is the only real-data test case and has not been verified end-to-end against R output. See `plan/MASTER_REFERENCE.md` section 11 for outstanding verification items.

---

## 1. Core Algorithms

### 1.1 Dense (NaN-free) Path: VERIFIED

All four ComBat modes and limma on fully-dense data are concordant with R at the documented tolerances:

| Mode                                 | rtol            | atol            | Verification           |
| ------------------------------------ | --------------- | --------------- | ---------------------- |
| ComBat 1 (parametric, loc+scale)     | 2e-5            | 1e-5            | 185 fixture tests pass |
| ComBat 2 (parametric, loc only)      | 1e-9            | 1e-9            | 185 fixture tests pass |
| ComBat 3 (non-parametric, loc+scale) | 5e-4            | 1e-4            | 185 fixture tests pass |
| ComBat 4 (non-parametric, loc only)  | 1e-9            | 1e-9            | 185 fixture tests pass |
| limma removeBatchEffect              | 1e-9            | 1e-9            | 185 fixture tests pass |
| ref_batch, auto mean_only            | identical logic | identical logic | Code review            |

Limitation: these tolerances were established on synthetic data. Real data with pathological value distributions may diverge further.

### 1.2 NaN Handling (Per-Cell Dropout): NOT VERIFIED AGAINST R

Both R and Python drop feature rows with any per-cell NaN (`na.omit` in R `sva::ComBat`; row-detection + exclusion in Python `combat()`). The logic:

1. Sub-matrix is extracted by affiliation group (all columns from qualifying blocks)
2. Per-cell NaN within qualifying batches may be present in the sub-matrix
3. R `sva::ComBat` v3.60.0: `na.rows <- apply(is.na(dat), 1, any); dat <- dat[!na.rows, , drop = FALSE]`
4. Python `combat()`: `nan_rows = np.isnan(data).any(axis=1); clean_data = data[~nan_rows]`

**Outcome**: functionally identical:

- R: rows with any NaN are dropped from adjustment, feature absent from output
- Python: rows with any NaN are excluded from adjustment, feature present as all-NaN in output

**Both lose quantified values** for features with per-cell dropout. This is inherent to `sva::ComBat`'s `na.omit` approach, not a bug in either implementation.

**Not verified**: per-cell NaN fixtures do not exist. The R fixture scripts (`generate_edgecase_fixtures.R`) have been updated to generate `percell_nan_*` datasets but have not been run. Concordance on per-cell NaN data is UNTESTED.

---

## 2. Pipeline Steps

### 2.1 Structural Missingness Detection (Spotting)

| Aspect                  | R                                      | Python                           | Verification            |
| ----------------------- | -------------------------------------- | -------------------------------- | ----------------------- |
| Per-batch non-NA count  | `original_batches_existence_counter`   | `notna[:, idx].sum(axis=1)`      | Code review: equivalent |
| Per-block non-NA count  | `value_existence_counter`              | aggregate of batch checks        | Code review: equivalent |
| needed_values threshold | 2 for modes 1/3/limma, 1 for modes 2/4 | Same defaults (user-overridable) | 185 tests pass          |
| Output                  | list of integer vectors                | list of integer tuples           | Equivalent              |

**Difference 9.4**: R tracks a redundant `value_existence_counter` at the block level that checks the same condition already verified by per-batch counters. Python omits this redundant check. Output is identical.

**Difference 9.5**: R tracks a `sum_counter` that accumulates non-NA counts from excluded blocks for a console message ("Amount of numerical values lost"). This counter does not affect algorithm output. Python does not have this logging.

### 2.2 Group by Affiliation

| Aspect   | R                              | Python                       | Verification |
| -------- | ------------------------------ | ---------------------------- | ------------ |
| Grouping | `unique()` on affiliation list | `reduce_to_unique_groups()`  | Equivalent   |
| Order    | First-appearance order         | Ordered dict insertion order | Equivalent   |

### 2.3 Sub-Matrix Extraction

| Aspect           | R                                                | Python                                | Verification            |
| ---------------- | ------------------------------------------------ | ------------------------------------- | ----------------------- |
| Row selection    | By sorted integer affiliation index              | By pre-computed row index array       | Equivalent              |
| Column selection | All columns where block_list matches affiliation | `np.isin(block_arr, affil)`           | Equivalent              |
| NaN handling     | Per-cell NaN passes through to engine            | Per-cell NaN passes through to engine | Code review: equivalent |

### 2.4 Adjustment Per Sub-Matrix

| Aspect                                             | R                                       | Python                                     | Verification                            |
| -------------------------------------------------- | --------------------------------------- | ------------------------------------------ | --------------------------------------- |
| Single-batch sub-df                                | Pass-through raw data                   | Pass-through raw data                      | Code review: equivalent                 |
| Single-feature sub-df                              | **Dropped entirely** (returns `list()`) | **Pass-through raw values**                | **KNOWN DIVERGENCE** (section 5)        |
| Multi-batch multi-feature sub-df with per-cell NaN | sva::ComBat na.omit drops affected rows | combat() row-detection drops affected rows | Logic matches, not tested with fixtures |
| Multi-batch multi-feature sub-df without NaN       | sva::ComBat adjusted                    | _combat_dense adjusted                     | 185 tests pass                          |

### 2.5 Reassembly

| Aspect           | R                                             | Python                            | Verification                                  |
| ---------------- | --------------------------------------------- | --------------------------------- | --------------------------------------------- |
| Method           | `plyr::rbind.fill` (align by row name)        | Pre-allocated array + `pd.concat` | Code review: equivalent for same feature sets |
| Dropped features | Absent from output (not in rbind.fill result) | Present as all-NaN rows           | **Cosmetic divergence**                       |

**Difference 9.3**: Features with empty affiliation (no qualifying blocks in any batch) are handled differently:

- R `splitting.r`: the empty-affiliation group still creates a sub-df entry via the foreach loop, but with 0 columns. The adjustment loop falls through to `else { return(list()) }`, and the feature is absent from rebuild output.
- Python `splitting.py:106-107`: `if len(affil) == 0: continue` skips adjustment entirely but the pre-allocated output array already has NaN for that row. The feature appears as all-NaN.

Both cases result in no usable data for the feature, but Python preserves the row in the output.

---

## 3. Sorting and Blocking

### 3.1 Sparsity Sort: VERIFIED

R: `find_na()` sums NAs per batch, orders ascending by NA count.
Python: `_sparsity_order()` sums present features, orders descending by completeness.

Inverse logic, same ordering. Verified by sort-strategy tests and non-blocking concordance.

### 3.2 Jaccard Sort: NOT FULLY VERIFIED

R: Pair-first-then-chain: computes all pairwise Jaccard similarities, sorts descending, greedily pairs highest-similarity batches.

Python: Nearest-neighbour chain: starts from batch with highest total similarity, always picks most similar unvisited batch next.

**These are different algorithms.** They can produce different batch orderings, which changes block composition and final adjusted values.

Verification status:

- Medium dataset, block=2, jaccard sort: fixture exists, concordance test passes at rtol=5e-4
- This only proves concordance for one specific dataset/configuration
- Not verified on large or high-batch-count datasets

### 3.3 Seriation Sort: NOT FULLY VERIFIED

R: `seriation::seriate(binary_df, margin=2)`: hierarchical clustering + optimal leaf ordering (combinatorial optimization).

Python: NumPy SVD on centred presence matrix, sort by PC1 score.

**Different mathematical approaches.** Same verification limitation as Jaccard.

### 3.4 Batch Label Renumbering After Sort: UNTRACKED DIVERGENCE

**Difference 9.1**: After sorting columns, R **renumbers** batch labels sequentially in first-appearance order.

R `sorting.r:154-174`:

```r
batch_data$batch <- new_desc_after
# e.g. [1,1,1,2,2,2,3,3,3]
```

Python `sorting.py:115-120`: keeps original batch IDs unchanged, only reorders the array:

```python
sorted_batch_list = batch_list[col_order]  # same IDs, reordered
```

**Impact**: Both produce the same effective block structure because boundaries are detected by `value != previous_value`, not by the absolute label value. The `utils::tail(batch_list, n=1)` call in R's main.r:401 that computes `number_batches` works correctly after R's renumbering but would be wrong on non-contiguous original labels. Python's `n_batches = len(np.unique(batch_list))` is always correct. This is a robustness advantage for Python, not a functional difference in output values.

### 3.5 R Uses `tail()` for Batch Count, Python Uses `len(unique())`: UNTRACKED DIVERGENCE

**Difference 9.6**: R `main.r:401` computes `number_batches <- utils::tail(batch_list, n=1)`, which assumes the last batch label IS the batch count. This works because R renumbers batches after sort. If sort is not used and labels are non-contiguous (e.g., [1,1,3,3,2,2]), `tail` gives 2 when the true count is 3.

Python: `n_batches = len(np.unique(batch_list))` always gives the correct count.

**Impact**: Only matters if sort is not used AND batch labels are non-contiguous. In that case, R's block validation (`block < number_batches`) would use a potentially wrong count. This is an R fragility, not a Python bug.

---

## 4. Unique-Removal Algorithm: MINOR ALGORITHMIC DIVERGENCE

R: Updates `new_affiliation_list` in-place as it iterates. Rescued singletons become available as rescue targets for subsequent singletons (chain rescues).

Python: Pre-computes `non_unique` set from original affiliation list. Rescued singletons do NOT become new targets.

**Difference**: R can rescue more singletons via chaining. In practice, only matters when many singletons exist AND they are subsets of each other in a chain. On the highmiss dataset (only fixture with UR toggle), both produce identical results because no chain rescues occur.

Verification: Only one fixture dataset (highmiss). Not verified on datasets where chain rescues would trigger.

---

## 5. Feature Retention Policy: INTENTIONAL DIVERGENCE

| Scenario                                     | R                                               | Python                                  |
| -------------------------------------------- | ----------------------------------------------- | --------------------------------------- |
| Single-feature group (1 feature, 2+ batches) | **Dropped** (return `list()`)                   | Pass-through raw values                 |
| Single-batch block (2+ features, 1 batch)    | Pass-through raw values                         | Pass-through raw values                 |
| Feature with any per-cell NaN                | Rows dropped by sva::ComBat, absent from output | Rows all-NaN in output, shape preserved |
| Empty affiliation (no qualifying batches)    | Feature absent from rebuild                     | Feature all-NaN in output               |

Python retains more data in all cases. The trade-off is that retained features are not batch-corrected.

---

## 6. Input / Output Differences

### 6.1 Data File Reading

| Aspect                          | R `read_main_data.r`                                                         | Python `io.py`                                                                                            | Impact                                                          |
| ------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| TSV reader                      | `utils::read.table(sep="\t", header=TRUE, row.names=1)`                      | `pd.read_csv(sep="\t", index_col=0)`                                                                      | Equivalent format                                               |
| Comment character               | `comment.char = ""` (disables `#` as comment)                                | No explicit handling; default engine treats `#` as data                                                   | None in practice. Both treat `#` as normal.                     |
| Column name handling            | `check.names = FALSE` (preserves special characters)                         | Default pandas name handling (may mangle some characters)                                                 | Minor. Potential column name changes on pathological input.     |
| Factor conversion               | `stringsAsFactors = FALSE`                                                   | No factor concept in Python                                                                               | R-only concern.                                                 |
| Empty row/col removal           | `janitor::remove_empty(which = c("rows", "cols"))`: removes both             | `df.dropna(how="all", axis=0)`: removes only all-NaN rows                                                 | **DIVERGENCE**: R destroys empty columns; Python preserves.     |
| Numeric coercion                | `vapply(as.numeric)` on every column. Non-numeric cells become NA silently.  | Pandas type inference. Non-numeric columns detected by `validate_data_matrix` and raise ValueError.       | **DIVERGENCE**: R silently accepts; Python rejects.             |
| Duplicate row handling          | `unique(main_data)` on FULL DATA FRAME. Identical rows are dropped silently. | `validate_data_matrix` checks only INDEX for duplicates, raises ValueError. Data content is not compared. | **DIVERGENCE**: R drops, Python rejects. Different dedup scope. |
| Duplicate feature name handling | R's `unique()` handles duplicate row names naturally (rows become unique).   | Raises `ValueError` on duplicate index names.                                                             | Python stricter.                                                |

### 6.2 Description File Reading

| Aspect     | R `handling_description.r`       | Python `io.py`          | Impact     |
| ---------- | -------------------------------- | ----------------------- | ---------- |
| CSV reader | `read.csv(sep=",", header=TRUE)` | `pd.read_csv()`         | Equivalent |
| Batch list | Column 3 of CSV                  | Column `batch` or pos 2 | Equivalent |

### 6.3 Output Writing

| Aspect        | R `main.r`                                             | Python `io.py` / `__main__.py`                          | Impact                |
| ------------- | ------------------------------------------------------ | ------------------------------------------------------- | --------------------- |
| Default name  | `cured_data.tsv`                                       | `<data_stem>_corrected.parquet` (or .tsv if no pyarrow) | Different defaults    |
| Write method  | `write.table(cured, sep="\t", col.names=NA)`           | `df.to_csv(sep="\t")` or `df.to_parquet()`              | Python offers Parquet |
| Visualization | `2^data`, `2^cured`, then plot feature/sample/CV means | Not supported                                           | R-only (assumes log2) |
| S4 rebuild    | Builds S4 SummarizedExperiment from corrected matrix   | Not supported                                           | R-only feature        |

---

## 7. Known Verification Gaps

The following areas have passing tests but the tests are inadequate for real-world validation:

| Area                               | Problem                                                                                                                                                     | Action Needed                                                       |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Per-cell NaN concordance           | No R fixtures with per-cell NaN exist. 185 tests use clean structural missingness.                                                                          | Generate `percell_nan_*` fixtures via R scripts, compare outputs.   |
| Murine dataset                     | No R reference output available. Python pipeline processes but can't verify correctness.                                                                    | Run R on murine data, compare feature counts and adjusted values.   |
| Jaccard sort at scale              | Only verified on medium (3 batch) dataset. Real data may have 10+ batches.                                                                                  | Generate fixtures with 10+ batches, verify ordering matches R.      |
| Seriation sort at scale            | Same limitation as Jaccard.                                                                                                                                 | Same.                                                               |
| Unique-removal chaining            | Only highmiss dataset tested. Chain rescues may not trigger there.                                                                                          | Construct a dataset with deliberate chain-rescue scenario, compare. |
| Feature retention impact           | 0.236 relative diff observed on benchmark with Zipfian batches. Root cause is confirmed (prior shift), but actual impact on downstream analysis is unknown. | Benchmark with real data, compare biological conclusions.           |
| Pipeline with all options combined | Sort + block + unique-removal + per-cell NaN not tested together.                                                                                           | Generate comprehensive fixture.                                     |
| Input edge cases                   | No tests for non-numeric data coercion, empty column handling, or duplicate row handling differences.                                                       | Add cross-validation tests for I/O differences.                     |

---

## 8. Intentional Python-Only Improvements

These are not present in R HarmonizR:

- CLI (`harmonizepy` command) with `--dry-run`, `--config`, `--summary`, `--json`
- Multiple output formats (TSV, CSV, Parquet)
- `needed_values` exposed as user parameter (R hard-codes it)
- Shell tab-completion via `argcomplete`
- Centralized input validation (`validation.py`)
- Parquet I/O (12x faster write than TSV)
- Self-contained seriation via NumPy SVD (no external package)
- Module-level logging
- Pre-allocated single-output-array rebuild (lower memory)
- Robust batch count via `len(unique())` instead of R's fragile `tail()`

## 9. R-Only Features (Not in Python)

- S4 `SummarizedExperiment` input/output
- Diagnostic plotting (sample means, feature means, CV). Note: R applies `2^data` before plotting, assuming log2-transformed input.
- Multi-core execution via `doParallel` + `foreach`
- Raw R matrix input
- R-native Bioconductor integration

---

## 10. Complete Difference Inventory (All Items)

This section catalogues every known difference between R HarmonizR and HarmonizePy, regardless of impact size.

| #   | Area                                 | R Behavior                                                                          | Python Behavior                                                                        | Impact                                                                              | Documented? |
| --- | ------------------------------------ | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ----------- |
| 1   | Single-feature groups                | Dropped entirely (`list()`)                                                         | Pass-through raw values                                                                | EB prior shift for shared features. rtol widens to 5e-4.                            | Section 5   |
| 2   | Per-cell NaN in engines              | sva::ComBat na.omit drops rows                                                      | combat() detects and drops rows                                                        | Both lose quantified values. R drops row from output; Python keeps as all-NaN.      | Section 1.2 |
| 3   | Unique-removal chaining              | In-place updates allow chain rescues                                                | Static non-unique set, no chaining                                                     | R can rescue more singletons in edge cases.                                         | Section 4   |
| 4   | Jaccard sort algorithm               | Pair-first-then-chain                                                               | Nearest-neighbour chain                                                                | Different orderings for 3+ batches.                                                 | Section 3.2 |
| 5   | Seriation sort algorithm             | seriation::seriate (hierarchical)                                                   | NumPy SVD / PC1 projection                                                             | Different orderings.                                                                | Section 3.3 |
| 6   | Batch label renumbering after sort   | Renumbers to 1,2,3... in appearance order                                           | Keeps original IDs                                                                     | Block structure same (boundaries detected by value change). Absolute labels differ. | Section 3.4 |
| 7   | Batch count computation              | `tail(batch_list, n=1)`                                                             | `len(np.unique(batch_list))`                                                           | R is fragile for non-contiguous labels without sort. Python always correct.         | Section 3.5 |
| 8   | Empty affiliation features           | Forcibly create sub-df with 0 columns, fall through to `list()`, absent from output | `if len(affil) == 0: continue`, all-NaN in output                                      | R drops, Python preserves. Consistent with retention policy.                        | Section 2.5 |
| 9   | Empty column removal on read         | `janitor::remove_empty(which = c("rows", "cols"))` removes both                     | `dropna(how="all", axis=0)` drops only all-NaN rows                                    | R destroys structural-missingness columns. Python preserves shape.                  | Section 6.1 |
| 10  | Non-numeric data on read             | `vapply(as.numeric)` silently creates NA for non-numeric cells                      | Pandas type inference; `validate_data_matrix` raises ValueError on non-numeric columns | R silently accepts bad data. Python rejects.                                        | Section 6.1 |
| 11  | Duplicate row handling               | `unique(main_data)` on full data frame, compares all columns                        | `validate_data_matrix` checks index only, raises ValueError                            | R silently drops duplicate data rows. Python rejects duplicate index names only.    | Section 6.1 |
| 12  | Duplicate feature name handling      | R's `unique()` naturally deduplicates by row name                                   | Raises ValueError on duplicate index names                                             | R drops silently. Python rejects. Consistent with stricter validation philosophy.   | Section 6.1 |
| 13  | Column name handling on read         | `check.names = FALSE` preserves special characters                                  | Default pandas name handling                                                           | Minor. Potential column name changes on pathological input.                         | Section 6.1 |
| 14  | Comment character on read            | `comment.char = ""` disables `#` as comment                                         | Default pandas engine; `#` treated as normal data                                      | None in practice.                                                                   | Section 6.1 |
| 15  | String factor conversion             | `stringsAsFactors = FALSE`                                                          | Not applicable (no factor concept)                                                     | None. R-only concern.                                                               | Section 6.1 |
| 16  | Default output filename              | `cured_data.tsv`                                                                    | `<stem>_corrected.parquet` (or .tsv)                                                   | Different.                                                                          | Section 6.3 |
| 17  | Visualization data prep              | `2^data` and `2^cured` before plotting (assumes log2)                               | Not supported                                                                          | R-only.                                                                             | Section 6.3 |
| 18  | S4 SummarizedExperiment              | Supported via prepare_S4.r                                                          | Not supported                                                                          | R-only feature.                                                                     | Section 6.3 |
| 19  | Parallel execution                   | doParallel + foreach multi-core                                                     | Single-threaded                                                                        | R uses multiple cores. Python is faster anyway (vectorized NumPy).                  | Section 9   |
| 20  | Diagnostic plotting                  | sample means, feature means, CV boxplots                                            | Not supported                                                                          | R-only.                                                                             | Section 9   |
| 21  | CLI and tooling                      | Library-only, no CLI                                                                | Full CLI, config, dry-run, JSON summary                                                | Python-only improvement.                                                            | Section 8   |
| 22  | needed_values exposure               | Hard-coded: 2 for modes 1/3/limma, 1 for modes 2/4                                  | Same defaults, user-overridable                                                        | Python more flexible.                                                               | Section 8   |
| 23  | Centralized validation               | Minimal input validation, relies on R type system                                   | `validation.py` with consistent error messages                                         | Python stricter.                                                                    | Section 8   |
| 24  | Rebuild method                       | `plyr::rbind.fill` aligns by row name                                               | Pre-allocated array + `np.ix_` writes per group                                        | Equivalent for same feature sets. Different approach.                               | Section 2.5 |
| 25  | Redundant block-level non-NA counter | `value_existence_counter` checked per block (redundant with per-batch check)        | Only per-batch check                                                                   | Zero impact. R has dead code.                                                       | Section 2.1 |
| 26  | Log-only sum counter in spotting     | `sum_counter` accumulates excluded-cell count for console message                   | No equivalent counter                                                                  | Zero impact on algorithm output.                                                    | Section 2.1 |
