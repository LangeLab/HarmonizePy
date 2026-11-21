# Pipeline

The `harmonize()` function implements the full HarmonizR-compatible pipeline. This page explains what happens at each stage and why.

---

## The structural missingness problem

Standard ComBat and limma require fully-dense matrices: every feature must be observed in every sample. Real omics data rarely satisfies this requirement. In proteomics, for example, a peptide may be reliably detected in batches 1 and 2 but absent from batch 3 entirely, not because it was zero but because it was below the detection limit for that instrument run. Treating that absence as NaN and feeding the matrix to ComBat will fail.

The HarmonizR approach solves this by exploiting the structure of the missingness. Rather than imputing or dropping, it groups features by which batches they appear in, then corrects each group independently using only the batches where that group has data.

---

## Pipeline stages

### 1. Read and validate

Inputs are validated before any computation: shapes, column names, types, and sample alignment. A `ValueError` is raised immediately if inputs are inconsistent, not after a long computation.

### 2. Batch sorting (optional)

If `sort` is specified, batches are reordered before any other processing, so that batches with similar sets of observed features sit next to each other and share a block. A feature counts as present in a batch when it has at least `needed_values` values there.

Three strategies are available:

**`"sparsity"`** - Orders batches by the number of features present, most first.

**`"jaccard"`** - Chains batches by Jaccard similarity: the features present in both batches over the features present in either. The chain starts at the batch most similar to all others and moves to the most similar batch not yet placed.

**`"seriation"`** - Orders batches by their score on the first principal component of the batches' presence vectors.

No measurement so far shows one strategy correcting better than another.

The sorting order affects which features land in which sub-matrix groups after blocking, but it does not change the final correction result for the no-block case (each feature still ends up in a group with the same batches, just potentially in a different order).

### 3. Blocking (optional)

If `block=N` is specified, every N neighbouring batches (in sort order) form a block, and affiliations are formed over blocks instead of batches: a block counts for a feature when every batch inside it holds at least `needed_values` values of the feature. With many batches, fewer distinct patterns arise and features join larger groups; on the medium and high-missingness test fixtures, blocks of two reduced the groups from 4 to 3. The engines still correct each original batch inside a sub-matrix.

Blocking costs observed values: a feature missing from one batch of a block loses the whole block. On the test fixtures it did not change peak memory. No measurement so far gives a recommended block size; with many batches and much missingness, compare the share of features corrected with and without blocking, for example from the dry run's sub-matrix count and the run log.

### 4. Affiliation spotting

For each feature, the pipeline determines its **affiliation**: the set of batches (or blocks) in which it has at least `needed_values` non-missing observations. Features with identical affiliations form a group.

A feature that has data in batches 1, 2, and 4 but not batch 3 has a different affiliation from a feature that has data in batches 1, 2, 3, and 4. They will be corrected separately.

Features with an empty affiliation (not enough data in any batch) are passed through as all-NaN rows.

### 5. Unique-combination rescue (unique removal)

Enabled by default (`unique_removal=True`). A feature whose affiliation pattern is unique across the dataset would form a singleton group that ComBat cannot correct (ComBat's EB prior estimation requires multiple features). Without this step, those features would be dropped.

The rescue algorithm crops the feature's affiliation to the nearest shared pattern (the most similar pattern that appears in at least one other feature). This allows the feature to join an existing group and receive a correction, at the cost of losing data from the batches that were cropped out.

Set `unique_removal=False` only if you need strict missingness-pattern matching and can afford to lose the affected features.

### 6. Sub-matrix extraction

Features are grouped by their (possibly rescued) affiliation, and one sub-matrix is extracted per group. Each sub-matrix contains exactly the features in that group and the samples from the batches in their shared affiliation.

Sub-matrices may still contain per-cell NaN (random missing values within a qualifying batch). These pass through to the correction engines, which handle them per-feature.

### 7. Batch correction

Each sub-matrix is corrected independently by the selected algorithm (ComBat or limma). See [[Algorithms]] for details on how per-cell NaN is handled inside the engines.

Single-feature groups are passed through without adjustment, with either engine. ComBat has no other feature to share the empirical-Bayes prior with, and limma would remove the feature's own batch means, true differences included. The feature retains its original values rather than being dropped.

### 8. Reassembly

Corrected sub-matrices are scattered back to their original positions in the output matrix. The output has the same shape, index, and column order as the input. Features that were all-NaN in the input remain all-NaN.

---

## Data flow summary

```bash
Input matrix (features x samples, may have structural NaN)
        |
        v
[Sort batches]          <- optional: sparsity / jaccard / seriation
        |
        v
[Form blocks]           <- optional: group N consecutive batches
        |
        v
Spot affiliations       <- per feature: which batches have >= needed_values observations
        |
        v
[Unique removal]        <- rescue singleton-affiliation features
        |
        v
Group by affiliation    <- features with identical affiliation -> one sub-matrix
        |
        v
Correct each sub-matrix <- ComBat or limma, per-cell NaN handled per-feature
        |
        v
Reassemble              <- scatter corrected values back to original positions
        |
        v
Output matrix (same shape as input, NaN preserved)
```

---

## NaN handling: what is and is not done

HarmonizePy never imputes. This is a hard rule.

**Structural NaN** (a feature absent from an entire batch) is handled by affiliation: that batch is excluded from the feature's group. The correction runs only on batches where the feature has enough data.

**Per-cell NaN** (a random missing value within a qualifying batch) is handled inside the correction engines using a per-feature computation. For each feature, only the observed samples are used for parameter estimation. The NaN positions in the output match the NaN positions in the input exactly.

If a feature has no qualifying affiliation group after all the steps above, it appears as an all-NaN row in the output. It is never filled with imputed or carry-forward values.

---

## Choosing pipeline parameters

- Small dataset, 2-4 batches: default (no `sort`, no `block`)
- 5-10 batches, bulk proteomics: `sort="sparsity"`, `block=2`
- 10+ batches, uneven batch sizes: `sort="jaccard"`, `block=2`
- Many batches, complex overlap structure: `sort="seriation"`, `block=2`
- To retain the most features: keep `unique_removal=True` (default)
- Strict missingness-pattern matching: `unique_removal=False`
