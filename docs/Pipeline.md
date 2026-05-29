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

If `sort` is specified, batches are reordered before any other processing. The goal is to place batches with similar data characteristics next to each other. This improves blocking quality (groups of 2 consecutive batches will be more compatible) and can modestly improve correction quality.

Three strategies are available:

**`"sparsity"`** - Ranks batches by the fraction of features with data in that batch (completeness). Batches with similar overall completeness become neighbours. This is the fastest of the three strategies.

**`"jaccard"`** - Ranks batches by pairwise feature-overlap similarity (Jaccard index over which features are observed). More computationally expensive than sparsity. Use it when batch sizes are uneven, because it measures feature overlap rather than just raw completeness counts.

**`"seriation"`** - Projects batches onto the first principal component of the feature-overlap space and sorts by that coordinate. This is the most computationally expensive sorting option. Use it for datasets with many batches (20+) and complex feature-overlap structure.

The sorting order affects which features land in which sub-matrix groups after blocking, but it does not change the final correction result for the no-block case (each feature still ends up in a group with the same batches, just potentially in a different order).

### 3. Blocking (optional)

If `block=N` is specified, consecutive batches are grouped into blocks of size N before affiliation spotting. Within each block, the batches are treated as a single pool: a feature must have data in enough batches within that block to qualify for adjustment in it.

Blocking serves two purposes:

- **Memory reduction**: instead of holding sub-matrices spanning all batches simultaneously, the pipeline processes one block at a time. This is significant for datasets with many batches (SCP cohorts with 20-100 batches).
- **Correction quality**: grouping similar batches reduces the risk that a poorly-matched batch distorts the EB prior for the whole group.

`block=2` (pairs of consecutive batches) is the standard starting point. Combined with `sort="sparsity"`, it is the recommended configuration for datasets with 5+ batches.

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

Single-feature groups are passed through without adjustment (ComBat requires multiple features for the EB prior). The feature retains its original values rather than being dropped.

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
