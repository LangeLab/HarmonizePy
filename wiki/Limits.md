# Limits

HarmonizePy corrects batch effects in a matrix of log-scale values, one affiliation group at a time, using only the batch labels and only the values that were measured. Everything outside that is outside the tool. This page lists the edges, so that a result is not read as more than it is.

## What it does not do

- It does not impute. A missing value stays missing, and some measured values become missing (see [Missing values](Missing-Values#missing-values-are-never-filled)).
- It does not transform. Values must already be on a log scale; the scale check is a heuristic warning, not a guarantee (see [Input format](Input-Format#scale)).
- It does not know about biology. The batch label is the only thing read about a sample. There is no argument for a condition, a covariate or a design, so a condition that is unevenly spread over batches is partly removed with the batch (see [Batch effects](Batch-Effects#when-the-batch-is-also-the-biology)).
- It does not anchor groups to a common level. Each affiliation group is corrected toward the mean of its own batches, so features measured in different sets of batches can come out at levels that differ for that reason alone (see [Pipeline](Pipeline#groups-are-corrected-to-their-own-level)).
- It does not correct features it cannot fit. Groups of one feature or one batch, features whose variance is zero under ComBat, and features with no usable batch are returned measured or empty, with the reason in the [correction report](Correction-Report). Under limma, a feature with no spread within its batches is corrected to a single value.
- It does not plot or diagnose. There are no PCA plots or batch-effect scores; besides the matrix, a run gives the report, the log and, on request, a JSON summary.
- It does not store a fit. Each run estimates everything from its own input, so new samples cannot be corrected against an earlier run.

## What the evidence covers

> [!IMPORTANT]
> I change a default only when a simulation with a known truth shows the new behaviour is better. Agreement with R is not a reason by itself. That simulation does not exist yet, so no default here claims to be the most accurate choice.

- The engines and the pipeline are compared against committed outputs of `sva::ComBat`, `limma::removeBatchEffect` and HarmonizR 1.10.0, with stated tolerances (see [R compatibility](R-Compatibility)). Agreement with R shows the same computation, not a correct answer.
- No test checks recovery of a known truth: a matrix where the true batch effects and the true biology are known, before and after correction. The batch-effect tests check that the differences between batch means shrink.
- No measurement here ranks the engines, the modes, the sort strategies or the block sizes on accuracy. The settings pages say what each one does, not which is best.

## Edges in the engines

- A batch with one sample: ComBat corrects its groups for location only; limma replaces that sample's values with the feature's mean of batch means (see [limma](limma#edges-worth-knowing)).
- In ComBat, a group with any missing cell uses a different variance rule from a complete group, following `sva::ComBat`, so one missing cell changes every feature of its group (see [ComBat](ComBat#how-a-group-is-corrected)).
- In ComBat, features in the same group affect each other's correction through the shared prior: adding or removing a feature can move the others.

## Scale

HarmonizePy holds one matrix in memory and starts no threads of its own. NumPy's linear algebra library can use several cores unless `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are set to 1, as the benchmarks set them. The benchmark suite has run it on matrices of up to 10,000 features by 100 samples and 5,000 features by 10,000 samples (see [Benchmarks](Benchmarks)).

## Roadmap

Planned, not available: accuracy measured against known truth; a common reference level for groups; biological covariates; a result object that stores the fit, so new batches can be corrected against it.

## Related pages

- [Correction report](Correction-Report): which features each limit touched in a given run.
- [R compatibility](R-Compatibility): where HarmonizePy and R differ, on purpose.
