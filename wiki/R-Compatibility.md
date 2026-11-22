# R compatibility

HarmonizePy is checked against the R packages it reimplements ([Leek et al. 2012](References#leek-et-al-2012); [Ritchie et al. 2015](References#ritchie-et-al-2015); [HarmonizR 1.10.0](References#harmonizr-1100)): `sva::ComBat` 3.60.0, `limma::removeBatchEffect` and HarmonizR 1.10.0, pinned in the repository's `renv.lock`. Matching R is a check that the computation is the same, not the goal itself.

> [!NOTE]
> Agreement with R shows that HarmonizePy computes what R computes. It is not evidence that either result is correct. I keep it as a regression check, and do not change HarmonizePy just to match R. Where HarmonizePy behaves differently on purpose, the difference and its reason are listed below.

## How the comparison works

The R outputs are generated once by scripts in `tests/fixtures/` and committed. The test suite runs the Python code on the same inputs and compares cell by cell with `numpy.testing.assert_allclose`, so the comparison runs in CI without R installed. Engine tests compare `combat()` and `remove_batch_effect()` with `sva` and `limma`; pipeline tests compare `harmonize()` with HarmonizR, including sorting, blocking, unique removal and `needed_values`, and check that the missing cells sit in the same places.

## Tolerances

The tolerances the tests assert, as relative and absolute tolerance:

| Comparison                 | rtol          | atol          |
| -------------------------- | ------------- | ------------- |
| limma, engine and pipeline | 1e-8 to 1e-10 | 1e-8 to 1e-10 |
| ComBat modes 2 and 4       | 1e-6 to 1e-4  | 1e-8 to 1e-4  |
| ComBat mode 1              | 1e-4 to 5e-4  | 1e-6 to 1e-4  |
| ComBat mode 3              | 1e-6 to 5e-4  | 1e-8 to 1e-4  |

- The range in each row covers the different fixtures: dense, sparse, high-missingness, blocked and sorted inputs.
- limma is a closed-form least-squares fit, so it agrees to rounding. ComBat mode 1 iterates to a convergence threshold of $`10^{-4}`$, with a slightly different stopping test from `sva` (see [ComBat](ComBat#parametric-and-non-parametric-priors)), so it agrees to about that threshold.
- The benchmark suite compares against HarmonizR on larger matrices as well; its results are in [Benchmarks](Benchmarks).

## Where HarmonizePy differs on purpose

### Features are kept

HarmonizR removes some features from its output. I keep every input feature except an exact repeat, so the output has one row per input feature whatever the data:

- A feature alone in its affiliation, over two or more batches: HarmonizR does not return it as measured. In the benchmarks it drops such features; on the [Quick start](Quick-Start) example it keeps P5 but empties its batch 2 values. HarmonizePy returns its measured values (`raw_alone_in_pattern`).
- A feature with no usable batch: R drops it; HarmonizePy returns an empty row (`missing_no_usable_batch`).

The [correction report](Correction-Report) says which rows were not corrected. To get HarmonizR's shape for a comparison, drop the empty rows yourself:

```python
result = harmonize("data.tsv", "batch.csv")
result = result.dropna(how="all")
```

### A feature missing a whole batch

Called directly on such a feature, `sva::ComBat` stops with a singular-matrix error; `combat()` fits it on the batches it has. `limma::removeBatchEffect` sets the coefficient it cannot estimate to zero; `remove_batch_effect()` fits contrasts over the batches the feature has. For a feature missing an interior batch the two limma versions agree; for one missing the first or last batch they differ (see [limma](limma#missing-cells)). Inside `harmonize()` the case does not arise, because every feature has values in every batch of its group.

### Constant features

Both leave rows that cannot be scaled out of the ComBat fit and return them unadjusted, with different tests:

- `sva` leaves out a row whose variance is exactly zero in any batch of two or more samples, in every mode. Its test takes the variance without removing missing cells, so a row with a missing cell is never left out, and such a row can make `sva::ComBat` stop with an error.
- HarmonizePy leaves out, in every mode, a row whose pooled variance is below $`10^{-12}`$, and in mode 1 also a row whose variance in a batch with two or more values is below $`10^{-12}`$ of its pooled variance. Near-constant rows are therefore left out too, and rows with missing cells are tested like any other.

### Unique removal

Both move a feature with a pattern of its own into the longest pattern of another feature inside its own, even one held by a single feature (see [Unique removal](Unique-Removal)); `(1,2,3,4), (1,2), (5,6), (5,6)` becomes `(1,2), (1,2), (5,6), (5,6)` in both. They differ in three ways: HarmonizR walks the features in row order and takes the first of equally long targets, where HarmonizePy takes the longest patterns first and breaks ties by group size, then the smallest pattern; HarmonizR empties a feature with no target, where HarmonizePy keeps its measured values; and HarmonizR accepts a target of one batch, which HarmonizePy refuses.

### Sort orders

All three sort strategies can order batches differently from HarmonizR, whose rules differ:

| Strategy    | HarmonizePy                 | HarmonizR                  |
| ----------- | --------------------------- | -------------------------- |
| `sparsity`  | most present features first | fewest missing cells first |
| `jaccard`   | nearest-neighbour chain     | pairs, then chained        |
| `seriation` | first principal component   | `seriation::seriate`       |

Sorting changes the output only together with blocking (see [Sorting and blocking](Sorting-and-Blocking)).

### Input handling

- A cell that is not a number: R turns it into a missing value; HarmonizePy refuses the column.
- Repeated rows: R removes rows whose values repeat another row, whatever their names; HarmonizePy removes a row only when its name and its values repeat, logs it, and refuses a repeated name with different values.
- Empty rows and columns: R removes them; HarmonizePy keeps them.
- Columns of the output: R returns only the columns of the groups it corrected, in the order of the groups, so a batch that no group uses has no columns in its output; HarmonizePy returns every input column, in the input order.
- The batch description: R reads its columns by position; HarmonizePy reads the columns named `ID` and `batch`, and falls back to position.
- `needed_values`: HarmonizePy takes it as an argument, with HarmonizR's defaults for ComBat.
- Output file: R writes `cured_data.tsv`; the `harmonizepy` command writes `<data stem>_corrected.tsv` next to the input, and `harmonize()` writes only to `output_file`.

## Where HarmonizR is the better choice

- Reproducing a published HarmonizR result exactly: the unique-removal rule and the sort orders differ, so some features land in different groups.
- Wanting HarmonizR's output shape: it removes features it does not correct, where HarmonizePy keeps them (`result.dropna(how="all")` removes the empty rows, but raw features stay).
- Working in R: HarmonizR accepts and returns S4 objects, plots, and can run its groups on several cores.

## Not provided

HarmonizR's S4 objects, plotting, and multi-core execution have no counterpart. HarmonizePy starts no threads or processes of its own.

## Comparing with your own R run

To compare a HarmonizePy result with a HarmonizR result on the same input, align the two on their shared features and samples, compare the cells both have, and count the features only one side returns:

```python
import numpy as np
import pandas as pd

py = pd.read_csv("data_corrected.tsv", sep="\t", index_col=0)
r = pd.read_csv("cured_data.tsv", sep="\t", index_col=0)

rows = py.index.intersection(r.index)
cols = py.columns.intersection(r.columns)
a, b = py.loc[rows, cols].to_numpy(), r.loc[rows, cols].to_numpy()

both = ~np.isnan(a) & ~np.isnan(b)
print("max relative difference:", np.max(np.abs(a[both] - b[both]) / np.abs(b[both])))
print("missing cells differ:", int((np.isnan(a) != np.isnan(b)).sum()))
print("features only in HarmonizePy:", len(py.index.difference(r.index)))
```

Features only in the HarmonizePy output are the kept features described above, and rows R removed as value repeats. A difference in missing cells on shared features comes from unique removal, from the sort order with blocking, or from HarmonizR returning whole features missing, which the benchmarks show in some ComBat modes (see [Benchmarks](Benchmarks#reading-the-concordance)).

## Related pages

- [Benchmarks](Benchmarks): concordance on larger and real matrices.
- [Limits](Limits): what agreement with R does and does not show.
