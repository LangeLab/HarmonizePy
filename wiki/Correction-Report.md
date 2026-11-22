# Correction report

After a run, some features were corrected, some kept their measured values, and some came out empty. The correction report says which, one row per feature, with the reason. It is the way to audit a run without reading the log.

I leave a feature uncorrected rather than correct it where its batch effect cannot be estimated, and that choice is only fair if it is visible. The report is where it is visible.

## Getting it

From Python, ask for it with `return_report=True`; `harmonize()` then returns a `HarmonizeResult`, which unpacks into the matrix and the report:

```python
result, report = harmonize("proteins.tsv", "batches.csv", return_report=True)
report["status"].value_counts()
```

From the command line, `--report PATH` writes it to `PATH`. Its format follows the extension: `.csv`, `.parquet` or `.pq`, otherwise tab-separated. For the [Quick start](Quick-Start) example:

```text
protein	status	batches_used	group_size	cropped	repeats_removed
P1	corrected	1,2	4	False	0
P2	corrected	1,2	4	False	0
P3	corrected	1,2	4	False	0
P4	corrected	1,2	4	True	0
P5	raw_alone_in_pattern		1	False	0
P6	missing_no_usable_batch		0	False	0
P7	raw_single_batch		1	False	0
P8	missing_no_usable_batch		0	False	0
```

## Columns

| Column            | Holds                     |
| ----------------- | ------------------------- |
| `status`          | outcome, below            |
| `batches_used`    | batch labels of the group |
| `group_size`      | features in the group     |
| `cropped`         | moved by unique removal   |
| `repeats_removed` | identical repeats dropped |

- `batches_used` lists the batches the feature was corrected over, sorted and comma-separated, and is empty when it was not corrected. With blocking it lists every batch of the group's blocks.
- `group_size` is 0 for a feature with no usable batch.
- `repeats_removed` counts rows with the same name and the same values that were removed before correction (see [Input format](Input-Format#repeated-features)).

## Statuses

| Status                    | Output values |
| ------------------------- | ------------- |
| `corrected`               | corrected     |
| `raw_single_batch`        | measured      |
| `raw_alone_in_pattern`    | measured      |
| `raw_zero_variance`       | measured      |
| `raw_too_few_features`    | measured      |
| `raw_too_few_values`      | measured      |
| `missing_no_usable_batch` | empty         |

A raw feature keeps its measured values inside its own group's batches. Outside them its cells are empty, as for a corrected feature.

### `corrected`

The engine fitted the feature with the rest of its group.

### `raw_single_batch`

The feature's group spans one batch, or with blocking one block that holds one batch. There is no other batch to correct it against. P7, measured only in batch 3, is the example.

### `raw_alone_in_pattern`

No other feature shares the feature's affiliation, and [unique removal](Unique-Removal) found no shared affiliation inside it, or was switched off. A group of one is not corrected: ComBat has no other feature to form a prior from, and limma would remove the feature's own batch differences, real ones included. P5 is the example.

### `raw_zero_variance`

ComBat only. The feature's variance is too small to scale by: its pooled variance is below $`10^{-12}`$ in any mode, or in mode 1 its variance in one batch is below $`10^{-12}`$ of its pooled variance. A feature constant within each batch is one example. Fitting it would also distort the prior for every other feature, so ComBat fits the group without it. In ComBat mode 2 the example's P6, with one value in each batch, has zero spread within batches and ends here.

### `raw_too_few_features`

ComBat only. After the features above were set aside, fewer than two features were left in the group, and a prior needs two.

### `raw_too_few_values`

limma only. The feature has no more values than batches it is observed in, such as one value in each batch, so the fit would leave no residual and collapse it to one number. With limma and `needed_values=1`, the example's P6 ends here (see [limma](limma#edges-worth-knowing)).

### `missing_no_usable_batch`

No batch has at least `needed_values` values of the feature (with blocking, no block whose every batch has), so it has no affiliation. Its row is empty in the output, even if it had measured values. P6 (one value in each of three batches) and P8 (empty) are the examples.

## Reading a run

> [!TIP]
> `report["status"].value_counts()` is the one-line summary of a run. Compare it between two settings before choosing one.

- A count of `cropped` rows tells how many features gave up batches to unique removal; switching it off turns them into `raw_alone_in_pattern`.
- Many `missing_no_usable_batch` rows can mean `needed_values` is higher than the data support, or blocks too large for the coverage.
- `raw_single_batch` rows are features whose group spans one batch. A protein measured in one batch only can never be corrected; one with a few values elsewhere may join a larger group with a lower `needed_values`.

## Related pages

- [Pipeline](Pipeline): the path from affiliation to status, as a diagram.
- [ComBat](ComBat#features-combat-leaves-raw): the rules behind the ComBat-only statuses.
- [CLI reference](CLI-Reference): `--report` and the other output flags.
