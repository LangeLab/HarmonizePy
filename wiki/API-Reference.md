# API reference

Every public name in `harmonizepy`, with its signature, arguments, return value and errors. Each entry links to the page that explains the behaviour behind it.

```python
from harmonizepy import (
    harmonize, HarmonizeConfig, HarmonizeResult, InputScaleWarning,
    combat, remove_batch_effect, adjust_combat, adjust_limma,
)
```

## `harmonize()`

The full pipeline: validate, form affiliations, correct each group, reassemble ([Pipeline](Pipeline)).

```python
harmonize(
    data,
    description,
    *,
    config=None,
    algorithm="ComBat",
    combat_mode=1,
    needed_values=None,
    sort=None,
    block=None,
    unique_removal=True,
    output_file=None,
    return_report=False,
) -> pd.DataFrame | HarmonizeResult
```

| Argument         | Type                                               | Default    |
| ---------------- | -------------------------------------------------- | ---------- |
| `data`           | DataFrame or path                                  | required   |
| `description`    | DataFrame or path                                  | required   |
| `config`         | `HarmonizeConfig` or `None`                        | `None`     |
| `algorithm`      | `"ComBat"` or `"limma"`                            | `"ComBat"` |
| `combat_mode`    | 1, 2, 3 or 4                                       | 1          |
| `needed_values`  | int >= 1 or `None`                                 | `None`     |
| `sort`           | `"sparsity"`, `"jaccard"`, `"seriation"` or `None` | `None`     |
| `block`          | int >= 2 or `None`                                 | `None`     |
| `unique_removal` | bool                                               | `True`     |
| `output_file`    | path or `None`                                     | `None`     |
| `return_report`  | bool                                               | `False`    |

- `data`, `description`: the two tables of [Input format](Input-Format). A data path is read by its extension; a description path is always read as comma-separated.
- `config`: when given, its fields replace `algorithm`, `combat_mode`, `needed_values`, `sort`, `block` and `unique_removal`, and those keyword arguments are ignored.
- `combat_mode`: see [ComBat](ComBat#the-four-modes). Ignored with limma.
- `needed_values`: values a feature needs in a batch for the batch to count ([Missing values](Missing-Values)). `None` picks 2 for ComBat modes 1 and 3 and for limma, 1 for modes 2 and 4.
- `sort`, `block`: see [Sorting and blocking](Sorting-and-Blocking). `block` must also be below the number of batches.
- `unique_removal`: see [Unique removal](Unique-Removal).
- `output_file`: also write the corrected matrix there: `.csv`, `.parquet` or `.pq` by extension, otherwise tab-separated.
- `return_report`: return the report with the matrix ([Correction report](Correction-Report)).

Returns the corrected matrix as a DataFrame, with the input's features and columns in the input's order; a feature repeated with the same name and values appears once. Features that could not be corrected keep their measured values or come out empty, as the report records. With `return_report=True`, returns a `HarmonizeResult`.

Raises `ValueError` on an invalid argument, an input that fails validation, or a feature name repeated with different values; `TypeError` if `unique_removal` is not a bool; `OSError` if a path cannot be read or `output_file` cannot be written; `ImportError` for Parquet without `pyarrow`. Warns with `InputScaleWarning` when the values do not look log-transformed.

```python
result = harmonize("data.tsv", "batch.csv")
result = harmonize(data_df, desc_df, algorithm="limma", sort="sparsity", block=2)  # needs 3+ batches
result, report = harmonize(data_df, desc_df, return_report=True)
```

## `HarmonizeResult`

A named tuple of two DataFrames, returned by `harmonize(..., return_report=True)`; it unpacks as `data, report`.

- `data`: the corrected matrix, as `harmonize()` returns it otherwise.
- `report`: one row per feature of `data`, in its order, with the columns `status`, `batches_used`, `group_size`, `cropped` and `repeats_removed` ([Correction report](Correction-Report#columns)).

## `HarmonizeConfig`

A frozen dataclass holding a run's settings, for reuse with `harmonize(data, description, config=cfg)`.

| Field            | Default    | Same as          |
| ---------------- | ---------- | ---------------- |
| `algorithm`      | `"ComBat"` | `algorithm`      |
| `combat_mode`    | 1          | `combat_mode`    |
| `needed_values`  | `None`     | `needed_values`  |
| `sort_strategy`  | `None`     | `sort`           |
| `block_size`     | `None`     | `block`          |
| `unique_removal` | `True`     | `unique_removal` |

The fields are checked when the instance is created, raising `ValueError` or `TypeError` as `harmonize()` would, and cannot be changed afterwards; make a new instance for other settings. The settings themselves are explained on the pages linked from `harmonize()` above.

## `InputScaleWarning`

A `UserWarning` subclass raised by `harmonize()` and the command line when the values look untransformed ([Input format](Input-Format#scale)). Filter it by category to silence it.

## `combat()`

The ComBat engine on one matrix, with no pipeline around it ([ComBat](ComBat)).

```python
combat(data, batch, *, par_prior=True, mean_only=False, ref_batch=None) -> ndarray
```

- `data`: a features by samples float array; missing cells allowed.
- `batch`: one whole-number label per sample.
- `par_prior`: parametric prior (modes 1 and 2) or non-parametric (modes 3 and 4).
- `mean_only`: correct the location only (modes 2 and 4).
- `ref_batch`: a batch label to leave unchanged and adjust the others toward; it needs at least two samples.

Returns a new array of the same shape; missing cells stay missing. Features ComBat cannot fit keep their input values ([ComBat](ComBat#features-combat-leaves-raw)). A feature with no value in a batch is fitted on the batches it has. Raises `ValueError` on a wrong shape, a batch length that does not match, labels that are not whole numbers, infinite values, or an invalid `ref_batch`.

## `remove_batch_effect()`

The limma engine on one matrix ([limma](limma)).

```python
remove_batch_effect(data, batch) -> ndarray
```

Takes the same `data` and `batch` as `combat()`. Returns a new array of the same shape; missing cells stay missing. A feature is fitted on the batches it has values in; one with values in a single batch, or missing a batch with no more values than observed batches, keeps its input values. Called directly, it corrects a single feature; inside `harmonize()` a group of one feature is left raw. Raises `ValueError` as `combat()` does, apart from `ref_batch`.

## `adjust_combat()` and `adjust_limma()`

DataFrame wrappers around the engines, for custom pipelines that manage their own groups.

```python
adjust_combat(sub_df, batch_labels, mode=1, ref_batch=None) -> pd.DataFrame
adjust_limma(sub_df, batch_labels) -> pd.DataFrame
```

`mode` is a ComBat mode, 1 to 4. Both return a corrected DataFrame with the index and columns of `sub_df`, and raise `ValueError` on labels that do not match the columns or are not whole numbers, or on input the engine refuses.

## Pipeline helpers

Importable from their modules for custom pipelines; not exported from `harmonizepy` itself.

```python
from harmonizepy.sorting import sort_batches
from harmonizepy.blocking import build_block_list
from harmonizepy.affiliation import remove_unique_combinations
```

- `sort_batches(data, batch_list, strategy, needed_values)` returns `(sorted_data, sorted_batch_list, col_order)`: the matrix with its columns reordered so that similar batches are adjacent, the labels in the new order, and the input column index at each new position (`np.argsort(col_order)` restores the input order).
- `build_block_list(batch_list, block_size)` returns the block label of each sample, starting at 1, with every `block_size` batches in order of first appearance sharing a block.
- `remove_unique_combinations(affiliation_list)` takes one affiliation (a tuple of batch or block labels) per feature and returns a new list in which each affiliation held by one feature only is replaced by the longest shared affiliation inside it, where one exists ([Unique removal](Unique-Removal#the-rule)).
