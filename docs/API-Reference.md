# API Reference

Complete reference for every public function and class in HarmonizePy.

```python
from harmonizepy import harmonize, combat, remove_batch_effect, HarmonizeConfig
from harmonizepy import adjust_combat, adjust_limma
```

---

## `harmonize()`

The main pipeline entry point. Handles structural missingness, sorts and blocks batches, groups features by affiliation, corrects each group, and reassembles the output.

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
) -> pd.DataFrame
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `data` | `DataFrame`, `str`, or `Path` | required | Features x samples matrix. If a path, read from that TSV/CSV/Parquet file. Feature names must be in `df.index`, sample names in `df.columns`. |
| `description` | `DataFrame`, `str`, or `Path` | required | Batch description. Must have columns `ID`, `sample`, `batch`. If a path, read from that CSV file. |
| `config` | `HarmonizeConfig` or `None` | `None` | If provided, all algorithm settings come from this object. Individual keyword arguments below are ignored. |
| `algorithm` | `"ComBat"` or `"limma"` | `"ComBat"` | Batch correction algorithm. |
| `combat_mode` | `int` (1-4) | `1` | ComBat variant. Ignored when `algorithm="limma"`. |
| `needed_values` | `int` or `None` | `None` | Minimum non-missing observations per batch for a feature to qualify. `None` auto-selects: 2 for modes 1, 3 and limma; 1 for modes 2, 4. |
| `sort` | `"sparsity"`, `"jaccard"`, `"seriation"`, or `None` | `None` | Batch sorting strategy applied before blocking. |
| `block` | `int >= 2` or `None` | `None` | Group this many consecutive sorted batches into one block. |
| `unique_removal` | `bool` | `True` | Rescue singleton-affiliation features by cropping to the nearest shared pattern. |
| `output_file` | `str`, `Path`, or `None` | `None` | Write the corrected matrix to this path. Format inferred from extension: `.tsv`, `.csv`, `.parquet`, `.pq`. |

### Returns

`pd.DataFrame` with the same shape, index, and column order as the input. NaN positions are preserved where data was insufficient for correction.

### Raises

`ValueError` on invalid arguments, mismatched sample IDs, or duplicate feature names.

### Examples

```python
from harmonizepy import harmonize

# From file paths
result = harmonize("data.tsv", "batch.csv")

# From DataFrames
result = harmonize(data_df, desc_df, algorithm="limma")

# Sort and block for many-batch datasets
result = harmonize(data_df, desc_df, sort="sparsity", block=2)

# Non-parametric ComBat for small batches
result = harmonize(data_df, desc_df, combat_mode=3, needed_values=1)

# Write output directly
result = harmonize("data.tsv", "batch.csv", output_file="corrected.parquet")
```

---

## `HarmonizeConfig`

Frozen dataclass for storing a reproducible run configuration. Pass as `config=` to `harmonize()` to override all individual keyword arguments.

```python
from harmonizepy import HarmonizeConfig

cfg = HarmonizeConfig(
    algorithm="ComBat",
    combat_mode=1,
    sort_strategy="sparsity",
    block_size=2,
    unique_removal=True,
)
```

### Fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `algorithm` | `str` | `"ComBat"` | `"ComBat"` or `"limma"` |
| `combat_mode` | `int` | `1` | ComBat mode 1-4 |
| `needed_values` | `int` or `None` | `None` | Minimum observations per batch (auto-selected if `None`) |
| `sort_strategy` | `str` or `None` | `None` | `"sparsity"`, `"jaccard"`, `"seriation"`, or `None` |
| `block_size` | `int` or `None` | `None` | Block size >= 2, or `None` |
| `unique_removal` | `bool` | `True` | Rescue singleton features |

`HarmonizeConfig` is frozen and slots-based. Once created, fields cannot be changed. Create a new instance to use different settings.

---

## `combat()`

Low-level ComBat engine. Operates directly on NumPy arrays. Does not handle structural missingness: for that, use `harmonize()`.

```python
combat(
    data,
    batch,
    *,
    par_prior=True,
    mean_only=False,
    ref_batch=None,
) -> ndarray
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `data` | `ndarray`, shape `(n_features, n_samples)` | required | Expression matrix. Per-cell NaN is allowed. |
| `batch` | array-like, shape `(n_samples,)` | required | Integer batch label per sample. |
| `par_prior` | `bool` | `True` | Use parametric prior (True = modes 1/2, False = modes 3/4). |
| `mean_only` | `bool` | `False` | Correct location only, skip scale adjustment (True = modes 2/4). |
| `ref_batch` | `int` or `None` | `None` | Reference batch to leave unadjusted. Other batches are adjusted toward this reference. |

### Returns

`ndarray` with the same shape as `data`. NaN positions preserved.

### Example

```python
import numpy as np
from harmonizepy import combat

data = np.random.randn(1000, 30)   # 1000 features, 30 samples
batch = np.array([1]*10 + [2]*10 + [3]*10)

corrected = combat(data, batch)
corrected_np = combat(data, batch, par_prior=False)  # non-parametric
corrected_loc = combat(data, batch, mean_only=True)   # location only
```

---

## `remove_batch_effect()`

Low-level limma engine. Operates directly on NumPy arrays. Does not handle structural missingness: for that, use `harmonize()`.

```python
remove_batch_effect(data, batch) -> ndarray
```

`data` is an ndarray of shape `(n_features, n_samples)`. Per-cell NaN is allowed. `batch` is a 1-D array-like of integer batch labels, length `n_samples`.

### Returns

`ndarray` with the same shape as `data`. NaN positions preserved.

### Example

```python
import numpy as np
from harmonizepy import remove_batch_effect

corrected = remove_batch_effect(data, batch)
```

---

## `adjust_combat()`

Wrapper that maps HarmonizR-style integer modes (1-4) to the `combat()` parameters. Operates on DataFrames rather than raw ndarrays.

```python
adjust_combat(sub_df, batch_labels, mode=1, ref_batch=None) -> pd.DataFrame
```

`sub_df` is a features x samples DataFrame. Returns a corrected DataFrame with the same shape, index, and columns. Intended for custom pipeline use when you are managing sub-matrices yourself.

---

## `adjust_limma()`

Wrapper applying limma correction to a DataFrame.

```python
adjust_limma(sub_df, batch_labels) -> pd.DataFrame
```

`sub_df` is a features x samples DataFrame. Returns a corrected DataFrame with the same shape, index, and columns.

---

## Semi-public submodule API

These functions are importable from their submodules for building custom pipelines. They are not re-exported from the top-level `harmonizepy` namespace.

```python
from harmonizepy.sorting import sort_batches
from harmonizepy.blocking import build_block_list
from harmonizepy.affiliation import remove_unique_combinations
```

| Function | Module | Description |
| --- | --- | --- |
| `sort_batches(data, batch_labels, strategy)` | `harmonizepy.sorting` | Returns a reordered copy of `batch_labels` according to the chosen strategy. |
| `build_block_list(batch_labels, block_size)` | `harmonizepy.blocking` | Returns a list of batch-group assignments for the given block size. |
| `remove_unique_combinations(affiliation_list)` | `harmonizepy.affiliation` | Rescues singleton-affiliation features by cropping to the nearest shared pattern. |

These interfaces are stable for the current version but are not covered by the same backward-compatibility guarantee as the top-level public API.
