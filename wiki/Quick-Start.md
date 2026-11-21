# Quick Start

## The simplest case

If you have a TSV data matrix and a batch description CSV, pass the file paths directly:

```python
from harmonizepy import harmonize

result = harmonize("data.tsv", "batch.csv")
```

`result` is a `pandas.DataFrame` with features as rows and samples as columns. The shape and column order match the input exactly. Missing values stay missing.

---

## From DataFrames

If you have already loaded your data, pass DataFrames directly:

```python
import pandas as pd
from harmonizepy import harmonize

data = pd.read_csv("data.tsv", sep="\t", index_col=0)
batch = pd.read_csv("batch.csv")

result = harmonize(data, batch)
```

The `data` DataFrame must have feature names in the index and sample names as columns. See [[Input-Format]] for the exact requirements.

---

## Writing output to a file

Pass `output_file` to write the result without a separate save step:

```python
result = harmonize("data.tsv", "batch.csv", output_file="corrected.tsv")
```

Supported formats: `.tsv`, `.csv`, `.parquet`, `.pq`. Parquet requires the `harmonizepy[io]` extra.

---

## Choosing an algorithm

The default is ComBat mode 1 (parametric, location + scale). This handles the majority of bulk-proteomics and microarray datasets well.

```python
# Default: ComBat mode 1
result = harmonize(data, batch)

# limma: removes each feature's batch means with a linear model, without shrinkage.
result = harmonize(data, batch, algorithm="limma")

# ComBat mode 3: non-parametric prior, location and scale.
result = harmonize(data, batch, combat_mode=3)
```

See [[Algorithms]] for a full comparison of all modes and guidance on when to switch.

---

## Datasets with many batches: add sorting and blocking

With many batches and much missingness, features spread over many small groups. Sorting and blocking form fewer, larger groups:

```python
result = harmonize(
    data,
    batch,
    sort="sparsity",
    block=2,
)
```

`sort="sparsity"` orders batches by the number of features present in them. `block=2` then groups every 2 neighbouring batches when forming sub-matrices; a feature needs enough values in both batches of a block for the block to count. Compare the sub-matrix count of a dry run with and without blocking before choosing.

See [[Pipeline]] for a conceptual explanation of what sorting and blocking do.

---

## Reproducible runs with HarmonizeConfig

To save and reuse a run configuration:

```python
from harmonizepy import harmonize, HarmonizeConfig

cfg = HarmonizeConfig(
    algorithm="ComBat",
    combat_mode=1,
    sort_strategy="sparsity",
    block_size=2,
)

result = harmonize(data, batch, config=cfg)
```

When `config` is provided, all individual keyword arguments (`algorithm`, `sort`, `block`, etc.) are ignored. See [[API-Reference]] for the full `HarmonizeConfig` field list.

---

## Low-level API: direct engine access

For cases where you manage batches yourself or work with NumPy arrays:

```python
import numpy as np
from harmonizepy import combat, remove_batch_effect

# data_matrix: (n_features, n_samples) float64 ndarray
# batch_labels: 1-D array of integer batch labels, length n_samples

corrected = combat(data_matrix, batch_labels)
corrected = remove_batch_effect(data_matrix, batch_labels)
```

Both functions return an ndarray of the same shape as the input. NaN positions are preserved. These functions do not handle structural missingness: for that, use `harmonize()`.

---

## CLI

The same pipeline is available from the terminal:

```bash
harmonizepy data.tsv batch.csv -o corrected.tsv
```

See [[CLI-Reference]] for all options.
