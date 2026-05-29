# Input Format

HarmonizePy requires two input files (or DataFrames): a **data matrix** and a **batch description**.

---

## Data matrix

The data matrix is a features-by-samples table. Features (proteins, peptides, metabolites, genes) are rows. Samples are columns.

### File format

- Tab-separated (`.tsv`) or comma-separated (`.csv`)
- The first column contains feature names (used as the DataFrame index)
- All remaining columns are samples, one column per sample
- Values are floating-point numbers
- Missing values are represented as empty cells or `NaN`

### Example

```tsv
ProteinID    Sample_1    Sample_2    Sample_3    Sample_4    Sample_5    Sample_6
P00001       12.4        11.8        12.1                    13.0        12.7
P00002       8.3         8.1         8.9         8.4         8.7         8.2
P00003                               7.2         7.1                     7.5
P00004       15.1        14.9        15.3        15.0        14.8        15.2
```

Here `P00001` is missing in Sample_4, and `P00003` is missing in Samples 1, 2, and 5. This is typical structural missingness. HarmonizePy handles this without imputation.

### Rules

- Feature names must be unique (duplicate index values raise a `ValueError`)
- All non-index values must be numeric. Non-numeric text cells (other than NaN markers) raise a `ValueError`
- An all-NaN row is allowed and passes through unchanged
- No minimum number of features is enforced, but at least 2 features are needed for ComBat's empirical Bayes prior estimation

### When passing a DataFrame

```python
# Feature names must be in df.index
# Sample names must be in df.columns
data = pd.read_csv("data.tsv", sep="\t", index_col=0)
```

The `index_col=0` is required. Without it, pandas will treat the feature name column as a data column and raise a validation error.

---

## Batch description

The batch description maps each sample to a batch. It is a CSV file with three required columns.

### Columns

| Column | Type | Description |
| --- | --- | --- |
| `ID` | string | Sample name. Must match a column name in the data matrix exactly. |
| `sample` | integer | Sample index. Must be unique per row. Used for ordering. |
| `batch` | integer | Batch label. Samples with the same batch value are corrected together. |

### Example

```csv
ID,sample,batch
Sample_1,1,1
Sample_2,2,1
Sample_3,3,2
Sample_4,4,2
Sample_5,5,3
Sample_6,6,3
```

In this example, Samples 1 and 2 are in batch 1, Samples 3 and 4 in batch 2, and Samples 5 and 6 in batch 3.

### Rules

- Column names are case-sensitive. `ID`, `sample`, and `batch` must be spelled exactly as shown.
- Every sample name in `ID` must appear as a column in the data matrix.
- Every data matrix column must appear in `ID`. Extra samples in either direction raise a `ValueError`.
- Batch labels must be integers. String labels raise a validation error.
- At least 2 distinct batches are required. A single batch raises a `ValueError`.
- Batch sizes do not need to be equal. Unbalanced batches are handled.

### When passing a DataFrame

```python
batch = pd.read_csv("batch.csv")
# No index_col needed. The ID, sample, and batch columns are used by name.
```

---

## Parquet input

If the `harmonizepy[io]` extra is installed, `.parquet` and `.pq` files can be used as data matrix input:

```python
result = harmonize("data.parquet", "batch.csv")
```

The parquet file must have feature names as the index (row labels). Batch description files are always CSV.

---

## Validation errors

HarmonizePy validates both inputs before running any computation. Common errors and their causes:

- `"Duplicate feature names in data matrix"`: two or more rows share the same index value
- `"Non-numeric values in data matrix"`: a data cell contains text or a non-numeric value
- `"Sample IDs in description not found in data columns"`: an `ID` value in the batch description does not match any column in the data matrix
- `"Data columns not present in description"`: a data column has no corresponding row in the batch description
- `"batch column must contain integers"`: the `batch` column holds strings or floats
- `"At least 2 batches required"`: all samples share the same batch label
- `"Missing required column: ID"`: the batch description is missing one of the three required columns
