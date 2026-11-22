# Quick start

One small matrix, from files to a corrected result, with the real output of every step. The same proteins appear throughout this wiki, so the cases met here (a protein missing from a batch, one measured in every batch, one with too few values anywhere) are explained there in full.

## 1. Two input files

The data matrix: features in rows, samples in columns, log2 values, empty cells for missing values. Save it as `proteins.tsv` (tab-separated):

```text
protein	s1	s2	s3	s4	s5	s6	s7	s8	s9
P1	20.1		19.6	21.3	20.9	21.8			
P2	18.2	18.9	17.7	19.8	20.4	19.1			
P3	22.5	21.7	23.0	23.9	24.4	23.2	21.1		
P4	19.0	19.8	18.4	20.6	21.2	20.1	17.9	18.6	18.1
P5				16.2	15.8	16.6	17.0	16.4	17.3
P6	20.5				21.0				19.2
P7							24.2	23.6	24.9
P8									
```

The batch description: one row per sample, with its batch. Save it as `batches.csv`:

```text
ID,sample,batch
s1,1,1
s2,2,1
s3,3,1
s4,4,2
s5,5,2
s6,6,2
s7,7,3
s8,8,3
s9,9,3
```

The `sample` column is not read, but the file needs three columns (see [Input format](Input-Format)).

## 2. Check the plan

`--dry-run` reads and validates both files and plans the run without correcting or writing anything:

```bash
harmonizepy proteins.tsv batches.csv --dry-run
```

<details>
<summary>Output: the plan, 3 sub-matrices</summary>

```text
  HarmonizePy 0.4.0 dry run
  ----------------------------------------------------
  Input data:      proteins.tsv
  Description:     batches.csv
  Output:          proteins_corrected.tsv  [tsv]
  ----------------------------------------------------
  Features:        8
  Samples:         9
  Batches:         3
  Sub-matrices:    3  (unique affiliation groups)
  ----------------------------------------------------
  Algorithm:       ComBat mode 1
  needed_values:   2
  Sort strategy:   none
  Block size:      none (all batches in one group)
  Unique removal:  enabled
  ----------------------------------------------------
  Inputs valid. Use without --dry-run to run correction.
```

</details>

`needed_values` is the number of values a protein needs in a batch for that batch to count (2 here). The batches that count for a protein are its affiliation, and proteins with the same affiliation form a group, which the dry run calls a sub-matrix. Three sub-matrices: six of the eight proteins fall into three groups by the batches they have enough values in, and the other two have no such batch. [Pipeline](Pipeline) shows which.

## 3. Run it

```bash
harmonizepy proteins.tsv batches.csv --report report.tsv
```

```text
INFO [harmonizepy] Input: 8 features x 9 samples
INFO [harmonizepy] Algorithm: ComBat mode 1 | batches: 3 | needed_values: 2
INFO [harmonizepy] Unique removal moved 1 feature(s) into a shared pattern (12.5%)
INFO [harmonizepy] Adjusting 3 unique affiliation group(s)
INFO [harmonizepy] 2 feature(s) passed through without correction (1 in single-batch groups, 1 alone in their pattern)
INFO [harmonizepy] Done (0.00s).
INFO [harmonizepy] Output written to proteins_corrected.tsv
INFO [harmonizepy] Report written to report.tsv
```

The run writes its files next to the input:

| File | Written |
| --- | --- |
| `proteins_corrected.tsv` | always |
| `proteins_corrected.log` | unless `--no-log` |
| `report.tsv` | with `--report` |

## 4. The same run from Python

```python
from harmonizepy import harmonize

result, report = harmonize("proteins.tsv", "batches.csv", return_report=True)
print(result.round(2))
```

```text
            s1     s2     s3     s4     s5     s6    s7    s8    s9
protein
P1       20.67    NaN  20.22  20.69  20.33  21.14   NaN   NaN   NaN
P2       19.02  19.66  18.57  18.98  19.52  18.35   NaN   NaN   NaN
P3       23.30  22.58  23.75  23.09  23.54  22.45   NaN   NaN   NaN
P4       19.84  20.56  19.30  19.77  20.32  19.32   NaN   NaN   NaN
P5         NaN    NaN    NaN  16.20  15.80  16.60  17.0  16.4  17.3
P6         NaN    NaN    NaN    NaN    NaN    NaN   NaN   NaN   NaN
P7         NaN    NaN    NaN    NaN    NaN    NaN  24.2  23.6  24.9
P8         NaN    NaN    NaN    NaN    NaN    NaN   NaN   NaN   NaN
```

`harmonize()` also takes DataFrames: the matrix with feature names as the index (`pd.read_csv("proteins.tsv", sep="\t", index_col=0)`) and the description with no index (`pd.read_csv("batches.csv")`).

## 5. What happened to each protein

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

- P1 to P4 were corrected together over batches 1 and 2. Their batch gap of about 1.5 shrank to between 0.1 and 0.3 ([Batch effects](Batch-Effects#what-correction-does-to-it)).
- P4 was measured in all three batches and was moved into that group by [unique removal](Unique-Removal); its batch 3 values are now empty.
- P3's single value in batch 3 is below `needed_values` (2 values per batch by default), so batch 3 is not part of its correction, and that value is empty too.
- P5 and P7 kept their measured values: no other protein has enough values in exactly their batches.
- P6 and P8 are empty: neither has two values in any batch.

## Where next

- Your own data: [Input format](Input-Format) for what the two files must hold, and the log scale the values should be on.
- Settings: [Choosing an engine](Algorithms), and [Sorting and blocking](Sorting-and-Blocking) for many batches.
- Reading results: [Correction report](Correction-Report).
- Everything the command takes: [CLI reference](CLI-Reference); the Python functions: [API reference](API-Reference).
