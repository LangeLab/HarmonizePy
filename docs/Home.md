# HarmonizePy

Pure-Python batch-effect harmonization for omics data. Validated against R `sva::ComBat`, `limma::removeBatchEffect`, and `HarmonizR` v1.10.0.

**Version:** 0.3.2 | **Status:** Alpha | **License:** GPL-3.0 | **Python:** 3.12 - 3.14

---

## What it does

Batch effects are systematic technical differences between samples processed at different times, in different labs, or on different instruments. Left uncorrected, they dominate downstream analyses and obscure real biology.

HarmonizePy corrects batch effects in omics datasets where **structural missingness** is common. In proteomics, metabolomics, and single-cell data, many features are absent from entire batches because they were simply not detected, not because they were measured as zero. Standard ComBat and limma implementations require complete matrices and fail on these inputs.

HarmonizePy solves this by grouping features by their batch-observation pattern, correcting each compatible group independently with ComBat or limma, and reassembling the original matrix. Missing values are never imputed: if a feature is missing, it stays missing.

## Key features

- **ComBat** (all 4 modes) and **limma** batch-correction engines implemented in pure NumPy
- **HarmonizR-compatible pipeline**: structural missingness handling, batch sorting, blocking, unique-combination rescue
- **No imputation**: NaN in, NaN out. The missing-value structure is preserved exactly
- **No R required at runtime**: runs on any Python environment
- **CLI included**: `harmonizepy data.tsv batch.csv -o corrected.tsv`
- **Validated against R reference outputs** across synthetic and real proteomics datasets

## 5-second example

```python
from harmonizepy import harmonize

result = harmonize("data.tsv", "batch.csv")
```

That is the entire API for most use cases. `result` is a `pandas.DataFrame` with the same shape as the input.

---

## Documentation

- [[Installation]]: install options, optional extras, dependencies
- [[Quick-Start]]: minimal working examples, choosing settings
- [[Input-Format]]: data matrix and batch description format, validation rules
- [[Algorithms]]: ComBat modes 1-4, limma, when to use each
- [[Pipeline]]: structural missingness, sorting, blocking, unique removal, NaN handling
- [[API-Reference]]: full signature for every public function and class
- [[CLI-Reference]]: all command-line flags, config files, output formats
- [[R-Compatibility]]: what matches R, what differs, numerical tolerances
- [[Benchmarks]]: dataset catalog, runtime expectations, how to reproduce

---

## References

- Johnson WE, Li C, Rabinovic A. "Adjusting batch effects in microarray expression data using empirical Bayes methods." *Biostatistics* 8(1):118-127, 2007.
- Ritchie ME et al. "limma powers differential expression analyses for RNA-sequencing and microarray studies." *Nucleic Acids Research* 43(7):e47, 2015.
- Voß H et al. "HarmonizR enables data harmonization across independent proteomic datasets with appropriate handling of missing values." *Nature Communications* 13:3523, 2022.
- Schlumbohm S, Neumann JE, Neumann P. "HarmonizR: blocking and singular feature data adjustment improve runtime efficiency and data preservation." *BMC Bioinformatics*, 2025.
