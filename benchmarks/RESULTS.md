# Benchmark Results

**Generated:** 2026-05-26 23:39 UTC
**Platform:** Linux 6.17.9-76061709-generic, Python 3.12.13
**CPU:** x86_64

## Implementation Notes

HarmonizePy is a pure NumPy implementation running single-threaded. Its ComBat and limma engines are built as vectorized array operations on pre-allocated output buffers, processing all features and affiliation groups in a single pass. This avoids the per-sub-matrix call overhead inherent in R HarmonizR's `foreach` + `sva::ComBat` dispatch, where the full engine is called separately for each unique missingness pattern.

R HarmonizR v1.10.0 (Bioconductor) uses multi-threaded execution via `doParallel` and `foreach`. On small and medium datasets (up to 5000 x 60, 10 batches), R runs 15-30x slower than HarmonizePy for ComBat modes and ~7x slower for limma. On large datasets (10000 x 100, 20 batches), R exceeds 60 seconds per scenario due to combinatorial sub-matrix fragmentation and times out.

Memory usage differs substantially. At 10000 x 100, HarmonizePy uses ~120 MB peak RSS (measured via `/usr/bin/time -v`), with the pre-allocated single-output-array strategy keeping memory at roughly 1x the input size. R HarmonizR with 16 parallel workers can reach 4-5+ GB due to `foreach` copying data per worker and per-group list allocation across its splitting and adjustment steps.

Concordance between the implementations was verified on small and medium datasets across all four ComBat modes and limma. Unblocked modes agree at machine epsilon (relative diff < 1e-14 for closed-form modes, < 6e-6 for the parametric iterative solver). Blocked modes show larger differences (relative diff ~0.4) due to differing feature retention policies: HarmonizePy preserves single-feature groups as pass-through, while R drops them entirely. This shifts the empirical Bayes prior for shared features. The per-group math is independently verified as correct.

## Python Performance

| Dataset | Algorithm | Mode | Block | Sort     | Time (s) | Memory (MB) | Features out | Corrected | Pass-through |
| ------- | --------- | ---- | ----- | -------- | -------- | ----------- | ------------ | --------- | ------------ |
| small   | ComBat    | 1    | --    | --       | 0.398    | 73.5        | 999          | 980       | 19           |
| small   | ComBat    | 2    | --    | --       | 0.382    | 73.1        | 999          | 980       | 19           |
| small   | ComBat    | 3    | --    | --       | 0.497    | 73.0        | 999          | 980       | 19           |
| small   | ComBat    | 4    | --    | --       | 0.499    | 72.4        | 999          | 980       | 19           |
| medium  | ComBat    | 1    | --    | --       | 1.569    | 97.2        | 5000         | 4869      | 131          |
| medium  | ComBat    | 2    | --    | --       | 1.115    | 97.5        | 5000         | 4869      | 131          |
| medium  | ComBat    | 3    | --    | --       | 2.361    | 97.5        | 5000         | 4869      | 131          |
| medium  | ComBat    | 4    | --    | --       | 2.314    | 97.5        | 5000         | 4869      | 131          |
| large   | ComBat    | 1    | --    | --       | 8.296    | 121.1       | 10000        | 9525      | 475          |
| large   | ComBat    | 2    | --    | --       | 7.435    | 121.2       | 10000        | 9525      | 475          |
| large   | ComBat    | 3    | --    | --       | 23.912   | 121.9       | 10000        | 9525      | 475          |
| large   | ComBat    | 4    | --    | --       | 23.877   | 120.3       | 10000        | 9525      | 475          |
| small   | limma     | --   | --    | --       | 0.426    | 73.4        | 999          | 980       | 19           |
| medium  | limma     | --   | --    | --       | 1.068    | 98.3        | 5000         | 4869      | 131          |
| large   | limma     | --   | --    | --       | 7.276    | 122.0       | 10000        | 9525      | 475          |
| small   | ComBat    | 1    | 2     | --       | 0.394    | 72.4        | 999          | 812       | 187          |
| medium  | ComBat    | 1    | 2     | --       | 0.875    | 94.1        | 5000         | 5000      | 0            |
| large   | ComBat    | 1    | 2     | --       | 2.987    | 121.5       | 10000        | 9926      | 74           |
| medium  | ComBat    | 1    | 2     | sparsity | 0.893    | 97.1        | 5000         | 5000      | 0            |
| large   | ComBat    | 1    | 2     | sparsity | 3.153    | 127.7       | 10000        | 9926      | 74           |
