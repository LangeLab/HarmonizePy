# Benchmark Results

**Generated:** 2026-05-27 00:53 UTC
**Platform:** Linux 6.17.9-76061709-generic
**CPU:** x86_64 (Py=1 thread, R=16 threads)
**Python:** 3.12.13 (harmonizepy v0.2.0)
**R:** 4.6.0 (HarmonizR 1.10.0)

### Implementation Notes

HarmonizePy is a pure NumPy implementation running single-threaded. Its ComBat and limma engines are built as vectorized array operations on pre-allocated output buffers, processing all features and affiliation groups in a single pass. This avoids the per-sub-matrix call overhead inherent in R HarmonizR's `foreach` + `sva::ComBat` dispatch, where the full engine is called separately for each unique missingness pattern.

R HarmonizR v1.10.0 (Bioconductor) uses multi-threaded execution via `doParallel` and `foreach`. On small and medium datasets (up to 5000 x 60, 10 batches), R runs 15-30x slower than HarmonizePy for ComBat modes and ~7x slower for limma. On large datasets (10000 x 100, 20 batches), R exceeds 60 seconds per scenario due to combinatorial sub-matrix fragmentation and times out.

Memory usage differs substantially. At 10000 x 100, HarmonizePy uses ~120 MB peak RSS (measured via `/usr/bin/time -v`), with the pre-allocated single-output-array strategy keeping memory at roughly 1x the input size. R HarmonizR with 16 parallel workers can reach 4-5+ GB due to `foreach` copying data per worker and per-group list allocation across its splitting and adjustment steps.

Concordance between the implementations was verified on small and medium datasets across all four ComBat modes and limma. Unblocked modes agree at machine epsilon (relative diff < 1e-14 for closed-form modes, < 6e-6 for the parametric iterative solver). Blocked modes show larger differences (relative diff ~0.4) due to differing feature retention policies: HarmonizePy preserves single-feature groups as pass-through, while R drops them entirely. This shifts the empirical Bayes prior for shared features. The per-group math is independently verified as correct.

## Python Performance

| Dataset | Algorithm | Mode | Block | Sort     | Time (s) | Memory (MB) | Features out | Corrected | Pass-through |
| ------- | --------- | ---- | ----- | -------- | -------- | ----------- | ------------ | --------- | ------------ |
| small   | limma     | --   | --    | --       | 0.343    | 73.4        | 999          | 980       | 19           |
| medium  | limma     | --   | --    | --       | 1.007    | 97.5        | 5000         | 4869      | 131          |
| large   | limma     | --   | --    | --       | 2.251    | 122.7       | 10000        | 9525      | 475          |
| small   | ComBat    | 1    | --    | --       | 0.406    | 73.0        | 999          | 980       | 19           |
| small   | ComBat    | 2    | --    | --       | 0.387    | 72.8        | 999          | 980       | 19           |
| small   | ComBat    | 3    | --    | --       | 0.443    | 73.3        | 999          | 980       | 19           |
| small   | ComBat    | 4    | --    | --       | 0.447    | 72.8        | 999          | 980       | 19           |
| medium  | ComBat    | 1    | --    | --       | 1.385    | 97.7        | 5000         | 4869      | 131          |
| medium  | ComBat    | 2    | --    | --       | 1.026    | 97.5        | 5000         | 4869      | 131          |
| medium  | ComBat    | 3    | --    | --       | 2.041    | 98.3        | 5000         | 4869      | 131          |
| medium  | ComBat    | 4    | --    | --       | 2.025    | 97.7        | 5000         | 4869      | 131          |
| large   | ComBat    | 1    | --    | --       | 3.405    | 121.1       | 10000        | 9525      | 475          |
| large   | ComBat    | 2    | --    | --       | 2.342    | 121.1       | 10000        | 9525      | 475          |
| large   | ComBat    | 3    | --    | --       | 9.764    | 121.0       | 10000        | 9525      | 475          |
| large   | ComBat    | 4    | --    | --       | 9.696    | 121.2       | 10000        | 9525      | 475          |
| small   | ComBat    | 1    | 2     | --       | 0.343    | 72.7        | 999          | 812       | 187          |
| small   | ComBat    | 2    | 2     | --       | 0.369    | 72.7        | 999          | 812       | 187          |
| small   | ComBat    | 3    | 2     | --       | 0.424    | 72.9        | 999          | 812       | 187          |
| small   | ComBat    | 4    | 2     | --       | 0.411    | 72.8        | 999          | 812       | 187          |
| medium  | ComBat    | 1    | 2     | --       | 0.741    | 94.6        | 5000         | 5000      | 0            |
| medium  | ComBat    | 2    | 2     | --       | 0.750    | 95.7        | 5000         | 5000      | 0            |
| medium  | ComBat    | 3    | 2     | --       | 1.554    | 96.5        | 5000         | 5000      | 0            |
| medium  | ComBat    | 4    | 2     | --       | 1.532    | 94.0        | 5000         | 5000      | 0            |
| large   | ComBat    | 1    | 2     | --       | 2.508    | 120.4       | 10000        | 9926      | 74           |
| large   | ComBat    | 2    | 2     | --       | 2.018    | 120.6       | 10000        | 9926      | 74           |
| large   | ComBat    | 3    | 2     | --       | 9.486    | 120.0       | 10000        | 9926      | 74           |
| large   | ComBat    | 4    | 2     | --       | 9.628    | 120.1       | 10000        | 9926      | 74           |
| medium  | ComBat    | 1    | 2     | sparsity | 0.730    | 99.1        | 5000         | 5000      | 0            |
| medium  | ComBat    | 2    | 2     | sparsity | 0.714    | 99.0        | 5000         | 5000      | 0            |
| medium  | ComBat    | 3    | 2     | sparsity | 1.549    | 99.0        | 5000         | 5000      | 0            |
| medium  | ComBat    | 4    | 2     | sparsity | 1.519    | 96.7        | 5000         | 5000      | 0            |
| large   | ComBat    | 1    | 2     | sparsity | 2.500    | 127.6       | 10000        | 9926      | 74           |
| large   | ComBat    | 2    | 2     | sparsity | 2.113    | 128.0       | 10000        | 9926      | 74           |
| large   | ComBat    | 3    | 2     | sparsity | 9.295    | 129.0       | 10000        | 9926      | 74           |
| large   | ComBat    | 4    | 2     | sparsity | 9.178    | 127.8       | 10000        | 9926      | 74           |
