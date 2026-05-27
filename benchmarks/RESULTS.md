# Benchmark Results

**Generated:** 2026-05-27 01:42 UTC
**Platform:** Linux 6.17.9-76061709-generic
**CPU:** x86_64 (Py=1 thread, R=16 threads)
**Python:** 3.12.13 (harmonizepy v0.2.0)
**R:** 4.6.0 (HarmonizR 1.10.0)

## Data Specifications

| Dataset   | Type                    | Features/Proteins | Samples/Cells | Batches | Missingness | File Size |
| --------- | ----------------------- | ----------------- | ------------- | ------- | ----------- | --------- |
| small     | Bulk proteomics, small  | 1000              | 20            | 5       | 30%         | 261 KB    |
| medium    | Bulk proteomics, medium | 5000              | 60            | 10      | 20%         | 4.3 MB    |
| large     | Bulk proteomics, large  | 10000             | 100           | 20      | 5%          | 16.7 MB   |
| scp_small | SCP cohort, small       | 3000              | 1000          | 20      | 50%         | 27.9 MB   |
| scp_large | SCP cohort, large       | 5000              | 10000         | 100     | 60%         | 379.0 MB  |


### Implementation Notes

HarmonizePy is a pure NumPy implementation running single-threaded. Its ComBat and limma engines are built as vectorized array operations on pre-allocated output buffers, processing all features and affiliation groups in a single pass. This avoids the per-sub-matrix call overhead inherent in R HarmonizR's `foreach` + `sva::ComBat` dispatch, where the full engine is called separately for each unique missingness pattern.

R HarmonizR v1.10.0 (Bioconductor) uses multi-threaded execution via `doParallel` and `foreach`. On small and medium datasets (up to 5000 x 60, 10 batches), R runs 15-30x slower than HarmonizePy for ComBat modes and ~7x slower for limma. On large datasets (10000 x 100, 20 batches), R exceeds 60 seconds per scenario due to combinatorial sub-matrix fragmentation and times out.

Memory usage differs substantially. At 10000 x 100, HarmonizePy uses ~120 MB peak RSS (measured via `/usr/bin/time -v`), with the pre-allocated single-output-array strategy keeping memory at roughly 1x the input size. R HarmonizR with 16 parallel workers can reach 4-5+ GB due to `foreach` copying data per worker and per-group list allocation across its splitting and adjustment steps.

Concordance between the implementations was verified on small and medium datasets across all four ComBat modes and limma. Unblocked modes agree at machine epsilon (relative diff < 1e-14 for closed-form modes, < 6e-6 for the parametric iterative solver). Blocked modes show larger differences (relative diff ~0.4) due to differing feature retention policies: HarmonizePy preserves single-feature groups as pass-through, while R drops them entirely. This shifts the empirical Bayes prior for shared features. The per-group math is independently verified as correct.

## Python Performance

| Dataset   | Algorithm | Mode | Block | Sort     | Time (s) | Memory (MB) | Features out | Corrected | Pass-through |
| --------- | --------- | ---- | ----- | -------- | -------- | ----------- | ------------ | --------- | ------------ |
| small     | limma     | --   | --    | --       | 0.020    | 72.9        | 999          | 980       | 19           |
| medium    | limma     | --   | --    | --       | 0.240    | 98.1        | 5000         | 4869      | 131          |
| large     | limma     | --   | --    | --       | 0.520    | 122.5       | 10000        | 9525      | 475          |
| small     | ComBat    | 1    | --    | --       | 0.040    | 72.5        | 999          | 980       | 19           |
| small     | ComBat    | 2    | --    | --       | 0.020    | 72.8        | 999          | 980       | 19           |
| small     | ComBat    | 3    | --    | --       | 0.100    | 72.7        | 999          | 980       | 19           |
| small     | ComBat    | 4    | --    | --       | 0.100    | 73.3        | 999          | 980       | 19           |
| medium    | ComBat    | 1    | --    | --       | 0.710    | 97.5        | 5000         | 4869      | 131          |
| medium    | ComBat    | 2    | --    | --       | 0.300    | 98.0        | 5000         | 4869      | 131          |
| medium    | ComBat    | 3    | --    | --       | 1.280    | 98.2        | 5000         | 4869      | 131          |
| medium    | ComBat    | 4    | --    | --       | 1.240    | 98.4        | 5000         | 4869      | 131          |
| large     | ComBat    | 1    | --    | --       | 1.660    | 121.5       | 10000        | 9525      | 475          |
| large     | ComBat    | 2    | --    | --       | 0.620    | 121.4       | 10000        | 9525      | 475          |
| large     | ComBat    | 3    | --    | --       | 7.880    | 120.7       | 10000        | 9525      | 475          |
| large     | ComBat    | 4    | --    | --       | 7.700    | 121.4       | 10000        | 9525      | 475          |
| small     | ComBat    | 1    | 2     | --       | 0.010    | 72.7        | 999          | 812       | 187          |
| small     | ComBat    | 2    | 2     | --       | 0.010    | 72.7        | 999          | 812       | 187          |
| small     | ComBat    | 3    | 2     | --       | 0.070    | 72.5        | 999          | 812       | 187          |
| small     | ComBat    | 4    | 2     | --       | 0.070    | 72.9        | 999          | 812       | 187          |
| medium    | ComBat    | 1    | 2     | --       | 0.090    | 93.7        | 5000         | 5000      | 0            |
| medium    | ComBat    | 2    | 2     | --       | 0.050    | 93.9        | 5000         | 5000      | 0            |
| medium    | ComBat    | 3    | 2     | --       | 0.870    | 94.5        | 5000         | 5000      | 0            |
| medium    | ComBat    | 4    | 2     | --       | 0.860    | 95.7        | 5000         | 5000      | 0            |
| large     | ComBat    | 1    | 2     | --       | 0.800    | 120.3       | 10000        | 9926      | 74           |
| large     | ComBat    | 2    | 2     | --       | 0.300    | 121.0       | 10000        | 9926      | 74           |
| large     | ComBat    | 3    | 2     | --       | 7.550    | 121.1       | 10000        | 9926      | 74           |
| large     | ComBat    | 4    | 2     | --       | 7.480    | 120.2       | 10000        | 9926      | 74           |
| medium    | ComBat    | 1    | 2     | sparsity | 0.090    | 98.9        | 5000         | 5000      | 0            |
| medium    | ComBat    | 2    | 2     | sparsity | 0.050    | 96.6        | 5000         | 5000      | 0            |
| medium    | ComBat    | 3    | 2     | sparsity | 0.910    | 98.6        | 5000         | 5000      | 0            |
| medium    | ComBat    | 4    | 2     | sparsity | 0.870    | 98.6        | 5000         | 5000      | 0            |
| large     | ComBat    | 1    | 2     | sparsity | 0.830    | 128.4       | 10000        | 9926      | 74           |
| large     | ComBat    | 2    | 2     | sparsity | 0.320    | 127.8       | 10000        | 9926      | 74           |
| large     | ComBat    | 3    | 2     | sparsity | 7.500    | 118.7       | 10000        | 9926      | 74           |
| large     | ComBat    | 4    | 2     | sparsity | 7.470    | 127.7       | 10000        | 9926      | 74           |
| scp_small | limma     | --   | --    | --       | 1.150    | 195.3       | 3000         | 21        | 2979         |
| scp_large | limma     | --   | --    | --       | 4.580    | 1686.0      | 5000         | 0         | 5000         |
| scp_small | ComBat    | 1    | --    | --       | 1.150    | 195.5       | 3000         | 21        | 2979         |
| scp_small | ComBat    | 2    | --    | --       | 1.160    | 195.3       | 3000         | 21        | 2979         |
| scp_small | ComBat    | 3    | --    | --       | 1.150    | 194.4       | 3000         | 21        | 2979         |
| scp_small | ComBat    | 4    | --    | --       | 1.180    | 195.3       | 3000         | 21        | 2979         |
| scp_large | ComBat    | 1    | --    | --       | 4.560    | 1687.2      | 5000         | 0         | 5000         |
| scp_large | ComBat    | 2    | --    | --       | 4.580    | 1687.3      | 5000         | 0         | 5000         |
| scp_large | ComBat    | 3    | --    | --       | 4.530    | 1687.3      | 5000         | 0         | 5000         |
| scp_large | ComBat    | 4    | --    | --       | 4.550    | 1686.7      | 5000         | 0         | 5000         |
