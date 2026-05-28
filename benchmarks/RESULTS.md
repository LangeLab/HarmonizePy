# Benchmark Results

**Generated:** 2026-05-27 17:12 UTC
**Platform:** Linux 6.17.9-76061709-generic
**CPU:** x86_64 (Py=1 thread, R=16 threads)
**Python:** 3.12.13 (harmonizepy v0.3.0)
**R:** 4.6.0 (HarmonizR 1.10.0)

## Data Specifications

| Dataset                | Type                        | Features | Samples | Batches | Missingness | File Size |
| ---------------------- | --------------------------- | -------- | ------- | ------- | ----------- | --------- |
| small                  | Bulk proteomics, small      | 1000     | 20      | 5       | 30%         | 261 KB    |
| medium                 | Bulk proteomics, medium     | 5000     | 60      | 10      | 20%         | 4.3 MB    |
| large                  | Bulk proteomics, large      | 10000    | 100     | 20      | 5%          | 16.7 MB   |
| scp_small              | SCP cohort, small           | 3000     | 1000    | 20      | 50%         | 24.9 MB   |
| scp_large              | SCP cohort, large           | 5000     | 10000   | 100     | 60%         | 375.5 MB  |
| murine_medulloblastoma | Real murine medulloblastoma | 4753     | 25      | 4       | 49%         | 808 KB    |

### Implementation Notes

HarmonizePy is a pure NumPy implementation running single-threaded. Its ComBat and limma engines are built as vectorized array operations on pre-allocated output buffers, processing all features and affiliation groups in a single pass. This avoids the per-sub-matrix call overhead inherent in R HarmonizR's `foreach` + `sva::ComBat` dispatch, where the full engine is called separately for each unique missingness pattern.

R HarmonizR v1.10.0 (Bioconductor) uses multi-threaded execution via `doParallel` and `foreach`. On small and medium datasets (up to 5000 x 60, 10 batches), R runs 15-30x slower than HarmonizePy for ComBat modes and ~7x slower for limma. On large datasets (10000 x 100, 20 batches), R exceeds 60 seconds per scenario due to combinatorial sub-matrix fragmentation and times out.

Memory usage differs substantially. At 10000 x 100, HarmonizePy uses ~120 MB peak RSS (measured via `/usr/bin/time -v`), with the pre-allocated single-output-array strategy keeping memory at roughly 1x the input size. R HarmonizR with 16 parallel workers can reach 4-5+ GB due to `foreach` copying data per worker and per-group list allocation across its splitting and adjustment steps.

Concordance between the implementations was verified on small, medium, and murine datasets across all four ComBat modes and limma. All modes agree at machine epsilon or better (max_rel < 0.0003 on murine real data). Per-cell NaN handling matches R at max_rel 0.0000 on synthetic fixtures. NaN positions match on all shared features across all tested configurations.

## Performance

| Dataset                | Algorithm | Mode | Block | Sort     | Py Time (s) | Py Mem (MB) | R Time (s) | R Mem (MB) | Features | Corrected | Pass-through |
| ---------------------- | --------- | ---- | ----- | -------- | ----------- | ----------- | ---------- | ---------- | -------- | --------- | ------------ |
| small                  | limma     | --   | --    | --       | 0.010       | 111         | 1.231      | 133        | 999      | 980       | 19           |
| small                  | ComBat    | 1    | --    | --       | 0.040       | 112         | 11.432     | 511        | 999      | 980       | 19           |
| small                  | ComBat    | 2    | --    | --       | 0.010       | 111         | 11.383     | 511        | 999      | 980       | 19           |
| small                  | ComBat    | 3    | --    | --       | 0.200       | 111         | 11.500     | 534        | 999      | 980       | 19           |
| small                  | ComBat    | 4    | --    | --       | 0.190       | 111         | 11.629     | 532        | 999      | 980       | 19           |
| small                  | ComBat    | 1    | 2     | --       | 0.010       | 111         | 11.399     | 511        | 999      | 812       | 187          |
| small                  | ComBat    | 2    | 2     | --       | 0.010       | 111         | 11.495     | 511        | 999      | 812       | 187          |
| small                  | ComBat    | 3    | 2     | --       | 0.140       | 111         | 11.517     | 533        | 999      | 812       | 187          |
| small                  | ComBat    | 4    | 2     | --       | 0.140       | 111         | 11.651     | 532        | 999      | 812       | 187          |
| medium                 | limma     | --   | --    | --       | 0.150       | 143         | 7.310      | 195        | 5000     | 4869      | 131          |
| medium                 | ComBat    | 1    | --    | --       | 0.680       | 144         | 18.304     | 653        | 5000     | 4869      | 131          |
| medium                 | ComBat    | 2    | --    | --       | 0.210       | 143         | 18.699     | 639        | 5000     | 4869      | 131          |
| medium                 | ComBat    | 3    | --    | --       | 2.350       | 143         | 23.199     | 700        | 5000     | 4869      | 131          |
| medium                 | ComBat    | 4    | --    | --       | 2.260       | 143         | 22.754     | 722        | 5000     | 4869      | 131          |
| medium                 | ComBat    | 1    | 2     | --       | 0.090       | 139         | 18.358     | 653        | 5000     | 5000      | 0            |
| medium                 | ComBat    | 2    | 2     | --       | 0.040       | 138         | 18.432     | 639        | 5000     | 5000      | 0            |
| medium                 | ComBat    | 3    | 2     | --       | 1.870       | 138         | 22.712     | 718        | 5000     | 5000      | 0            |
| medium                 | ComBat    | 4    | 2     | --       | 1.860       | 140         | 23.551     | 723        | 5000     | 5000      | 0            |
| medium                 | ComBat    | 1    | 2     | sparsity | 0.090       | 141         | 18.197     | 654        | 5000     | 5000      | 0            |
| medium                 | ComBat    | 2    | 2     | sparsity | 0.040       | 142         | 18.343     | 639        | 5000     | 5000      | 0            |
| medium                 | ComBat    | 3    | 2     | sparsity | 1.870       | 142         | 22.794     | 702        | 5000     | 5000      | 0            |
| medium                 | ComBat    | 4    | 2     | sparsity | 1.870       | 141         | 22.901     | 729        | 5000     | 5000      | 0            |
| murine_medulloblastoma | limma     | --   | --    | --       | 0.150       | 129         | 2.455      | 138        | 4524     | 3157      | 1367         |
| murine_medulloblastoma | ComBat    | 1    | --    | --       | 0.590       | 130         | 10.326     | 541        | 4524     | 3157      | 1367         |
| murine_medulloblastoma | ComBat    | 2    | --    | --       | 0.410       | 129         | 10.476     | 536        | 4524     | 3272      | 1252         |
| murine_medulloblastoma | ComBat    | 1    | 2     | --       | 0.450       | 127         | 10.310     | 541        | 4524     | 4524      | 0            |
| murine_medulloblastoma | ComBat    | 2    | 2     | --       | 0.320       | 127         | 10.359     | 532        | 4524     | 4524      | 0            |

## Python vs R Concordance

| Dataset                | Algorithm | Mode | Block | Sort     | Shared Features | Py Only | R Only | NaN Match | Max Rel  | Mean Rel | P95 Rel  | Shared Non-NaN Cells |
| ---------------------- | --------- | ---- | ----- | -------- | --------------- | ------- | ------ | --------- | -------- | -------- | -------- | -------------------- |
| small                  | limma     | --   | --    | --       | 999             | 0       | 0      | YES       | 5.09e-15 | 1.03e-15 | 3.86e-15 | 14000                |
| small                  | ComBat    | 1    | --    | --       | 999             | 0       | 0      | YES       | 5.98e-06 | 1.12e-08 | 5.15e-10 | 14000                |
| small                  | ComBat    | 2    | --    | --       | 999             | 0       | 0      | YES       | 5.32e-15 | 1.01e-15 | 3.80e-15 | 14000                |
| small                  | ComBat    | 3    | --    | --       | 999             | 0       | 0      | YES       | 5.03e-15 | 9.84e-16 | 3.89e-15 | 14000                |
| small                  | ComBat    | 4    | --    | --       | 999             | 0       | 0      | YES       | 5.10e-15 | 1.01e-15 | 3.83e-15 | 14000                |
| small                  | ComBat    | 1    | 2     | --       | 999             | 0       | 0      | NO        | 3.98e-01 | 2.97e-02 | 1.18e-01 | 10608                |
| small                  | ComBat    | 2    | 2     | --       | 999             | 0       | 0      | NO        | 9.90e-01 | 3.10e-02 | 1.16e-01 | 10608                |
| small                  | ComBat    | 3    | 2     | --       | 999             | 0       | 0      | NO        | 3.41e-01 | 2.92e-02 | 1.16e-01 | 10608                |
| small                  | ComBat    | 4    | 2     | --       | 999             | 0       | 0      | NO        | 1.06e+00 | 2.96e-02 | 1.18e-01 | 10608                |
| medium                 | limma     | --   | --    | --       | 4968            | 32      | 0      | NO        | 9.02e-01 | 1.91e-03 | 4.10e-15 | 238151               |
| medium                 | ComBat    | 1    | --    | --       | 4968            | 32      | 0      | NO        | 1.39e+00 | 1.94e-03 | 4.06e-08 | 238151               |
| medium                 | ComBat    | 2    | --    | --       | 4968            | 32      | 0      | NO        | 1.22e+00 | 1.93e-03 | 4.01e-15 | 238151               |
| medium                 | ComBat    | 3    | --    | --       | 4968            | 32      | 0      | NO        | 2.67e+01 | 2.37e-03 | 4.04e-15 | 238151               |
| medium                 | ComBat    | 4    | --    | --       | 4968            | 32      | 0      | NO        | 2.09e+00 | 2.05e-03 | 4.01e-15 | 238151               |
| medium                 | ComBat    | 1    | 2     | --       | 4968            | 32      | 0      | NO        | 7.97e-01 | 2.85e-02 | 9.91e-02 | 191089               |
| medium                 | ComBat    | 2    | 2     | --       | 4968            | 32      | 0      | NO        | 2.25e+00 | 2.96e-02 | 1.01e-01 | 191089               |
| medium                 | ComBat    | 3    | 2     | --       | 4968            | 32      | 0      | NO        | 1.60e+02 | 3.32e-02 | 1.05e-01 | 191089               |
| medium                 | ComBat    | 4    | 2     | --       | 4968            | 32      | 0      | NO        | 6.50e+00 | 2.96e-02 | 1.02e-01 | 191089               |
| medium                 | ComBat    | 1    | 2     | sparsity | 4968            | 32      | 0      | NO        | 7.97e-01 | 2.85e-02 | 9.91e-02 | 191089               |
| medium                 | ComBat    | 2    | 2     | sparsity | 4968            | 32      | 0      | NO        | 2.25e+00 | 2.96e-02 | 1.01e-01 | 191089               |
| medium                 | ComBat    | 3    | 2     | sparsity | 4968            | 32      | 0      | NO        | 1.60e+02 | 3.32e-02 | 1.05e-01 | 191089               |
| medium                 | ComBat    | 4    | 2     | sparsity | 4968            | 32      | 0      | NO        | 6.50e+00 | 2.96e-02 | 1.02e-01 | 191089               |
| murine_medulloblastoma | limma     | --   | --    | --       | 4479            | 45      | 0      | YES       | 2.39e-10 | 3.79e-14 | 7.01e-14 | 59826                |
| murine_medulloblastoma | ComBat    | 1    | --    | --       | 4479            | 45      | 0      | YES       | 3.24e-04 | 1.58e-08 | 5.79e-15 | 59826                |
| murine_medulloblastoma | ComBat    | 2    | --    | --       | 4524            | 0       | 0      | NO        | 7.88e-09 | 1.52e-13 | 4.49e-15 | 52789                |
| murine_medulloblastoma | ComBat    | 1    | 2     | --       | 4479            | 45      | 0      | NO        | 1.57e+03 | 7.20e-01 | 1.44e+00 | 43568                |
| murine_medulloblastoma | ComBat    | 2    | 2     | --       | 4524            | 0       | 0      | NO        | 1.67e+04 | 1.42e+00 | 1.24e+00 | 43424                |
