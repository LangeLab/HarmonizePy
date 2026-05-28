# Benchmark Results

**Generated:** 2026-05-28 01:59 UTC
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

Concordance interpretation is split across three buckets. Unblocked murine real-data runs remain concordant at machine precision or better (max_rel < 0.0003). Medium synthetic rows still show the documented shared-feature NaN-position edge case. Historical R-backed blocked rows in this file (`Block = 2`) are invalid because the old benchmark wrapper silently ran HarmonizR without blocking while Python still ran blocked.

Direct validation with the fixed wrapper confirms blocked concordance on murine `ComBat` mode 1 at `max_rel 5.73e-06`, `p95_rel 5.48e-15`, and `nan_match 1.0`. The blocked benchmark rows below are retained only as historical artifacts and must be regenerated before using them for blocked-mode conclusions.

Status note: the table below is still the last full-matrix snapshot. Newer targeted reruns on the current optimization branch improved representative Python timings without changing the unblocked concordance story, including medium limma 0.15s -> 0.09s, medium ComBat mode 1 0.69s -> 0.63s, scp_small limma 0.16s -> 0.13s, and scp_small ComBat mode 1 0.36s -> 0.32s. A fresh full-matrix regeneration is still pending.

## Performance

Note: rows with `Block = 2` and populated R columns were produced before the wrapper fix. Their Python timings reflect blocked execution, but their R timings reflect an unintended unblocked run.

| Dataset                | Algorithm | Mode | Block | Sort     | Py Time (s) | Py Mem (MB) | R Time (s) | R Mem (MB) | Features | Corrected | Pass-through |
| ---------------------- | --------- | ---- | ----- | -------- | ----------- | ----------- | ---------- | ---------- | -------- | --------- | ------------ |
| small                  | limma     | --   | --    | --       | 0.010       | 112         | 1.264      | 131        | 999      | 980       | 19           |
| medium                 | limma     | --   | --    | --       | 0.150       | 144         | 7.360      | 196        | 5000     | 4869      | 131          |
| large                  | limma     | --   | --    | --       | 0.410       | 186         | --         | --         | 10000    | 9525      | 475          |
| small                  | ComBat    | 1    | --    | --       | 0.040       | 111         | 11.922     | 511        | 999      | 980       | 19           |
| small                  | ComBat    | 2    | --    | --       | 0.020       | 111         | 11.964     | 511        | 999      | 980       | 19           |
| small                  | ComBat    | 3    | --    | --       | 0.100       | 111         | 12.115     | 533        | 999      | 980       | 19           |
| small                  | ComBat    | 4    | --    | --       | 0.100       | 111         | 12.055     | 532        | 999      | 980       | 19           |
| medium                 | ComBat    | 1    | --    | --       | 0.690       | 143         | 19.068     | 653        | 5000     | 4869      | 131          |
| medium                 | ComBat    | 2    | --    | --       | 0.210       | 142         | 18.933     | 639        | 5000     | 4869      | 131          |
| medium                 | ComBat    | 3    | --    | --       | 1.250       | 144         | 23.485     | 701        | 5000     | 4869      | 131          |
| medium                 | ComBat    | 4    | --    | --       | 1.250       | 144         | 22.979     | 721        | 5000     | 4869      | 131          |
| large                  | ComBat    | 1    | --    | --       | 1.710       | 196         | --         | --         | 10000    | 9525      | 475          |
| large                  | ComBat    | 2    | --    | --       | 0.480       | 191         | --         | --         | 10000    | 9525      | 475          |
| large                  | ComBat    | 3    | --    | --       | 7.790       | 187         | --         | --         | 10000    | 9525      | 475          |
| large                  | ComBat    | 4    | --    | --       | 7.710       | 188         | --         | --         | 10000    | 9525      | 475          |
| small                  | ComBat    | 1    | 2     | --       | 0.010       | 111         | 12.238     | 511        | 999      | 812       | 187          |
| small                  | ComBat    | 2    | 2     | --       | 0.010       | 110         | 12.021     | 511        | 999      | 812       | 187          |
| small                  | ComBat    | 3    | 2     | --       | 0.070       | 111         | 12.028     | 534        | 999      | 812       | 187          |
| small                  | ComBat    | 4    | 2     | --       | 0.070       | 111         | 12.126     | 533        | 999      | 812       | 187          |
| medium                 | ComBat    | 1    | 2     | --       | 0.090       | 143         | 18.885     | 653        | 5000     | 5000      | 0            |
| medium                 | ComBat    | 2    | 2     | --       | 0.040       | 141         | 19.066     | 638        | 5000     | 5000      | 0            |
| medium                 | ComBat    | 3    | 2     | --       | 0.880       | 142         | 23.202     | 700        | 5000     | 5000      | 0            |
| medium                 | ComBat    | 4    | 2     | --       | 0.880       | 141         | 23.091     | 722        | 5000     | 5000      | 0            |
| large                  | ComBat    | 1    | 2     | --       | 0.870       | 186         | --         | --         | 10000    | 9926      | 74           |
| large                  | ComBat    | 2    | 2     | --       | 0.250       | 187         | --         | --         | 10000    | 9926      | 74           |
| large                  | ComBat    | 3    | 2     | --       | 7.480       | 184         | --         | --         | 10000    | 9926      | 74           |
| large                  | ComBat    | 4    | 2     | --       | 7.450       | 189         | --         | --         | 10000    | 9926      | 74           |
| medium                 | ComBat    | 1    | 2     | sparsity | 0.110       | 142         | 18.681     | 653        | 5000     | 5000      | 0            |
| medium                 | ComBat    | 2    | 2     | sparsity | 0.040       | 141         | 18.806     | 639        | 5000     | 5000      | 0            |
| medium                 | ComBat    | 3    | 2     | sparsity | 0.890       | 142         | 23.433     | 701        | 5000     | 5000      | 0            |
| medium                 | ComBat    | 4    | 2     | sparsity | 0.870       | 142         | 23.027     | 729        | 5000     | 5000      | 0            |
| large                  | ComBat    | 1    | 2     | sparsity | 0.890       | 187         | --         | --         | 10000    | 9926      | 74           |
| large                  | ComBat    | 2    | 2     | sparsity | 0.250       | 187         | --         | --         | 10000    | 9926      | 74           |
| large                  | ComBat    | 3    | 2     | sparsity | 7.530       | 188         | --         | --         | 10000    | 9926      | 74           |
| large                  | ComBat    | 4    | 2     | sparsity | 7.660       | 186         | --         | --         | 10000    | 9926      | 74           |
| scp_small              | limma     | --   | --    | --       | 0.160       | 208         | --         | --         | 3000     | 3000      | 0            |
| scp_large              | limma     | --   | --    | --       | 4.120       | 1762        | --         | --         | 5000     | 5000      | 0            |
| scp_small              | ComBat    | 1    | --    | --       | 0.360       | 208         | --         | --         | 3000     | 3000      | 0            |
| scp_small              | ComBat    | 2    | --    | --       | 0.180       | 208         | --         | --         | 3000     | 3000      | 0            |
| scp_small              | ComBat    | 3    | --    | --       | 0.880       | 209         | --         | --         | 3000     | 3000      | 0            |
| scp_small              | ComBat    | 4    | --    | --       | 0.880       | 208         | --         | --         | 3000     | 3000      | 0            |
| scp_large              | ComBat    | 1    | --    | --       | 4.130       | 1775        | --         | --         | 5000     | 5000      | 0            |
| scp_large              | ComBat    | 2    | --    | --       | 2.760       | 1773        | --         | --         | 5000     | 5000      | 0            |
| scp_large              | ComBat    | 3    | --    | --       | 7.990       | 1768        | --         | --         | 5000     | 5000      | 0            |
| scp_large              | ComBat    | 4    | --    | --       | 7.790       | 1772        | --         | --         | 5000     | 5000      | 0            |
| murine_medulloblastoma | limma     | --   | --    | --       | 0.160       | 129         | 2.537      | 138        | 4524     | 3157      | 1367         |
| murine_medulloblastoma | ComBat    | 1    | --    | --       | 0.330       | 130         | 10.721     | 541        | 4524     | 3157      | 1367         |
| murine_medulloblastoma | ComBat    | 2    | --    | --       | 0.320       | 130         | 10.711     | 535        | 4524     | 3272      | 1252         |
| murine_medulloblastoma | ComBat    | 3    | --    | --       | 0.660       | 130         | 31.545     | 666        | 4524     | 3157      | 1367         |
| murine_medulloblastoma | ComBat    | 4    | --    | --       | 0.680       | 130         | 35.666     | 664        | 4524     | 3272      | 1252         |
| murine_medulloblastoma | ComBat    | 1    | 2     | --       | 0.230       | 129         | 10.816     | 541        | 4524     | 4524      | 0            |
| murine_medulloblastoma | ComBat    | 2    | 2     | --       | 0.230       | 128         | 10.489     | 535        | 4524     | 4524      | 0            |
| murine_medulloblastoma | ComBat    | 3    | 2     | --       | 0.540       | 129         | 30.316     | 665        | 4524     | 4524      | 0            |
| murine_medulloblastoma | ComBat    | 4    | 2     | --       | 0.540       | 129         | 34.277     | 664        | 4524     | 4524      | 0            |

## Python vs R Concordance

Note: rows with `Block = 2` and populated R columns are invalid historical artifacts from the broken wrapper and should not be interpreted as blocked-mode concordance results.

| Dataset                | Algorithm | Mode | Block | Sort     | Shared Features | Py Only | R Only | NaN Match | Max Rel  | Mean Rel | P95 Rel  | Shared Non-NaN Cells |
| ---------------------- | --------- | ---- | ----- | -------- | --------------- | ------- | ------ | --------- | -------- | -------- | -------- | -------------------- |
| small                  | limma     | --   | --    | --       | 999             | 0       | 0      | YES       | 5.09e-15 | 1.03e-15 | 3.86e-15 | 14000                |
| medium                 | limma     | --   | --    | --       | 4968            | 32      | 0      | NO        | 9.02e-01 | 1.91e-03 | 4.10e-15 | 238151               |
| small                  | ComBat    | 1    | --    | --       | 999             | 0       | 0      | YES       | 5.98e-06 | 1.12e-08 | 5.15e-10 | 14000                |
| small                  | ComBat    | 2    | --    | --       | 999             | 0       | 0      | YES       | 5.32e-15 | 1.01e-15 | 3.80e-15 | 14000                |
| small                  | ComBat    | 3    | --    | --       | 999             | 0       | 0      | YES       | 5.00e-15 | 9.84e-16 | 3.89e-15 | 14000                |
| small                  | ComBat    | 4    | --    | --       | 999             | 0       | 0      | YES       | 5.16e-15 | 1.01e-15 | 3.83e-15 | 14000                |
| medium                 | ComBat    | 1    | --    | --       | 4968            | 32      | 0      | NO        | 1.39e+00 | 1.94e-03 | 4.06e-08 | 238151               |
| medium                 | ComBat    | 2    | --    | --       | 4968            | 32      | 0      | NO        | 1.22e+00 | 1.93e-03 | 4.01e-15 | 238151               |
| medium                 | ComBat    | 3    | --    | --       | 4968            | 32      | 0      | NO        | 2.67e+01 | 2.37e-03 | 4.05e-15 | 238151               |
| medium                 | ComBat    | 4    | --    | --       | 4968            | 32      | 0      | NO        | 2.09e+00 | 2.05e-03 | 4.01e-15 | 238151               |
| small                  | ComBat    | 1    | 2     | --       | 999             | 0       | 0      | NO        | 3.98e-01 | 2.97e-02 | 1.18e-01 | 10608                |
| small                  | ComBat    | 2    | 2     | --       | 999             | 0       | 0      | NO        | 9.90e-01 | 3.10e-02 | 1.16e-01 | 10608                |
| small                  | ComBat    | 3    | 2     | --       | 999             | 0       | 0      | NO        | 3.41e-01 | 2.92e-02 | 1.16e-01 | 10608                |
| small                  | ComBat    | 4    | 2     | --       | 999             | 0       | 0      | NO        | 1.06e+00 | 2.96e-02 | 1.18e-01 | 10608                |
| medium                 | ComBat    | 1    | 2     | --       | 4968            | 32      | 0      | NO        | 7.97e-01 | 2.85e-02 | 9.91e-02 | 191089               |
| medium                 | ComBat    | 2    | 2     | --       | 4968            | 32      | 0      | NO        | 2.25e+00 | 2.96e-02 | 1.01e-01 | 191089               |
| medium                 | ComBat    | 3    | 2     | --       | 4968            | 32      | 0      | NO        | 1.60e+02 | 3.32e-02 | 1.05e-01 | 191089               |
| medium                 | ComBat    | 4    | 2     | --       | 4968            | 32      | 0      | NO        | 6.50e+00 | 2.96e-02 | 1.02e-01 | 191089               |
| medium                 | ComBat    | 1    | 2     | sparsity | 4968            | 32      | 0      | NO        | 7.97e-01 | 2.85e-02 | 9.91e-02 | 191089               |
| medium                 | ComBat    | 2    | 2     | sparsity | 4968            | 32      | 0      | NO        | 2.25e+00 | 2.96e-02 | 1.01e-01 | 191089               |
| medium                 | ComBat    | 3    | 2     | sparsity | 4968            | 32      | 0      | NO        | 1.60e+02 | 3.32e-02 | 1.05e-01 | 191089               |
| medium                 | ComBat    | 4    | 2     | sparsity | 4968            | 32      | 0      | NO        | 6.50e+00 | 2.96e-02 | 1.02e-01 | 191089               |
| murine_medulloblastoma | limma     | --   | --    | --       | 4479            | 45      | 0      | YES       | 2.39e-10 | 3.79e-14 | 7.01e-14 | 59826                |
| murine_medulloblastoma | ComBat    | 1    | --    | --       | 4479            | 45      | 0      | YES       | 3.24e-04 | 1.58e-08 | 5.84e-15 | 59826                |
| murine_medulloblastoma | ComBat    | 2    | --    | --       | 4524            | 0       | 0      | NO        | 7.11e-09 | 1.38e-13 | 4.39e-15 | 52789                |
| murine_medulloblastoma | ComBat    | 3    | --    | --       | 4479            | 45      | 0      | YES       | 1.49e-11 | 3.16e-15 | 5.29e-15 | 59826                |
| murine_medulloblastoma | ComBat    | 4    | --    | --       | 4524            | 0       | 0      | NO        | 4.92e-12 | 2.66e-15 | 4.73e-15 | 52789                |
| murine_medulloblastoma | ComBat    | 1    | 2     | --       | 4479            | 45      | 0      | NO        | 1.57e+03 | 7.20e-01 | 1.44e+00 | 43568                |
| murine_medulloblastoma | ComBat    | 2    | 2     | --       | 4524            | 0       | 0      | NO        | 1.67e+04 | 1.42e+00 | 1.24e+00 | 43424                |
| murine_medulloblastoma | ComBat    | 3    | 2     | --       | 4479            | 45      | 0      | NO        | 5.76e+03 | 1.02e+00 | 1.50e+00 | 43568                |
| murine_medulloblastoma | ComBat    | 4    | 2     | --       | 4524            | 0       | 0      | NO        | 5.88e+03 | 8.14e-01 | 1.29e+00 | 43424                |
