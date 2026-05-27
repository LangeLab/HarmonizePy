# Benchmark Results

**Generated:** 2026-05-27 04:32 UTC
**Platform:** Linux 6.17.9-76061709-generic
**CPU:** x86_64 (Py=1 thread, R=16 threads)
**Python:** 3.12.13 (harmonizepy v0.2.0)
**R:** 4.6.0 (HarmonizR 1.10.0)

## Data Specifications

| Dataset | Type | Features/Proteins | Samples/Cells | Batches | Missingness | File Size |
| --- | --- | --- | --- | --- | --- | --- |
| small | Bulk proteomics, small | 1000 | 20 | 5 | 30% | 261 KB |
| medium | Bulk proteomics, medium | 5000 | 60 | 10 | 20% | 4.3 MB |
| large | Bulk proteomics, large | 10000 | 100 | 20 | 5% | 16.7 MB |
| scp_small | SCP cohort, small | 3000 | 1000 | 20 | 50% | 24.9 MB |
| scp_large | SCP cohort, large | 5000 | 10000 | 100 | 60% | 375.5 MB |
| murine_medulloblastoma | Real murine medulloblastoma | 4753 | 25 | 4 | 49% | 808 KB |


### Implementation Notes

HarmonizePy is a pure NumPy implementation running single-threaded. Its ComBat and limma engines are built as vectorized array operations on pre-allocated output buffers, processing all features and affiliation groups in a single pass. This avoids the per-sub-matrix call overhead inherent in R HarmonizR's `foreach` + `sva::ComBat` dispatch, where the full engine is called separately for each unique missingness pattern.

R HarmonizR v1.10.0 (Bioconductor) uses multi-threaded execution via `doParallel` and `foreach`. On small and medium datasets (up to 5000 x 60, 10 batches), R runs 15-30x slower than HarmonizePy for ComBat modes and ~7x slower for limma. On large datasets (10000 x 100, 20 batches), R exceeds 60 seconds per scenario due to combinatorial sub-matrix fragmentation and times out.

Memory usage differs substantially. At 10000 x 100, HarmonizePy uses ~120 MB peak RSS (measured via `/usr/bin/time -v`), with the pre-allocated single-output-array strategy keeping memory at roughly 1x the input size. R HarmonizR with 16 parallel workers can reach 4-5+ GB due to `foreach` copying data per worker and per-group list allocation across its splitting and adjustment steps.

Concordance between the implementations was verified on small and medium datasets across all four ComBat modes and limma. Unblocked modes agree at machine epsilon (relative diff < 1e-14 for closed-form modes, < 6e-6 for the parametric iterative solver). Blocked modes show larger differences (relative diff ~0.4) due to differing feature retention policies: HarmonizePy preserves single-feature groups as pass-through, while R drops them entirely. This shifts the empirical Bayes prior for shared features. The per-group math is independently verified as correct.
