# Benchmark Results

Benchmark summary for HarmonizePy 0.4.0: runtime and memory of the Python implementation, and concordance with HarmonizR 1.10.0 on the datasets where an R baseline is practical. The raw outputs of a run go to `benchmarks/results/`, which is not tracked; rerun the commands below to reproduce them on your own hardware.

## How to reproduce

The summary combines two passes: an R-backed parity pass over the validation datasets, and a Python-only pass over the two largest datasets.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python -m benchmarks.bench run \
  --datasets small medium dia murine scp_small \
  --with-r --with-r-cores 1 \
  --n-reps 3 \
  --out benchmarks/results/benchmark_r_parity.json \
  --md benchmarks/results/benchmark_r_parity.md

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python -m benchmarks.bench run \
  --datasets large scp_large \
  --n-reps 1 \
  --no-warmup \
  --out benchmarks/results/benchmark_python_extremes.json \
  --md benchmarks/results/benchmark_python_extremes.md
```

- The parity pass attaches cached HarmonizR baselines run on one R core (`python -m benchmarks.bench cache-r` fills the cache).
- The Python-only pass uses one timed repetition without warmup, so its memory stability is reported as `INSUFFICIENT-DATA` by construction.
- Python runs on one thread, with BLAS limited to one thread by the environment variables above.

## Datasets

| Dataset | Kind | Features | Samples | Batches | Missing | Size | R baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `small` | Synthetic bulk | 1000 | 20 | 5 | 30% | 261 KB | yes |
| `medium` | Synthetic bulk | 5000 | 60 | 10 | 20% | 4.3 MB | yes |
| `large` | Synthetic bulk | 10000 | 100 | 20 | 5% | 16.7 MB | no |
| `murine` | Real (medulloblastoma) | 4753 | 25 | 4 | natural | 808 KB | yes |
| `dia` | Real (DIA proteomics) | 8470 | 36 | 2 | natural | 4.8 MB | yes |
| `scp_small` | Synthetic SCP | 3000 | 1000 | 20 | 50% | 24.9 MB | yes |
| `scp_large` | Synthetic SCP | 5000 | 10000 | 100 | 60% | 375.5 MB | no |

- Missingness in the synthetic sets removes whole batches per feature (bulk) or follows abundance (SCP).
- `dia` has two batches, so it runs without blocking only; `block=2` would form a single block.
- `large` and `scp_large` have no R baseline; they show how the Python implementation scales.

## Environment

- AMD Ryzen 9 3950X (16 cores, 32 threads), 125 GiB RAM, Linux.
- Python 3.13.15, NumPy 2.5.4 with scipy-openblas 0.3.34, pandas 3.1.0, HarmonizePy 0.4.0.
- R baselines: HarmonizR 1.10.0, sva 3.60.0, Bioconductor 3.23, R 4.6.0, one core, on the same hardware; the cached outputs and timings come from that installation.

HarmonizR runs on one core because its `doParallel` fork workers inherit the multi-threaded OpenBLAS pool and compete with its threads, so a multi-core timing would measure that interaction rather than the correction.

## Runtime

Wall time of `harmonize()` in seconds: the median of three repetitions for Python, one run for R. The R column is the cached single-core HarmonizR baseline from the same hardware; it was not measured in the same batch as the Python column, so the two compare only roughly.

| Scenario | Python (s) | R (s) |
| --- | --- | --- |
| `small` limma | 0.021 | 1.01 |
| `small` ComBat 1 | 0.118 | 5.58 |
| `medium` ComBat 1 | 1.91 | 19.5 |
| `medium` ComBat 1, block 2 | 0.217 | 5.32 |
| `medium` ComBat 3, block 2 | 0.246 | 22.8 |
| `murine` ComBat 1 | 0.477 | 6.67 |
| `murine` ComBat 4, block 2 | 0.376 | 28.2 |
| `dia` ComBat 1 | 0.476 | 12.3 |
| `dia` ComBat 3 | 1.75 | 456 |
| `scp_small` limma | 0.172 | 49.7 |
| `scp_small` ComBat 1 | 0.723 | 54.9 |

Python-only pass, one repetition:

| Scenario | Python (s) |
| --- | --- |
| `large` limma | 0.401 |
| `large` ComBat 1 | 3.08 |
| `large` ComBat 2 | 0.869 |
| `large` ComBat 2, block 2 | 0.497 |
| `large` ComBat 3 | 6.26 |
| `large` ComBat 3, block 2 | 4.83 |
| `scp_large` limma | 2.38 |
| `scp_large` ComBat 1 | 4.25 |
| `scp_large` ComBat 3 | 4.80 |

- Python was faster than single-core HarmonizR in every parity scenario, from about 5 times (`murine` ComBat 2) to about 290 times (`scp_small` limma).
- Blocking cuts the number of groups: on `medium` ComBat 1 it takes the run from 1.91 s to 0.217 s, and on `large` ComBat 2 from 0.869 s to 0.497 s.
- The non-parametric modes (3 and 4) compare each feature with every other feature in its group, so their cost grows with the square of the group size; they are the slowest modes on `dia`, `murine`, and `large`.

## Memory

Peak Python heap (tracemalloc) of one `harmonize()` call, against the R heap reported by `gc()`.

| Dataset | Python min (MB) | Python max (MB) | R min (MB) | R max (MB) |
| --- | --- | --- | --- | --- |
| `small` | 0.7 | 1.4 | 67 | 352 |
| `medium` | 8.6 | 9.9 | 71 | 369 |
| `murine` | 4.4 | 6.4 | 68 | 359 |
| `dia` | 18.9 | 21.2 | 71 | 369 |
| `scp_small` | 70.7 | 73.6 | 98 | 381 |
| `large` | 33.9 | 46.4 | -- | -- |
| `scp_large` | 1171 | 1179 | -- | -- |

- Minimum and maximum over the scenarios of each dataset; `--` means no R baseline.
- Python's retained RSS stayed flat across repetitions in all but two parity scenarios, `dia` ComBat 1 (356 KB) and `scp_small` ComBat 4 (2.7 MB), where the last samples still grew slightly.
- R runs every scenario in one process, so later R scenarios can reuse memory allocated by earlier ones.

## Concordance with HarmonizR

Python against R on the features both outputs contain. Relative difference is `|py - r| / |r|` over cells where both have a value.

| Scenario group | Max relative difference | Missing cells match |
| --- | --- | --- |
| `small`, all modes, with and without block 2 | 6e-06 | yes |
| `dia`, all modes | 5e-14 | yes |
| `scp_small` limma, ComBat 1 and 2 | 7e-06 | yes |
| `scp_small` ComBat 3 and 4 | 5e-15 | no |
| `murine` limma, ComBat 1 and 3 | 3e-04 | yes |
| `murine` ComBat 2 and 4 | 7e-09 | no |
| `medium` block 2, with and without sorting | 7e-06 | yes |
| `medium` without blocking, all modes | 27 | no |

- ComBat mode 1 differs by up to 3e-04 relative (`murine`); its parametric solver stops at a convergence tolerance. The other modes differ by at most 7e-09.
- `murine` ComBat 2 and 4: HarmonizR returns 870 features entirely missing although the input has values for them; HarmonizePy corrects them. HarmonizR also drops 229 features that HarmonizePy keeps: 221 with values in a single batch, kept raw, and 8 empty rows.
- `medium` without blocking: the differences come from unique removal. HarmonizR also moves a feature with a pattern of its own into the pattern of a single other feature; HarmonizePy moves it only into a pattern that two or more features share. Of the 109 features with limma that differ by more than 1e-4 relative or in their missing cells, 99 are kept raw by HarmonizePy as alone in their pattern and corrected by HarmonizR, and the other 10 are in groups whose size those moves change. With blocking there are fewer patterns and the outputs agree. The parity document describes the rule.
- `scp_small` ComBat 3 and 4: HarmonizR returns 1256 of the 3000 features entirely missing, although the input has values for them; HarmonizePy corrects them. On every cell both outputs have, the values agree to rounding.
