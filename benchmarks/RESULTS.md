# Benchmark Results

This is the durable benchmark summary for the current HarmonizePy benchmark suite. It is intentionally more descriptive than the generated raw Markdown and JSON outputs.

The raw `out` and `md` artifacts under `benchmarks/results/` are local run products from this machine. They are useful for ad hoc inspection, but they are not treated here as public, stable benchmark deliverables. If you want the raw benchmark outputs, rerun the commands below on your own hardware.

## How To Reproduce

The current summary combines two benchmark passes:

- an R-backed parity pass over the representative validation datasets
- a Python-only extreme pass over the largest configured stress datasets

Reproduce them locally with:

```bash
uv run python -m benchmarks.bench run \
  --datasets small medium dia murine scp_small \
  --with-r --with-r-cores 1 \
  --n-reps 3 \
  --out benchmarks/results/benchmark_r_parity.json \
  --md benchmarks/results/benchmark_r_parity.md

uv run python -m benchmarks.bench run \
  --datasets large scp_large \
  --n-reps 1 \
  --no-warmup \
  --out benchmarks/results/benchmark_python_extremes.json \
  --md benchmarks/results/benchmark_python_extremes.md
```

Interpretation notes:

- The parity pass uses cached HarmonizR baselines at 1 R core and includes concordance.
- The extreme pass is Python-only by design because those scales are not currently practical or reliable as HarmonizR parity baselines.
- The extreme pass uses a single timed repetition with no warmup so that the heaviest showcase runs remain tractable.

## Benchmark Design

The dataset mix is deliberate rather than exhaustive. The benchmark suite is trying to answer four separate questions:

- Does HarmonizePy stay numerically aligned with HarmonizR on small and medium synthetic bulk workloads?
- Does that parity still hold on real matrices such as DIA proteomics and murine medulloblastoma data?
- Is there at least one R-backed SCP anchor, so the single-cell style regime is not omitted from parity checks?
- How does the Python implementation behave on stress-scale inputs where current HarmonizR baselines are not representative or do not complete reliably?

That is why the suite keeps `small`, `medium`, `dia`, `murine`, and `scp_small` in the R-backed pass, while `large` and `scp_large` are showcased separately as Python-only runs.

## Dataset Catalog

`small` is the smallest synthetic bulk sanity check. It is `1000 x 20` with `5` batches and `30%` per-batch missingness. At `261 KB`, it is cheap to rerun and serves as the first correctness anchor for parity, blocking, and general pipeline behavior.

`medium` is the main synthetic bulk workload. It is `5000 x 60` with `10` batches and `20%` per-batch missingness, and it includes both blocked and `sparsity`-sorted variants. At `4.3 MB`, it is still manageable enough for repeated parity work while being large enough to surface the current no-block parity gap.

`large` is the synthetic bulk stress dataset. It is `10000 x 100` with `20` batches and `5%` per-batch missingness. At `16.7 MB`, it is used to show how blocking and sorting affect Python behavior at a materially larger scale. It is Python-only because the current HarmonizR baseline is not reliable enough here to make a fair parity claim.

`murine` is the real murine medulloblastoma matrix. It is `4753 x 25` with `4` batches and no synthetic missingness injection. At `808 KB`, it provides a real-data checkpoint that is still small enough to keep the parity pass stable and repeatable.

`dia` is the real DIA proteomics dataset. It is `8470 x 36` with `2` batches and CSV input files. At `4.8 MB`, it is an important real proteomics anchor, but it only appears in no-block scenarios because `block=2` would collapse into a degenerate single block for a two-batch design.

`scp_small` is the smaller synthetic SCP cohort. It is `3000 x 1000` with `20` batches and `50%` abundance-pattern missingness. At `24.9 MB`, it gives the suite one R-backed SCP anchor so the parity story is not limited to bulk-style datasets.

`scp_large` is the large SCP stress dataset. It is `5000 x 10000` with `100` batches and `60%` abundance-pattern missingness. At `375.5 MB`, it is the heaviest configured workload in the suite. It is intentionally Python-only because current HarmonizR runs do not finish reliably enough here, and the memory cost makes the R side a poor reference baseline.

## Benchmark Environment

These numbers were produced on the following machine and software stack.

### System

The benchmark machine was `Linux 6.17.9-76061709-generic` on an `AMD Ryzen 9 3950X 16-Core Processor` with `32` logical CPUs, `16` physical cores, `2` threads per core, `1` socket, and `125 GiB` of RAM. At the time of capture, about `95 GiB` was free and about `99 GiB` was available.

The harness configuration reported Python at `1` process or thread, HarmonizR at `1` core for parity baselines, and system or BLAS threading at `32` threads.

### Python stack

The Python stack for the current `0.3.2` release documentation is Python `3.12.13` with NumPy `2.4.4`, pandas `3.0.2`, and HarmonizePy `0.3.2`. NumPy was linked against `scipy-openblas 0.3.31.188.0` with `DYNAMIC_ARCH`, `NO_AFFINITY`, and `MAX_THREADS=64`. NumPy reported `X86_V3` SIMD support on this machine.

### R stack

The R stack was R `4.6.0`, HarmonizR `1.10.0`, and Bioconductor `3.23`, with `doParallel 1.0.17`, `foreach 1.5.2`, `sva 3.60.0`, `limma 3.68.3`, and `BiocParallel 1.46.0`. R was linked against `/usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3`, with LAPACK from `/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.20.so`.

## Why R Is Limited To One Core Here

The current benchmark policy intentionally pins HarmonizR baselines to `1` R core.

This is not because multi-core HarmonizR is conceptually unsupported. It is because the current `doParallel` plus `foreach` implementation interacts badly with multi-threaded OpenBLAS on Unix. Forked workers inherit the parent's BLAS thread pool, so increasing HarmonizR worker count can cause the workers to compete with BLAS threads instead of doing useful work. In practice that means worse wall-clock performance and much higher memory usage.

The project's own investigation showed the pattern clearly:

- the slowdown is a `doParallel(fork)` plus OpenBLAS oversubscription problem, not a correctness problem in HarmonizR
- the effect becomes worse on workloads that create many sub-matrices, especially blocked runs and higher missingness regimes
- memory usage rises sharply because each worker duplicates a large R process with loaded Bioconductor dependencies and matrix state

That is why the current parity policy treats single-core HarmonizR as the fair and stable reference point. The large Python-only runs are also separated partly for that reason: once the R side becomes dominated by worker and BLAS interaction costs, the result stops being a useful numerical baseline and turns into a systems artifact.

## R-Backed Parity Summary

The R-backed parity cohort is:

- `small`
- `medium`
- `dia`
- `murine`
- `scp_small`

Headline result:

- Python is faster than single-core HarmonizR on every scenario in the parity pass.
- Concordance is excellent for `small`, `dia`, `scp_small`, and the blocked or blocked-plus-sorted `medium` scenarios.
- `murine` remains close to R, with the expected small differences on some mean-only and blocking paths.
- Unblocked `medium` is the one obvious parity outlier and should be read as a known caveat, not representative of the blocked pipeline.

### Parity Speed Commentary

The ratios are useful for scale, but the absolute runtime matters too. The table below keeps one representative row per dataset so the reader can see how long the comparison actually takes in both Python and R.

| Dataset | Representative scenario | Python | R | Ratio context | Key point |
| --- | --- | --- | --- | --- | --- |
| `small` | `limma` | `0.0150s` | `1.007s` | `11.95x` to `67.13x` faster across the dataset | Python already has a large lead on the smallest matrix. |
| `dia` | `ComBat m3` | `1.7140s` | `456.024s` | `239.67x` to `266.06x` faster on non-parametric ComBat | Real proteomics still strongly favors Python. |
| `murine` | `ComBat m4 block=2` | `0.3656s` | `28.223s` | `5.98x` to `77.19x` faster across the dataset | Python stays ahead on both blocked and unblocked paths. |
| `medium` | `ComBat m2 block=2` | `0.0696s` | `5.361s` | `7.27x` to `77.03x` faster depending on mode and blocking | The blocked medium pipeline is especially efficient. |
| `scp_small` | `ComBat m1` | `1.0176s` | `54.932s` | `53.98x` to `286.20x` faster across the dataset | Python remains decisively ahead even on the smaller SCP workload. |

### Parity Memory Commentary

| Dataset | Python memory | R memory | Comment |
| --- | --- | --- | --- |
| `small` | below `1.05 MB` heap | not a concern at this scale | The smallest dataset is cheap on both sides. |
| `medium` | below `9.52 MB` heap | materially higher than Python | Blocking and sorting stay well within a modest Python footprint. |
| `murine` | below `7.26 MB` heap | higher than Python | Real data remains inexpensive on the Python side. |
| `dia` | `18.28 MB` to `31.18 MB` heap | up to about `368.5 MB` heap | Real proteomics shows a large Python memory advantage. |
| `scp_small` | about `74.13 MB` heap | up to about `380.8 MB` heap | This is the clearest SCP memory separation in the parity cohort. |

Two `scp_small` Python scenarios showed growing repeated-run RSS tails:

- `scp_small_ComBat_m1`
- `scp_small_ComBat_m4`

That does not invalidate the benchmark, but it is the main memory-stability note worth retaining from the parity pass.

### Concordance Commentary

Strong parity regions:

- `small` shows machine-precision or near-machine-precision concordance across the reported scenarios.
- `dia` is effectively exact across all reported scenarios.
- `scp_small` stays at machine-precision or near-machine-precision concordance.
- `medium` with `block=2` and `block=2, sort=sparsity` returns near-exact concordance with matching NaN structure.

Important caveat:

- `medium` without blocking is the clear anomaly in the current parity run. The worst row is `medium_ComBat_m3` with max relative difference `2.67e+01`, and the unblocked `medium` rows show NaN mismatch.

Representative concordance points:

| Scenario | Max relative difference | Interpretation |
| --- | --- | --- |
| `small_ComBat_m1` | `5.98e-06` | Near-exact parity. |
| `dia_ComBat_m3` | `5.12e-14` | Effectively exact. |
| `scp_small_ComBat_m1` | `7.04e-06` | Near-exact SCP parity at the smaller scale. |
| `medium_ComBat_m1_b2` | `7.39e-06` | Blocked medium remains near-exact. |
| `medium_ComBat_m3` | `2.67e+01` and NaN mismatch | This is the headline parity outlier. |

The practical reading is simple: the blocked medium pipeline is in good shape, but the unblocked medium path is still the place where the parity story weakens and needs the most caution.

## Python-Only Extreme Runs

The Python-only extreme cohort is:

- `large`
- `scp_large`

These are not presented as R parity evidence. They are operational showcase runs for the largest configured workloads.

### Extreme Speed Commentary

Only Python seconds are shown here because these workloads are intentionally outside the current R-backed parity subset.

| Scenario | Time | Comment |
| --- | --- | --- |
| `large_ComBat_m2_b2` | `0.7868s` | Fastest reported blocked bulk scenario in the extreme pass. |
| `large_limma` | `0.8957s` | Limma remains very competitive on `large`. |
| `large_ComBat_m1` | `6.5268s` | Unblocked parametric ComBat is much heavier than blocked `m2`. |
| `scp_large_limma` | `3.8705s` | Fastest SCP-large scenario. |
| `scp_large_ComBat_m1` | `9.2512s` | Slowest showcase scenario in the current extreme run. |

Main points:

- On `large`, blocking still helps materially. `ComBat m2` drops from `1.7190s` to `0.7868s` with `block=2`.
- On `scp_large`, `limma` is the fastest path at `3.8705s`.
- The slowest showcase scenario is `scp_large_ComBat_m1` at `9.2512s`.

### Extreme Memory Commentary

| Scenario | Python heap | Retained RSS delta | Comment |
| --- | --- | --- | --- |
| `large_ComBat_m1` | `46.65 MB` | `20428 KB` | `large` remains modest in memory terms. |
| `large_limma` | `33.67 MB` | `7332 KB` | Limma is lighter than ComBat on `large`. |
| `scp_large_limma` | `1170.88 MB` | `12416 KB` | SCP-large immediately pushes into gigabyte-scale memory use. |
| `scp_large_ComBat_m1` | `1262.97 MB` | `49416 KB` | Highest retained RSS delta in the showcase. |
| `scp_large_ComBat_m3` | `1252.55 MB` | `25612 KB` | Non-parametric SCP-large remains similarly memory heavy. |

Main points:

- `large` remains comfortably below `50 MB` Python heap in this run.
- `scp_large` is the real memory stress case. Python heap is roughly `1.17 GB` to `1.26 GB` depending on algorithm and mode.
- Because the extreme pass used one timed repetition, RSS stability is reported as `INSUFFICIENT-DATA` there by construction.

## How To Read This Document

Use the parity sections to answer:

- how close HarmonizePy is to HarmonizR on the validated benchmark subset
- where NaN structure and shared-feature behavior remain aligned
- where the current parity caveat lives, namely unblocked `medium`

Use the extreme section to answer:

- what runtime to expect on the largest configured Python-only workloads
- how much blocking helps on `large`
- what memory envelope `scp_large` currently needs

## Bottom Line

- HarmonizePy is already strong on the R-backed benchmark subset, with Python faster than R across the board and excellent concordance outside the known unblocked `medium` outlier set.
- The benchmark design is intentional. It preserves real-data and SCP parity anchors while keeping the stress-scale runs useful rather than forcing weak or misleading R baselines.
- Single-core HarmonizR is used here because it is the stable reference on this stack. Multi-core HarmonizR currently competes with BLAS threads and inflates memory enough to distort the comparison.
- The Python-only extreme runs show that the implementation remains usable on `large` and `scp_large`, with `scp_large` serving as the main stress-test showcase for both runtime and memory.
