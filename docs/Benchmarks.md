# Benchmarks

This page summarizes benchmark design, dataset catalog, runtime expectations, and how to reproduce results locally.

---

## Design philosophy

The benchmark suite answers four separate questions:

- Does HarmonizePy stay numerically aligned with HarmonizR on small and medium synthetic bulk workloads?
- Does that parity hold on real matrices such as DIA proteomics and murine medulloblastoma data?
- Is there at least one R-backed SCP anchor, so the single-cell regime is covered?
- How does the Python implementation behave on stress-scale inputs where HarmonizR baselines are not practical?

The suite is deliberately split into two passes rather than one combined run.

---

## Dataset catalog

| Dataset | Shape | Batches | Missingness | Type | Benchmark pass |
| --- | --- | --- | --- | --- | --- |
| `small` | 1000 x 20 | 5 | 30% per-batch | Synthetic bulk | R-backed parity |
| `medium` | 5000 x 60 | 10 | 20% per-batch | Synthetic bulk | R-backed parity |
| `dia` | 8470 x 36 | 2 | Structural | Real DIA proteomics | R-backed parity |
| `murine` | 4753 x 25 | 4 | None injected | Real murine medulloblastoma | R-backed parity |
| `scp_small` | 3000 x 1000 | 20 | 50% abundance-pattern | Synthetic SCP | R-backed parity |
| `large` | 10000 x 100 | 20 | 5% per-batch | Synthetic bulk stress | Python-only |
| `scp_large` | 5000 x 10000 | 100 | 60% abundance-pattern | Synthetic SCP stress | Python-only |

**R-backed parity datasets** are benchmarked against single-core HarmonizR baselines. Concordance and runtime ratios are both reported.

**Python-only datasets** are too large for reliable HarmonizR baselines. They are showcased as operational stress runs, not parity evidence.

---

## Runtime headline

All results are from a single machine: AMD Ryzen 9 3950X (16 physical cores), 125 GiB RAM, Python 3.12.13 with NumPy 2.4.4 linked against scipy-openblas 0.3.31.

| Dataset | Representative scenario | Python | R (1 core) | Runtime ratio (R/Python) |
| --- | --- | --- | --- | --- |
| `small` | limma | 0.015 s | 1.007 s | ~67x |
| `medium` | ComBat m2, block=2 | 0.070 s | 5.361 s | ~77x |
| `dia` | ComBat m3 | 1.714 s | 456 s | ~266x |
| `murine` | ComBat m4, block=2 | 0.366 s | 28.2 s | ~77x |
| `scp_small` | ComBat m1 | 1.018 s | 54.9 s | ~54x |

On every benchmarked scenario in this suite, Python runtime was shorter than single-core HarmonizR. The largest runtime differences appear on non-parametric ComBat (modes 3 and 4) and on SCP-scale data.

---

## Concordance summary

| Scenario | Max relative difference | Status |
| --- | --- | --- |
| `small` ComBat m1 | 5.98e-06 | Near-exact |
| `dia` ComBat m3 | 5.12e-14 | Effectively exact |
| `scp_small` ComBat m1 | 7.04e-06 | Near-exact |
| `medium` ComBat m1, block=2 | 7.39e-06 | Near-exact |
| `medium` ComBat m3 (unblocked) | 2.67e+01 | **Known outlier** |

The unblocked medium ComBat m3 scenario is the clearest parity caveat in the current suite. The blocked medium pipeline (`block=2`) is in good shape.

---

## Why R is limited to one core

HarmonizR supports parallelism via `doParallel` and `foreach`. The current benchmark pins it to 1 core because `doParallel` with Unix forked workers and OpenBLAS compete for threads when worker count is increased. This produces worse wall-clock performance and higher memory, not a speedup. The 1-core baseline is the fair and stable reference point for this machine.

---

## How to reproduce

Run the R-backed parity pass and the Python-only extreme pass separately:

```bash
# R-backed parity pass (requires a working R + HarmonizR environment)
uv run python -m benchmarks.bench run \
  --datasets small medium dia murine scp_small \
  --with-r --with-r-cores 1 \
  --n-reps 3 \
  --out benchmarks/results/benchmark_r_parity.json \
  --md benchmarks/results/benchmark_r_parity.md

# Python-only extreme pass
uv run python -m benchmarks.bench run \
  --datasets large scp_large \
  --n-reps 1 \
  --no-warmup \
  --out benchmarks/results/benchmark_python_extremes.json \
  --md benchmarks/results/benchmark_python_extremes.md
```

The R-backed pass requires R with `sva`, `limma`, `HarmonizR`, and `seriation` installed (managed via `renv` in the repository). The Python-only pass has no R requirement.

R baseline caches are stored under `benchmarks/results/r_cache/`. If you have already generated them, pass `--use-r-cache` to skip regenerating.

---

## Benchmark CLI

The benchmark system has its own CLI with six subcommands:

- `run`: run Python (and optionally R) benchmarks and produce JSON/Markdown reports
- `cache-r`: pre-generate and cache R baseline outputs for the given datasets
- `validity`: run concordance checks between Python and cached R outputs
- `profile`: profile a specific scenario in-process
- `report`: regenerate Markdown/JSON reports from existing JSON output
- `generate-data`: generate synthetic benchmark datasets

```bash
uv run python -m benchmarks.bench --help
uv run python -m benchmarks.bench run --help
```

Raw JSON and Markdown outputs are written to `benchmarks/results/` and are not tracked in version control. The curated summary in `benchmarks/RESULTS.md` is the authoritative benchmark record for each release.
