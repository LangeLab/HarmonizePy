# Benchmark Results

**Generated:** 2026-05-26 22:37 UTC
**Platform:** Linux 6.17.9-76061709-generic, Python 3.12.13
**CPU:** x86_64
**R available:** True

## Python Performance

| Dataset | Algorithm | Mode | Block | Sort | Time (s) | Memory (MB) | Features out | Corrected | Pass-through |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | ComBat | 1 | -- | -- | 0.427 | 72.4 | 999 | 980 | 19 |
| small | ComBat | 2 | -- | -- | 0.421 | 73.0 | 999 | 980 | 19 |
| small | ComBat | 3 | -- | -- | 0.502 | 72.6 | 999 | 980 | 19 |
| small | ComBat | 4 | -- | -- | 0.519 | 73.3 | 999 | 980 | 19 |
| medium | ComBat | 1 | -- | -- | 2.108 | 96.2 | 4999 | 4947 | 52 |
| medium | ComBat | 2 | -- | -- | 1.338 | 96.9 | 4999 | 4947 | 52 |
| medium | ComBat | 3 | -- | -- | 2.160 | 96.9 | 4999 | 4947 | 52 |
| medium | ComBat | 4 | -- | -- | 2.111 | 96.4 | 4999 | 4947 | 52 |
| large | ComBat | 1 | -- | -- | 6.585 | 131.9 | 10000 | 1407 | 8593 |
| large | ComBat | 2 | -- | -- | 6.478 | 132.3 | 10000 | 1407 | 8593 |
| large | ComBat | 3 | -- | -- | 6.748 | 132.7 | 10000 | 1407 | 8593 |
| large | ComBat | 4 | -- | -- | 6.788 | 132.8 | 10000 | 1407 | 8593 |
| small | limma | -- | -- | -- | 0.420 | 73.4 | 999 | 980 | 19 |
| medium | limma | -- | -- | -- | 1.204 | 96.7 | 4999 | 4947 | 52 |
| large | limma | -- | -- | -- | 6.490 | 132.7 | 10000 | 1407 | 8593 |
| small | ComBat | 1 | 2 | -- | 0.351 | 72.9 | 999 | 812 | 187 |
| medium | ComBat | 1 | 2 | -- | 0.758 | 92.2 | 4999 | 4999 | 0 |
| large | ComBat | 1 | 2 | -- | 2.732 | 116.7 | 10000 | 10000 | 0 |
| medium | ComBat | 1 | 2 | sparsity | 0.748 | 94.9 | 4999 | 4999 | 0 |
| large | ComBat | 1 | 2 | sparsity | 2.716 | 124.1 | 10000 | 10000 | 0 |
