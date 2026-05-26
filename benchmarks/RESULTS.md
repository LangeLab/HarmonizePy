# Benchmark Results

**Generated:** 2026-05-26 21:36 UTC
**Platform:** Linux 6.17.9-76061709-generic, Python 3.12.13
**CPU:** x86_64
**R available:** True

## Python Performance

| Dataset | Algorithm | Mode | Block | Sort | Time (s) | Memory (MB) | Features out |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small | ComBat | 1 | -- | -- | 0.338 | 73.1 | 1000 |
| small | ComBat | 2 | -- | -- | 0.376 | 72.8 | 1000 |
| small | ComBat | 3 | -- | -- | 0.418 | 73.2 | 1000 |
| small | ComBat | 4 | -- | -- | 0.490 | 72.9 | 1000 |
| small | limma | -- | -- | -- | 0.348 | 73.4 | 1000 |
| small | ComBat | 1 | 2 | -- | 0.321 | 71.8 | 1000 |

## R HarmonizR Performance

| Dataset | Algorithm | Mode | Block | Sort | R Time (s) |
| --- | --- | --- | --- | --- | --- |
| small | ComBat | 1 | -- | -- | 5.022 |
| small | ComBat | 3 | -- | -- | 5.661 |
| small | limma | -- | -- | -- | 0.947 |
| small | ComBat | 1 | 2 | -- | 4.900 |

## Python vs R Concordance

| Dataset | Algorithm | Mode | Max relative diff |
| --- | --- | --- | --- |
| small | ComBat | 1 | 7.26e-10 |
| small | ComBat | 3 | 5.29e-15 |
| small | limma | -- | 5.26e-15 |
| small | ComBat | 1 | 2.36e-01 |
