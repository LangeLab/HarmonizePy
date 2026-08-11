# Installation

## Requirements

- Python 3.12, 3.13, or 3.14
- numpy >= 1.24
- pandas >= 2.0

No R installation required at runtime.

---

## Install options

### From PyPI

```bash
pip install harmonizepy
```

### From GitHub (latest development version)

```bash
pip install git+https://github.com/LangeLab/HarmonizePy.git
```

### Development install with uv

```bash
uv python install 3.12.13
uv sync --dev --python 3.12.13
```

The `--dev` flag installs pytest, mypy, and ruff alongside the package.

---

## Optional extras

| Extra | Install command | What it adds |
| --- | --- | --- |
| `io` | `pip install harmonizepy[io]` | Parquet output support and faster CSV/TSV reading via `pyarrow >= 14.0` |
| `config` | `pip install harmonizepy[config]` | YAML config files via `pyyaml >= 6.0`. JSON and TOML work without this extra. |
| `completion` | `pip install harmonizepy[completion]` | Shell tab-completion for the CLI via `argcomplete >= 3.0` |
| `all` | `pip install harmonizepy[all]` | All three extras above |

Parquet is the recommended output format for large datasets. Install the `io` extra if you plan to use `--output-format parquet` or write `.parquet` files directly from Python.

---

## Verify the install

```bash
harmonizepy --version
```

Or from Python:

```python
import harmonizepy
print(harmonizepy.__version__)
```

Expected output: `0.3.2`
