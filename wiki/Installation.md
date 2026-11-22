# Installation

HarmonizePy is a pure-Python package: NumPy and pandas are its only required dependencies, and it needs no R installation. It runs on Python 3.12 or later, and is tested on Linux with Python 3.12, 3.13 and 3.14, and on macOS and Windows with Python 3.12.

## From GitHub

HarmonizePy is not on PyPI yet. Install it from the repository:

```bash
pip install git+https://github.com/LangeLab/HarmonizePy.git
```

To add optional extras, name them on the same install:

```bash
pip install "harmonizepy[io,config] @ git+https://github.com/LangeLab/HarmonizePy.git"
```

## Optional extras

| Extra        | Adds                 | Needed for               |
| ------------ | -------------------- | ------------------------ |
| `io`         | `pyarrow >= 14.0`    | Parquet input and output |
| `config`     | `pyyaml >= 6.0`      | YAML config files        |
| `completion` | `argcomplete >= 3.0` | shell tab completion     |
| `all`        | all three            |                          |

- TOML and JSON config files need no extra; Python reads them itself.
- Without `io`, a `.parquet` or `.pq` path raises an `ImportError` that names the extra.

## For development

The repository uses [uv](https://docs.astral.sh/uv/), with a lock file:

```bash
git clone https://github.com/LangeLab/HarmonizePy.git
cd HarmonizePy
uv sync --dev
uv run pytest
```

`--dev` adds the test, type-checking and lint tools, among them pytest, mypy and ruff. The tests compare against committed R outputs, so they need no R installation.

## Check the install

```bash
harmonizepy --version
```

prints `harmonizepy` and the installed version. From Python, `harmonizepy.__version__` holds the same.

## Next

[Quick start](Quick-Start) runs one small matrix from start to finish.
