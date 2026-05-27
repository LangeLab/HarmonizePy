"""Input and output helpers.

Read/write HarmonizR-compatible data and batch description files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_HAVE_PYARROW: bool = True
try:
    import pyarrow  # noqa: F401
except ImportError:
    _HAVE_PYARROW = False

# CSV/TSV reading: pyarrow engine is faster for tall/narrow data but slower
# for very wide data (thousands of columns).  We sample the header line to
# count columns and pick the engine accordingly.
_WIDE_COLUMN_THRESHOLD = 500


def _count_columns(path: str, sep: str) -> int:
    """Return the number of columns in the header line of a delimited file.

    Reads a single line from the file (header), splits by *sep*, and
    returns the length.  Cost is microseconds for any reasonable file.
    """
    with Path(path).open() as fh:
        header = fh.readline()
    return len(header.split(sep))


def _use_pyarrow_csv(path: str, sep: str) -> bool:
    """Return ``True`` if pyarrow engine should be used for this CSV/TSV file.

    PyArrow is preferred for files with ``<= _WIDE_COLUMN_THRESHOLD`` columns
    (where it is typically faster) and slower for very wide files.
    """
    return bool(_HAVE_PYARROW and _count_columns(path, sep) <= _WIDE_COLUMN_THRESHOLD)


def _convert_pyarrow_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pyarrow-backed columns and index to standard numpy dtypes in-place."""
    if not _HAVE_PYARROW:
        return df
    from pandas import ArrowDtype

    # Columns
    pa_cols = [col for col in df.columns if isinstance(df[col].dtype, ArrowDtype)]
    for col in pa_cols:
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            df[col] = df[col].astype(int)
        elif pd.api.types.is_float_dtype(dtype):
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(str)

    # Index
    if isinstance(df.index.dtype, ArrowDtype):
        df.index = df.index.astype(str)

    return df


def read_main_data(path: str) -> pd.DataFrame:
    """Read a features x samples data matrix.

    Format is inferred from the file extension:

    - ``.tsv`` / ``.txt`` or any other extension: tab-separated (default, R-compatible).
    - ``.csv``: comma-separated.
    - ``.parquet`` / ``.pq``: Apache Parquet (requires ``pyarrow``).

    The file must have a header row and use the first column as row names
    (feature identifiers), matching R
    ``read.table(sep="\\t", header=TRUE, row.names=1)``.

    Parameters
    ----------
    path : str
        Path to the data file.

    Returns
    -------
    DataFrame
        Numeric matrix with feature names as index and sample names as columns.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ImportError
        If *path* is parquet and ``pyarrow`` is not installed.

    Examples
    --------
    >>> from harmonizepy.io import read_main_data
    >>> df = read_main_data("data.tsv")  # doctest: +SKIP
    >>> df = read_main_data("data.csv")  # doctest: +SKIP
    >>> df = read_main_data("data.parquet")  # doctest: +SKIP
    """
    ext = Path(path).suffix.lower()

    if ext in (".parquet", ".pq"):
        if not _HAVE_PYARROW:
            raise ImportError(
                "Parquet reading requires pyarrow: pip install harmonizepy[io]"
            )
        df = pd.read_parquet(path, engine="pyarrow")
    elif ext == ".csv":
        if _HAVE_PYARROW and _count_columns(path, ",") <= _WIDE_COLUMN_THRESHOLD:
            df = pd.read_csv(path, sep=",", index_col=0, engine="pyarrow", dtype_backend="pyarrow")
        else:
            df = pd.read_csv(path, sep=",", index_col=0)
    else:
        if _HAVE_PYARROW and _count_columns(path, "\t") <= _WIDE_COLUMN_THRESHOLD:
            df = pd.read_csv(path, sep="\t", index_col=0, engine="pyarrow", dtype_backend="pyarrow")
        else:
            df = pd.read_csv(path, sep="\t", index_col=0)

    df = _convert_pyarrow_dtypes(df)

    df = df.dropna(how="all", axis=0)

    # Normalise index name: pyarrow engine may set it to empty string
    # where the default C parser leaves it as None.
    if df.index.name == "":
        df.index.name = None

    logger.debug(
        "Read data matrix: %d features x %d samples from %s",
        df.shape[0],
        df.shape[1],
        path,
    )
    return df


def read_description(path: str) -> pd.DataFrame:
    """Read a CSV batch description file.

    Parameters
    ----------
    path : str
        Path to the comma-separated file.  Expected columns: ``ID``
        (sample name), ``sample`` (integer index), ``batch`` (integer
        batch label).

    Returns
    -------
    DataFrame
        Batch description with columns ID, sample, batch.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.

    Examples
    --------
    >>> from harmonizepy.io import read_description
    >>> desc = read_description("batch.csv")  # doctest: +SKIP
    """
    if _use_pyarrow_csv(path, ","):
        desc = pd.read_csv(path, engine="pyarrow", dtype_backend="pyarrow")
    else:
        desc = pd.read_csv(path)
    desc = _convert_pyarrow_dtypes(desc)

    logger.debug(
        "Read description: %d samples, %d columns from %s", desc.shape[0], desc.shape[1], path
    )
    return desc


def write_output(df: pd.DataFrame, path: str) -> None:
    """Write a corrected matrix to *path*.

    Format is inferred from the file extension:

    - ``.tsv`` / ``.txt``: tab-separated (default).
    - ``.csv``: comma-separated.
    - ``.parquet`` / ``.pq``: Apache Parquet (requires ``pyarrow``).

    Parameters
    ----------
    df : DataFrame
        Features x samples.
    path : str
        Output file path.

    Raises
    ------
    OSError
        If *path* is not writable.
    ImportError
        If *path* is parquet and ``pyarrow`` is not installed.

    Examples
    --------
    >>> import pandas as pd
    >>> from harmonizepy.io import write_output
    >>> df = pd.DataFrame({"s1": [1.0, 2.0]}, index=["f1", "f2"])
    >>> write_output(df, "/tmp/out.tsv")  # doctest: +SKIP
    """
    ext = Path(path).suffix.lower()
    if ext in (".parquet", ".pq"):
        if not _HAVE_PYARROW:
            raise ImportError(
                "Parquet output requires pyarrow: pip install harmonizepy[io]"
            )
        df.to_parquet(path, index=True, engine="pyarrow")
    elif ext == ".csv":
        df.to_csv(path)
    else:
        df.to_csv(path, sep="\t")

    logger.debug("Wrote output: %d features x %d samples to %s", df.shape[0], df.shape[1], path)
