"""Input and output helpers.

Read/write HarmonizR-compatible data and batch description files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
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
    """Convert pyarrow-backed and nullable columns and the index to NumPy dtypes in-place.

    An integer column with a missing cell becomes float64 with NaN, since NumPy
    integers cannot hold a missing value; an integer column without one stays
    int64. A column of blanks becomes float64. Both readers and Parquet then
    give the same dtypes for the same content.
    """
    if not _HAVE_PYARROW:
        return df
    from pandas import ArrowDtype

    # Columns
    ext_cols = [col for col in df.columns if pd.api.types.is_extension_array_dtype(df[col].dtype)]
    for col in ext_cols:
        dtype = df[col].dtype
        if df[col].isna().all():
            # A column of blanks has no type of its own (pyarrow reads it as null).
            df[col] = df[col].astype(float)
        elif pd.api.types.is_integer_dtype(dtype):
            df[col] = df[col].astype(float if df[col].isna().any() else int)
        elif pd.api.types.is_float_dtype(dtype):
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(str)

    # Index
    if isinstance(df.index.dtype, ArrowDtype):
        df.index = df.index.astype(str)

    return df


def _names_for_log(names: list[str]) -> str:
    """Return up to ten names for a log line."""
    shown = ", ".join(names[:10])
    return shown + (", ..." if len(names) > 10 else "")


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated features.

    A feature is a repeat when its name and all of its values match an earlier
    row, with missing cells in the same places; the repeat is removed and logged
    at INFO level. Features with different names are kept even when their values
    are identical. A name that appears more than once with different values
    raises ``ValueError``. The input frame is not changed.

    Parameters
    ----------
    df : DataFrame
        Features x samples matrix, with feature names as the index.

    Returns
    -------
    DataFrame
        The matrix without the removed rows, in the original order.

    Raises
    ------
    ValueError
        If a feature name appears more than once with different values.
    """
    has_repeated_name = df.index.duplicated(keep=False)
    if not has_repeated_name.any():
        return df

    # One pass over the rows that share a name: compare each row with the first
    # row of the same name, with missing cells equal to missing cells.
    shared = df[has_repeated_name]
    codes, _ = pd.factorize(shared.index, use_na_sentinel=False)
    if all(pd.api.types.is_numeric_dtype(dtype) for dtype in shared.dtypes):
        values = shared.to_numpy(dtype=np.float64)
        _, first_row = np.unique(codes, return_index=True)
        reference = values[first_row[codes]]
        same = (values == reference) | (np.isnan(values) & np.isnan(reference))
        differs = ~same.all(axis=1)
    else:
        # Mixed dtypes: drop exact repeats of (name code, values); a code left twice
        # has two value sets. Columns are renamed to positions so that the code
        # column, labelled -1, cannot collide with a sample name.
        keyed = shared.set_axis(range(shared.shape[1]), axis=1)
        keyed[-1] = codes
        distinct = keyed.drop_duplicates()
        conflict_codes = distinct[-1][distinct[-1].duplicated()]
        differs = np.isin(codes, conflict_codes.to_numpy())
    if differs.any():
        name = shared.index[np.flatnonzero(differs)[0]]
        raise ValueError(
            f"duplicate feature names with different values: {str(name)!r} appears more "
            "than once. Each feature name must have one set of values."
        )

    is_repeat = df.index.duplicated(keep="first")
    removed_names = [str(n) for n in pd.unique(df.index[is_repeat])]
    logger.info(
        "Removed %d repeated feature(s) with identical values: %s",
        int(is_repeat.sum()),
        _names_for_log(removed_names),
    )
    return df.loc[~is_repeat]


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
    ValueError
        If a feature name appears more than once with different values. A
        feature repeated with identical values is removed and logged.

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
            raise ImportError("Parquet reading requires pyarrow: pip install harmonizepy[io]")
        df = pd.read_parquet(path, engine="auto")
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

    # Repeats are checked before blank rows are dropped, so that a name repeated
    # once with values and once blank is reported, not silently resolved.
    df = clean_features(df)
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
        batch label).  Without the names ``ID`` and ``batch``, the first
        column holds the sample IDs and the third the batch labels.

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
            raise ImportError("Parquet output requires pyarrow: pip install harmonizepy[io]")
        df.to_parquet(path, index=True, engine="pyarrow")
    elif ext == ".csv":
        df.to_csv(path)
    else:
        df.to_csv(path, sep="\t")

    logger.debug("Wrote output: %d features x %d samples to %s", df.shape[0], df.shape[1], path)
