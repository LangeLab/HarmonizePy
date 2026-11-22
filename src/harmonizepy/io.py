"""Reading and writing the data matrix and the batch description.

The data matrix is read from tab- or comma-separated text with pandas' C
parser, or from Parquet with pyarrow, with the first column as feature names,
as R ``read.table(header = TRUE, row.names = 1)`` reads HarmonizR's input.
Repeated features are resolved by ``clean_features``. Outputs are written as
TSV, CSV, or Parquet by suffix.
"""

from __future__ import annotations

import csv
import logging
import secrets
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HAVE_PYARROW: bool = True
try:
    import pyarrow  # noqa: F401
except ImportError:
    _HAVE_PYARROW = False


def _header_fields(path: str, sep: str) -> list[str]:
    """Return the fields of the header line of a delimited file, quotes removed."""
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh, delimiter=sep), [])


def _reject_repeated_sample_names(path: str, sep: str) -> None:
    """Raise ``ValueError`` if a sample name appears twice in the header.

    The parsers rename a repeated header (``s1`` becomes ``s1.1``), so the check
    reads the header itself. The first field names the feature column.
    """
    counts = Counter(_header_fields(path, sep)[1:])
    repeated = sorted(name for name, count in counts.items() if count > 1)
    if repeated:
        raise ValueError(
            f"data contains duplicate sample names: {repeated[:5]}. "
            "Each column must have a unique name."
        )


# Output format for each file suffix; any other suffix writes TSV.
_FORMAT_BY_SUFFIX: dict[str, str] = {
    ".tsv": "tsv",
    ".txt": "tsv",
    ".csv": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
}


def format_from_suffix(path: str) -> str:
    """Return ``"tsv"``, ``"csv"``, or ``"parquet"`` from the suffix of *path*; TSV otherwise."""
    return _FORMAT_BY_SUFFIX.get(Path(path).suffix.lower(), "tsv")


# pandas' default missing-value markers. They are applied to every column
# except the one that holds names, so that a feature called ``NA`` or a sample
# called ``null`` keeps its name.
_MISSING_MARKERS: tuple[str, ...] = (
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
)


def _read_delimited(path: str, sep: str, *, name_col: int, index_col: int | None) -> pd.DataFrame:
    """Read a delimited file with pandas' C parser, keeping column *name_col* as text.

    The name column is read as written: no type inference (``01`` stays
    ``01``) and no missing-value markers (``NA`` stays ``NA``). The other
    columns read blanks and pandas' default markers as missing.

    The C parser runs on the calling thread. pyarrow's CSV reader is not used:
    it starts a thread pool, and on this package's inputs it is no faster on
    narrow files and slower on wide ones.
    """
    n_cols = len(_header_fields(path, sep))
    # Keys are column positions; pandas accepts them, though its stubs name str keys.
    markers: dict[Any, list[str]] = {
        i: list(_MISSING_MARKERS) for i in range(n_cols) if i != name_col
    }
    return pd.read_csv(
        path,
        sep=sep,
        index_col=index_col,
        engine="c",
        dtype={name_col: str},
        keep_default_na=False,
        na_values=markers,
    )


def _to_numpy_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert extension-typed columns and index, as Parquet produces, to NumPy dtypes.

    The frame is changed in place and returned. A column of blanks becomes
    float64; an integer column with a missing cell becomes float64 with NaN,
    since NumPy integers cannot hold a missing value, and without one it stays
    int64; other columns become float64 or text. Text files and Parquet files
    with the same content then give the same dtypes.
    """
    if not _HAVE_PYARROW:
        return df
    from pandas import ArrowDtype

    ext_cols = [col for col in df.columns if pd.api.types.is_extension_array_dtype(df[col].dtype)]
    for col in ext_cols:
        dtype = df[col].dtype
        if df[col].isna().all():
            df[col] = df[col].astype(float)
        elif pd.api.types.is_integer_dtype(dtype):
            df[col] = df[col].astype(float if df[col].isna().any() else int)
        elif pd.api.types.is_float_dtype(dtype):
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(str)

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
    raises ``ValueError``.

    The rows that share a name are checked in one pass: each is compared with
    the first row of its name, missing cells equal to missing cells. For
    frames that are not all numeric, exact repeats of name and values are
    dropped and any name left twice has two value sets.

    The input frame is not changed; when no name repeats, it is returned as is.

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

    shared = df[has_repeated_name]
    codes, _ = pd.factorize(shared.index, use_na_sentinel=False)
    if all(pd.api.types.is_numeric_dtype(dtype) for dtype in shared.dtypes):
        values = shared.to_numpy(dtype=np.float64)
        _, first_row = np.unique(codes, return_index=True)
        reference = values[first_row[codes]]
        same = (values == reference) | (np.isnan(values) & np.isnan(reference))
        differs = ~same.all(axis=1)
    else:
        # Columns renamed to positions so the code column, -1, cannot collide
        # with a sample name.
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


def _read_data_matrix(path: str) -> pd.DataFrame:
    """Read a data matrix file as ``read_main_data`` does, without removing repeats."""
    ext = Path(path).suffix.lower()
    if ext in (".parquet", ".pq"):
        if not _HAVE_PYARROW:
            raise ImportError("Parquet reading requires pyarrow: pip install harmonizepy[io]")
        df = pd.read_parquet(path, engine="pyarrow", use_threads=False)
    else:
        sep = "," if ext == ".csv" else "\t"
        _reject_repeated_sample_names(path, sep)
        df = _read_delimited(path, sep, name_col=0, index_col=0)
    df = _to_numpy_dtypes(df)
    if df.index.name == "":
        df.index.name = None
    return df


def read_main_data(path: str) -> pd.DataFrame:
    """Read a features x samples data matrix.

    Format is inferred from the file extension:

    - ``.tsv`` / ``.txt`` or any other extension: tab-separated (default, R-compatible).
    - ``.csv``: comma-separated.
    - ``.parquet`` / ``.pq``: Apache Parquet (requires ``pyarrow``).

    The file must have a header row and use the first column as row names
    (feature identifiers), matching R
    ``read.table(sep="\\t", header=TRUE, row.names=1)``. Blank cells and pandas'
    default missing-value markers (among them ``NA``, ``NaN``, ``nan``,
    ``null``, ``N/A``, ``#N/A``, ``None``, ``<NA>``) are read as missing in
    the sample columns. Feature names are read as written, as text: a feature
    named ``NA`` keeps that name, and ``001`` stays ``001``. An empty index name is set to ``None``.

    Parameters
    ----------
    path : str
        Path to the data file.

    Returns
    -------
    DataFrame
        Matrix with feature names as index and sample names as columns, one
        row per input row except removed repeats (see ``clean_features``).
        Rows with no value are kept.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ImportError
        If *path* is parquet and ``pyarrow`` is not installed.
    ValueError
        If a sample name appears twice in the header of a delimited file, or
        a feature name appears more than once with different values. A
        feature repeated with identical values is removed and logged.

    Examples
    --------
    >>> from harmonizepy.io import read_main_data
    >>> df = read_main_data("data.tsv")  # doctest: +SKIP
    >>> df = read_main_data("data.csv")  # doctest: +SKIP
    >>> df = read_main_data("data.parquet")  # doctest: +SKIP
    """
    df = clean_features(_read_data_matrix(path))
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
        Path to the comma-separated file, with at least three columns; the ID
        column is read as written, as text, so ``01`` stays ``01``:
        ``ID`` (sample name), ``sample`` (not read), and ``batch`` (whole-number
        batch label). Without the names ``ID`` and ``batch``, the first
        column holds the sample IDs and the third the batch labels.

    Returns
    -------
    DataFrame
        The file's columns as read, one row per sample, with a default
        integer index. ``harmonize()`` checks them with
        ``validate_description``; this function does not.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.

    Examples
    --------
    >>> from harmonizepy.io import read_description
    >>> desc = read_description("batch.csv")  # doctest: +SKIP
    """
    header = _header_fields(path, ",")
    id_col = header.index("ID") if "ID" in header else 0
    desc = _to_numpy_dtypes(_read_delimited(path, ",", name_col=id_col, index_col=None))

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
    >>> write_output(df, "corrected.tsv")  # doctest: +SKIP
    """
    write_frame(df, path, format_from_suffix(path))

    logger.debug("Wrote output: %d features x %d samples to %s", df.shape[0], df.shape[1], path)


def write_frame(df: pd.DataFrame, path: str, fmt: str) -> None:
    """Write *df* to *path* as ``"tsv"``, ``"csv"``, or ``"parquet"``, with the index.

    The file is written through ``write_atomically``, so a failed write leaves
    no partial file at *path* and keeps any file that was there.

    Raises
    ------
    ImportError
        If *fmt* is ``"parquet"`` and ``pyarrow`` is not installed.
    OSError
        If *path* is not writable.
    """
    if fmt == "parquet":
        if not _HAVE_PYARROW:
            raise ImportError("Parquet output requires pyarrow: pip install harmonizepy[io]")
        write_atomically(path, lambda tmp: df.to_parquet(tmp, index=True, engine="pyarrow"))
    elif fmt == "csv":
        write_atomically(path, lambda tmp: df.to_csv(tmp))
    else:
        write_atomically(path, lambda tmp: df.to_csv(tmp, sep="\t"))


def write_atomically(path: str, write: Callable[[str], object]) -> None:
    """Call *write* on a temporary file beside *path*, then rename it to *path*.

    The temporary file is in the target's directory, so the rename is atomic
    on one file system, and it is created with an ordinary open, so it gets
    the permissions the user's umask gives any new file. When *write* fails,
    the temporary file is removed, the error is raised again, and a file that
    already existed at *path* is unchanged.

    Raises
    ------
    OSError
        If the directory of *path* does not exist or is not writable, or the
        rename fails.
    """
    target = Path(path)
    tmp = target.with_name(f".{target.name}.{secrets.token_hex(6)}.tmp")
    try:
        with tmp.open("xb"):
            pass
    except OSError as error:
        # Name the path the user gave, not the hidden temporary file.
        raise type(error)(error.errno, error.strerror, str(target)) from error
    try:
        write(str(tmp))
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
