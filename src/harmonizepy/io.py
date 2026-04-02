"""Input and output helpers.

Read/write HarmonizR-compatible data and batch description files.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd


def read_main_data(path: str) -> pd.DataFrame:
    """Read a features x samples data matrix.

    Format is inferred from the file extension:

    - ``.tsv`` / ``.txt`` or any other extension: tab-separated (default, R-compatible).
    - ``.csv``: comma-separated.
    - ``.feather`` / ``.ftr``: Apache Feather (requires ``pyarrow``).

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

    Examples
    --------
    >>> from harmonizepy.io import read_main_data
    >>> df = read_main_data("data.tsv")  # doctest: +SKIP
    >>> df = read_main_data("data.csv")  # doctest: +SKIP
    >>> df = read_main_data("data.feather")  # doctest: +SKIP
    """
    ext = Path(path).suffix.lower()
    if ext in (".feather", ".ftr"):
        # Feather encodes the index as a column named after the original index.
        raw = pd.read_feather(path)
        # The first column is always the row-name (feature identifier) column.
        df = raw.set_index(raw.columns[0])
    elif ext == ".csv":
        df = pd.read_csv(path, sep=",", index_col=0)
    else:
        # Default: tab-separated (.tsv, .txt, or unrecognised extension)
        df = pd.read_csv(path, sep="\t", index_col=0)

    # Drop completely empty rows/columns (mirrors janitor::remove_empty)
    df = cast(pd.DataFrame, df.dropna(how="all", axis=0).dropna(how="all", axis=1))
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
    return pd.read_csv(path)


def write_output(df: pd.DataFrame, path: str) -> None:
    """Write a corrected matrix as TSV (feature names in first column).

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

    Examples
    --------
    >>> import pandas as pd
    >>> from harmonizepy.io import write_output
    >>> df = pd.DataFrame({"s1": [1.0, 2.0]}, index=["p1", "p2"])
    >>> write_output(df, "/tmp/out.tsv")  # doctest: +SKIP
    """
    df.to_csv(path, sep="\t")
