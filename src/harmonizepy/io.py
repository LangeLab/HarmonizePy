"""Input and output helpers.

Read/write HarmonizR-compatible data and batch description files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


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
        raw = pd.read_feather(path)
        df = raw.set_index(raw.columns[0])
    elif ext == ".csv":
        df = pd.read_csv(path, sep=",", index_col=0)
    else:
        df = pd.read_csv(path, sep="\t", index_col=0)

    df = df.dropna(how="all", axis=0)

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
    desc = pd.read_csv(path)
    logger.debug(
        "Read description: %d samples, %d columns from %s", desc.shape[0], desc.shape[1], path
    )
    return desc


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
    >>> df = pd.DataFrame({"s1": [1.0, 2.0]}, index=["f1", "f2"])
    >>> write_output(df, "/tmp/out.tsv")  # doctest: +SKIP
    """
    df.to_csv(path, sep="\t")
    logger.debug("Wrote output: %d features x %d samples to %s", df.shape[0], df.shape[1], path)
