"""Input and output helpers.

Read/write HarmonizR-compatible data and batch description files.
"""

from __future__ import annotations

import pandas as pd


def read_main_data(path: str) -> pd.DataFrame:
    """Read a TSV data matrix (features × samples).

    Expects tab-separated values with a header row and the first column
    as row names (feature identifiers).  Matches R
    ``read.table(sep="\\t", header=TRUE, row.names=1)``.

    Returns
    -------
    DataFrame
        Numeric matrix with feature names as index and sample names as columns.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    # Drop completely empty rows/columns (mirrors janitor::remove_empty)
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    return df


def read_description(path: str) -> pd.DataFrame:
    """Read a CSV batch description file.

    Expects columns: ``ID`` (sample name), ``sample`` (numeric index),
    ``batch`` (integer batch label).

    Returns
    -------
    DataFrame
        Batch description with columns ID, sample, batch.
    """
    return pd.read_csv(path)


def write_output(df: pd.DataFrame, path: str) -> None:
    """Write a corrected matrix as TSV (feature names in first column).

    Parameters
    ----------
    df : DataFrame
        Features × samples.
    path : str
        Output file path.
    """
    df.to_csv(path, sep="\t")
