"""Rebuild: recombine adjusted sub-DataFrames into the final output.

Mirrors R ``HarmonizR:::rebuild`` which uses ``plyr::rbind.fill``.
"""

from __future__ import annotations

import pandas as pd


def rebuild(sub_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack corrected sub-DataFrames back into a single matrix.

    Parameters
    ----------
    sub_dfs : list[DataFrame]
        Corrected sub-DataFrames from :func:`splitting.splitting`.
        Each has the same columns (full sample set), with NaN in
        columns that were not part of that sub-frame's affiliation.

    Returns
    -------
    DataFrame
        Combined features × samples matrix.  Row order follows the
        order of sub-DataFrames (grouped by affiliation).
    """
    if not sub_dfs:
        return pd.DataFrame()

    return pd.concat(sub_dfs, axis=0)
