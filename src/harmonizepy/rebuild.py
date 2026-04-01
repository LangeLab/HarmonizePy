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
        Combined features x samples matrix.  Row order follows the
        order of sub-DataFrames (grouped by affiliation).

    Raises
    ------
    ValueError
        Propagated from ``pandas.concat`` if sub-DataFrames have
        incompatible dtypes.

    Examples
    --------
    >>> import pandas as pd
    >>> from harmonizepy.rebuild import rebuild
    >>> sub1 = pd.DataFrame({"s1": [1.0], "s2": [2.0]}, index=["p1"])
    >>> sub2 = pd.DataFrame({"s1": [3.0], "s2": [4.0]}, index=["p2"])
    >>> rebuild([sub1, sub2]).shape
    (2, 2)
    """
    if not sub_dfs:
        return pd.DataFrame()

    return pd.concat(sub_dfs, axis=0)
