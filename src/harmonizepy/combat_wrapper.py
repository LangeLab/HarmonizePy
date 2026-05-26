"""ComBat integration layer.

Thin public API that maps HarmonizR-style integer modes (1-4) to the
underlying ``harmonizepy.combat.combat`` parameters.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from .combat import combat

_MODE_MAP: dict[int, dict[str, bool]] = {
    1: {"par_prior": True, "mean_only": False},
    2: {"par_prior": True, "mean_only": True},
    3: {"par_prior": False, "mean_only": False},
    4: {"par_prior": False, "mean_only": True},
}


def adjust_combat(
    sub_df: pd.DataFrame,
    batch_labels: np.ndarray,
    mode: int = 1,
    ref_batch: int | None = None,
) -> pd.DataFrame:
    """Apply ComBat batch correction to a complete sub-matrix.

    Parameters
    ----------
    sub_df : pd.DataFrame
        Features x samples.  **Must have no missing values.**
    batch_labels : array-like
        Integer batch label per sample (length == ``sub_df.shape[1]``).
    mode : {1, 2, 3, 4}
        1 = parametric, location + scale (default).
        2 = parametric, location only.
        3 = non-parametric, location + scale.
        4 = non-parametric, location only.
    ref_batch : int or None
        Optional reference batch that is left unadjusted.

    Returns
    -------
    pd.DataFrame
        Batch-corrected matrix with the same shape, index, and columns
        as *sub_df*.

    Raises
    ------
    ValueError
        If *sub_df* contains NaN, has < 2 rows, or *mode* is invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> from harmonizepy import adjust_combat
    >>> df = pd.DataFrame({"s1": [1.0, 2.0], "s2": [3.0, 4.0],
    ...                     "s3": [5.0, 6.0], "s4": [7.0, 8.0]})
    >>> corrected = adjust_combat(df, [0, 0, 1, 1], mode=2)
    """
    if mode not in _MODE_MAP:
        raise ValueError(
            f"combat_mode must be 1, 2, 3, or 4, got {mode}. "
            f"Modes 1/2 are parametric; 3/4 are non-parametric."
        )

    batch_labels = np.asarray(batch_labels, dtype=np.intp).ravel()
    if len(np.unique(batch_labels)) < 2:
        return cast(pd.DataFrame, sub_df.copy())

    corrected = combat(
        sub_df.to_numpy(dtype=np.float64),
        batch_labels,
        ref_batch=ref_batch,
        **_MODE_MAP[mode],
    )

    return pd.DataFrame(corrected, index=sub_df.index, columns=sub_df.columns)
