"""ComBat integration layer.

Thin public API that maps HarmonizR-style integer modes (1–4) to the
underlying :func:`harmonizepy.combat.combat` parameters.
"""

from __future__ import annotations

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
        Features × samples.  **Must have no missing values.**
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
    """
    if mode not in _MODE_MAP:
        raise ValueError(f"Invalid ComBat mode {mode}. Must be 1–4.")

    batch_labels = np.asarray(batch_labels, dtype=np.intp).ravel()
    if len(np.unique(batch_labels)) < 2:
        return sub_df.copy()

    corrected = combat(
        sub_df.to_numpy(dtype=np.float64),
        batch_labels,
        ref_batch=ref_batch,
        **_MODE_MAP[mode],
    )

    return pd.DataFrame(corrected, index=sub_df.index, columns=sub_df.columns)
