"""Core pipeline entry point for HarmonizePy.

Orchestrates the full batch-correction workflow:

    read → spot missing → split → adjust (ComBat / limma) → rebuild → write

This mirrors the R ``HarmonizR::harmonizR()`` function.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .io import read_main_data, read_description, write_output
from .spotting import spotting_missing_values
from .splitting import splitting
from .rebuild import rebuild


def harmonize(
    data: pd.DataFrame | str,
    description: pd.DataFrame | str,
    *,
    algorithm: Literal["ComBat", "limma"] = "ComBat",
    combat_mode: int = 1,
    output_file: str | None = None,
) -> pd.DataFrame:
    """Run the full HarmonizePy batch-correction pipeline.

    Parameters
    ----------
    data : DataFrame or str
        Features × samples matrix (may contain NaN for structural
        missingness).  If a string, read from that TSV path.
    description : DataFrame or str
        Batch description with columns ``ID``, ``sample``, ``batch``.
        If a string, read from that CSV path.
    algorithm : ``"ComBat"`` or ``"limma"``
        Adjustment algorithm.
    combat_mode : int
        ComBat mode 1–4 (ignored when *algorithm* is ``"limma"``).
    output_file : str or None
        If given, write the corrected matrix to this TSV path.

    Returns
    -------
    DataFrame
        Batch-corrected matrix (features × samples).  NaN values
        remain where the feature lacked sufficient data in a batch.
    """
    # --- Read inputs --------------------------------------------------------
    if isinstance(data, str):
        data = read_main_data(data)
    else:
        data = data.copy()

    if isinstance(description, str):
        description = read_description(description)

    # --- Extract batch labels -----------------------------------------------
    # Column 3 (0-indexed: column "batch") as numeric
    batch_list = description.iloc[:, 2].values.astype(int)

    if len(batch_list) != data.shape[1]:
        raise ValueError(
            f"Description has {len(batch_list)} samples but data has "
            f"{data.shape[1]} columns."
        )

    # --- Remove duplicate features ------------------------------------------
    data = data[~data.index.duplicated(keep="first")]

    # --- Block list (no blocking for now — equals batch_list) ---------------
    block_list = batch_list.copy()

    # --- Determine needed_values --------------------------------------------
    if algorithm == "limma" or combat_mode in (1, 3):
        needed_values = 2
    else:
        needed_values = 1

    # --- Spot missing values ------------------------------------------------
    affiliation_list = spotting_missing_values(
        data, batch_list, block_list, needed_values
    )

    # --- Split, adjust, rebuild ---------------------------------------------
    sub_dfs = splitting(
        affiliation_list, data, batch_list, block_list,
        algorithm=algorithm, combat_mode=combat_mode,
    )
    result = rebuild(sub_dfs)

    # --- Write output -------------------------------------------------------
    if output_file is not None:
        write_output(result, output_file)

    return result
