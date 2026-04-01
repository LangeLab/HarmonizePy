"""Core pipeline entry point for HarmonizePy.

Orchestrates the full batch-correction workflow:

    read → spot missing → split → adjust (ComBat / limma) → rebuild → write

This mirrors the R ``HarmonizR::harmonizR()`` function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .io import read_main_data, read_description, write_output
from .spotting import spotting_missing_values
from .splitting import splitting
from .rebuild import rebuild
from .validation import validate_data_matrix, validate_description, validate_harmonize_args


def harmonize(
    data: pd.DataFrame | str | Path,
    description: pd.DataFrame | str | Path,
    *,
    algorithm: Literal["ComBat", "limma"] = "ComBat",
    combat_mode: int = 1,
    needed_values: int | None = None,
    output_file: str | Path | None = None,
) -> pd.DataFrame:
    """Run the full HarmonizePy batch-correction pipeline.

    Parameters
    ----------
    data : DataFrame, str, or Path
        Features x samples matrix (may contain NaN for structural
        missingness).  If a path, read from that TSV file.
    description : DataFrame, str, or Path
        Batch description with columns ``ID``, ``sample``, ``batch``.
        If a path, read from that CSV file.
    algorithm : ``"ComBat"`` or ``"limma"``
        Adjustment algorithm.
    combat_mode : int
        ComBat mode 1-4 (ignored when *algorithm* is ``"limma"``).
    needed_values : int or None
        Minimum non-missing values per batch for a feature to enter
        a sub-matrix.  ``None`` (default) auto-selects: 2 for modes
        1, 3 and limma; 1 for modes 2, 4.
    output_file : str, Path, or None
        If given, write the corrected matrix to this TSV path.

    Returns
    -------
    DataFrame
        Batch-corrected matrix (features x samples).  NaN values
        remain where the feature lacked sufficient data in a batch.

    Raises
    ------
    ValueError
        On invalid arguments, mismatched inputs, or duplicate features.

    Examples
    --------
    >>> from harmonizepy import harmonize
    >>> result = harmonize("data.tsv", "batch.csv")
    >>> result = harmonize(df, desc_df, algorithm="limma")
    >>> result = harmonize(df, desc_df, combat_mode=3, needed_values=1)
    """
    # --- Validate arguments -------------------------------------------------
    validate_harmonize_args(algorithm, combat_mode, needed_values or 2)

    # --- Read inputs --------------------------------------------------------
    if isinstance(data, (str, Path)):
        data = read_main_data(str(data))
    else:
        data = data.copy()

    if isinstance(description, (str, Path)):
        description = read_description(str(description))

    # --- Validate inputs ----------------------------------------------------
    validate_data_matrix(data)
    validate_description(description, data)

    # --- Extract batch labels -----------------------------------------------
    batch_list = description.iloc[:, 2].values.astype(int)

    # --- Block list (no blocking for now - equals batch_list) ---------------
    block_list = batch_list.copy()

    # --- Determine needed_values --------------------------------------------
    if needed_values is None:
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
        write_output(result, str(output_file))

    return result
