"""Core pipeline entry point for HarmonizePy.

Orchestrates the full batch-correction workflow:

    read → [sort] → [block] → spot missing → [unique removal]
        → split → adjust (ComBat / limma) → rebuild → [re-sort] → write

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
from .sorting import sort_batches
from .blocking import build_block_list
from .affiliation import remove_unique_combinations
from .validation import validate_data_matrix, validate_description, validate_harmonize_args


def harmonize(
    data: pd.DataFrame | str | Path,
    description: pd.DataFrame | str | Path,
    *,
    algorithm: Literal["ComBat", "limma"] = "ComBat",
    combat_mode: int = 1,
    needed_values: int | None = None,
    sort: str | None = None,
    block: int | None = None,
    unique_removal: bool = True,
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
    sort : ``"sparsity"``, ``"jaccard"``, ``"seriation"``, or ``None``
        Batch sorting strategy.  Reorders batch columns so similar
        batches become adjacent before blocking.  Has no effect on
        output when *block* is ``None``; allowed without *block* to
        match R behaviour.
    block : int or None
        Block size: number of consecutive (optionally sorted) batches
        to group into one block during dissection.  Must be >= 2 and
        strictly less than the total number of unique batches.
        ``None`` (default) disables blocking.
    unique_removal : bool
        When ``True`` (default), singleton features — those whose
        batch-presence pattern is unique across all features — are
        rescued by cropping to the nearest shared pattern before
        splitting.  Mirrors R's ``ur=TRUE`` default.
    output_file : str, Path, or None
        If given, write the corrected matrix to this TSV path.

    Returns
    -------
    DataFrame
        Batch-corrected matrix (features x samples), in the original
        column order.  NaN values remain where the feature lacked
        sufficient data in a batch.

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
    >>> result = harmonize(df, desc_df, sort="sparsity", block=2)
    """
    # --- Validate basic arguments (before data load) ----------------------
    validate_harmonize_args(
        algorithm, combat_mode, needed_values or 2,
        sort_strategy=sort,
        unique_removal=unique_removal,
    )

    # --- Read inputs -------------------------------------------------------
    if isinstance(data, (str, Path)):
        data = read_main_data(str(data))
    else:
        data = data.copy()

    if isinstance(description, (str, Path)):
        description = read_description(str(description))

    # --- Validate inputs ---------------------------------------------------
    validate_data_matrix(data)
    validate_description(description, data)

    # --- Extract batch labels aligned to data column order -----------------
    sample_to_batch = dict(
        zip(description.iloc[:, 0].astype(str), description.iloc[:, 2].astype(int))
    )
    batch_list = np.array(
        [sample_to_batch[col] for col in data.columns], dtype=np.int64
    )

    # --- Validate block_size now that we know n_batches --------------------
    n_batches = len(np.unique(batch_list))
    validate_harmonize_args(
        algorithm, combat_mode, needed_values or 2,
        block_size=block,
        n_batches=n_batches,
    )

    # --- Determine needed_values -------------------------------------------
    if needed_values is None:
        if algorithm == "limma" or combat_mode in (1, 3):
            needed_values = 2
        else:
            needed_values = 1

    # --- Sort batches ------------------------------------------------------
    col_order: np.ndarray | None = None
    if sort is not None:
        data, batch_list, col_order = sort_batches(
            data, batch_list, strategy=sort, needed_values=needed_values
        )

    # --- Build block list --------------------------------------------------
    if block is not None:
        block_list = build_block_list(batch_list, block_size=block)
    else:
        block_list = batch_list.copy()

    # --- Spot missing values -----------------------------------------------
    affiliation_list = spotting_missing_values(
        data, batch_list, block_list, needed_values
    )

    # --- Unique removal ----------------------------------------------------
    if unique_removal:
        affiliation_list = remove_unique_combinations(affiliation_list)

    # --- Split, adjust, rebuild --------------------------------------------
    sub_dfs = splitting(
        affiliation_list, data, batch_list, block_list,
        algorithm=algorithm, combat_mode=combat_mode,
    )
    result = rebuild(sub_dfs)

    # --- Re-sort columns to original order ---------------------------------
    if col_order is not None:
        result = result.iloc[:, np.argsort(col_order)]

    # --- Write output ------------------------------------------------------
    if output_file is not None:
        write_output(result, str(output_file))

    return result
