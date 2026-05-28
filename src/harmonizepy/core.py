"""Core pipeline entry point for HarmonizePy.

Orchestrates the full batch-correction workflow:

    read → [sort] → [block] → spot missing → [unique removal]
        → split → adjust (ComBat / limma) → concat → [re-sort] → write

This mirrors the R ``HarmonizR::harmonizR()`` function.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from .affiliation import build_affiliation_list, remove_unique_combinations
from .blocking import build_block_list
from .io import read_description, read_main_data, write_output
from .sorting import sort_batches
from .splitting import splitting
from .types import HarmonizeConfig
from .validation import validate_data_matrix, validate_description, validate_harmonize_args

logger = logging.getLogger(__name__)


def harmonize(
    data: pd.DataFrame | str | Path,
    description: pd.DataFrame | str | Path,
    *,
    config: HarmonizeConfig | None = None,
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
    config : HarmonizeConfig or None
        If provided, all algorithm settings are taken from this object.
        Individual keyword arguments (``algorithm``, ``combat_mode``,
        ``needed_values``, ``sort``, ``block``, ``unique_removal``) are
        ignored when *config* is given.  Useful for re-running or
        storing reproducible run configurations.
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
        When ``True`` (default), singleton features (those whose
        batch-presence pattern is unique across all features) are
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
    >>> cfg = HarmonizeConfig(algorithm="limma", sort_strategy="sparsity", block_size=2)
    >>> result = harmonize(df, desc_df, config=cfg)
    """
    # --- Apply config (overrides individual kwargs when provided) ----------
    if config is not None:
        algorithm = config.algorithm  # type: ignore[assignment]
        combat_mode = config.combat_mode
        needed_values = config.needed_values
        sort = config.sort_strategy
        block = config.block_size
        unique_removal = config.unique_removal

    # --- Determine needed_values (before validation) ------------------------
    if needed_values is None:
        if algorithm == "limma" or combat_mode in (1, 3):
            needed_values = 2
        else:
            needed_values = 1

    _start_time = time.monotonic()

    # --- Validate basic arguments (before data load) ----------------------
    validate_harmonize_args(
        algorithm,
        combat_mode,
        needed_values,
        sort_strategy=sort,
        unique_removal=unique_removal,
    )

    # --- Read inputs -------------------------------------------------------
    if isinstance(data, (str, Path)):
        logger.debug("Reading data from %s", data)
        data = read_main_data(str(data))
    assert isinstance(data, pd.DataFrame)  # narrow: always DataFrame after load

    if isinstance(description, (str, Path)):
        logger.debug("Reading description from %s", description)
        description = read_description(str(description))
    assert isinstance(description, pd.DataFrame)  # narrow: always DataFrame after load

    # --- Validate inputs ---------------------------------------------------
    validate_data_matrix(data)
    validate_description(description, data)

    n_features, n_samples = data.shape
    logger.info(
        "Input: %d features x %d samples",
        n_features,
        n_samples,
    )

    # --- Extract batch labels aligned to data column order -----------------
    # Look up ID and batch columns by name (robust to column reordering),
    # falling back to positional indexing for backwards compatibility.
    if "ID" in description.columns and "batch" in description.columns:
        id_col = description["ID"]
        batch_col = description["batch"]
    else:
        id_col = description.iloc[:, 0]
        batch_col = description.iloc[:, 2]
    sample_to_batch = dict(zip(id_col.astype(str), batch_col.astype(int), strict=True))
    batch_list = np.array([sample_to_batch[col] for col in data.columns], dtype=np.int64)

    # --- Validate block_size now that we know n_batches --------------------
    n_batches = len(np.unique(batch_list))
    validate_harmonize_args(
        algorithm,
        combat_mode,
        needed_values,
        block_size=block,
        n_batches=n_batches,
    )

    logger.info(
        "Algorithm: %s%s | batches: %d | needed_values: %d",
        algorithm,
        f" mode {combat_mode}" if algorithm == "ComBat" else "",
        n_batches,
        needed_values,
    )

    # --- Sort batches ------------------------------------------------------
    col_order: npt.NDArray[np.intp] | None = None
    if sort is not None:
        logger.info("Sorting %d batches by '%s'", n_batches, sort)
        data, batch_list, col_order = sort_batches(  # type: ignore[assignment]
            data, batch_list, strategy=sort, needed_values=needed_values
        )

    # --- Build block list --------------------------------------------------
    if block is not None:
        logger.info("Blocking: %d batches → blocks of size %d", n_batches, block)
        block_list = build_block_list(batch_list, block_size=block)
    else:
        block_list = batch_list

    # --- Spot missing values -----------------------------------------------
    affiliation_list = build_affiliation_list(data, batch_list, block_list, needed_values)
    n_empty = sum(1 for a in affiliation_list if len(a) == 0)
    logger.debug(
        "Spotting: %d features with data, %d features dropped (insufficient observations)",
        n_features - n_empty,
        n_empty,
    )

    # --- Unique removal ----------------------------------------------------
    if unique_removal:
        before_ur = sum(1 for a in affiliation_list if len(a) == 0)
        affiliation_list = remove_unique_combinations(affiliation_list)
        after_ur = sum(1 for a in affiliation_list if len(a) == 0)
        rescued = before_ur - after_ur
        if rescued > 0:
            logger.info(
                "Unique removal rescued %d feature(s) (%.1f%%)",
                rescued,
                100.0 * rescued / n_features,
            )
        else:
            logger.debug("Unique removal: no singleton patterns found")

    # --- Split, adjust, rebuild --------------------------------------------
    n_groups = len({a for a in affiliation_list if len(a) > 0})
    logger.info("Adjusting %d unique affiliation group(s)…", n_groups)
    sub_dfs = splitting(
        affiliation_list,
        data,
        batch_list,
        block_list,
        algorithm=algorithm,
        combat_mode=combat_mode,
    )
    result = pd.concat(sub_dfs, axis=0) if sub_dfs else pd.DataFrame()

    # --- Re-sort columns to original order ---------------------------------
    if col_order is not None and result.shape[1] > 0:
        result = result.iloc[:, np.argsort(col_order)]

    # --- Warn if all features were dropped --------------------------------
    if n_empty == n_features:
        logger.warning(
            "All %d features dropped: no feature meets needed_values=%d "
            "in any batch. Output is all-NaN.",
            n_features,
            needed_values,
        )

    # --- Write output ------------------------------------------------------
    if output_file is not None:
        logger.info("Writing output to %s", output_file)
        write_output(result, str(output_file))

    _elapsed = time.monotonic() - _start_time
    logger.info("Done (%.2fs).", _elapsed)
    return result
