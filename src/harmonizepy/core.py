"""Core pipeline entry point for HarmonizePy.

Orchestrates the full batch-correction workflow:

    read → [sort] → [block] → spot missing → [unique removal]
        → split → adjust (ComBat / limma) → concat → [re-sort] → write

This mirrors the R ``HarmonizR::harmonizR()`` function.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from .affiliation import build_affiliation_list, remove_unique_combinations
from .blocking import build_block_list
from .io import clean_features, read_description, read_main_data, write_output
from .sorting import sort_batches
from .splitting import splitting
from .types import HarmonizeConfig
from .validation import (
    sample_batch_map,
    validate_data_matrix,
    validate_description,
    validate_harmonize_args,
)

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
        sufficient data in a batch.  A feature repeated with the same
        name and identical values appears once; the removal is logged.

    Raises
    ------
    ValueError
        On invalid arguments, mismatched inputs, or a feature name that
        appears more than once with different values.

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
        algorithm = config.algorithm
        combat_mode = config.combat_mode
        needed_values = config.needed_values
        sort = config.sort_strategy
        block = config.block_size
        unique_removal = config.unique_removal

    start_time = time.monotonic()
    inputs = prepare_inputs(
        data,
        description,
        algorithm=algorithm,
        combat_mode=combat_mode,
        needed_values=needed_values,
        sort=sort,
        block=block,
        unique_removal=unique_removal,
    )
    plan = plan_affiliations(
        inputs.data,
        inputs.batch_list,
        sort=sort,
        block=block,
        needed_values=inputs.needed_values,
        unique_removal=unique_removal,
    )

    logger.info("Adjusting %d unique affiliation group(s)", plan.n_groups)
    result = splitting(
        plan.affiliation_list,
        plan.data,
        plan.batch_list,
        plan.block_list,
        algorithm=algorithm,
        combat_mode=combat_mode,
        output_col_order=plan.col_order,
        output_columns=inputs.data.columns if plan.col_order is not None else None,
        data_np=plan.data_np,
    )

    n_features = inputs.data.shape[0]
    if plan.n_empty == n_features:
        logger.warning(
            "All %d features dropped: no feature meets needed_values=%d "
            "in any batch. Output is all-NaN.",
            n_features,
            inputs.needed_values,
        )

    if output_file is not None:
        logger.info("Writing output to %s", output_file)
        write_output(result, str(output_file))

    logger.info("Done (%.2fs).", time.monotonic() - start_time)
    return result


def resolve_needed_values(algorithm: str, combat_mode: int, needed_values: int | None) -> int:
    """Return the minimum observations per batch, choosing the default when it is ``None``.

    The default is 2 for limma and the location-and-scale ComBat modes (1 and 3),
    which estimate a variance per batch, and 1 for the location-only modes (2
    and 4).
    """
    if needed_values is not None:
        return needed_values
    return 2 if algorithm == "limma" or combat_mode in (1, 3) else 1


@dataclass(frozen=True)
class PreparedInputs:
    """Validated inputs of a run: the cleaned matrix and its batch labels."""

    data: pd.DataFrame
    batch_list: npt.NDArray[np.int64]
    needed_values: int
    n_batches: int


def prepare_inputs(
    data: pd.DataFrame | str | Path,
    description: pd.DataFrame | str | Path,
    *,
    algorithm: str,
    combat_mode: int,
    needed_values: int | None,
    sort: str | None,
    block: int | None,
    unique_removal: bool,
) -> PreparedInputs:
    """Read, clean, and validate the inputs of a run, and map samples to batches.

    Shared by ``harmonize()`` and the command line dry run, so that both apply
    the same checks. Paths are read with ``read_main_data`` and
    ``read_description``; repeated features are removed by ``clean_features``.

    Raises
    ------
    ValueError, TypeError
        From the argument, matrix, and description checks in ``validation``.
    OSError, ImportError
        From reading a path: a missing or unreadable file, or Parquet without
        ``pyarrow``.
    """
    resolved = resolve_needed_values(algorithm, combat_mode, needed_values)
    validate_harmonize_args(
        algorithm, combat_mode, resolved, sort_strategy=sort, unique_removal=unique_removal
    )
    if isinstance(data, (str, Path)):
        logger.debug("Reading data from %s", data)
        data = read_main_data(str(data))
    if isinstance(description, (str, Path)):
        logger.debug("Reading description from %s", description)
        description = read_description(str(description))

    data = clean_features(data)
    validate_data_matrix(data)
    validate_description(description, data)
    logger.info("Input: %d features x %d samples", data.shape[0], data.shape[1])

    sample_to_batch = sample_batch_map(description)
    batch_list = np.array([sample_to_batch[col] for col in data.columns], dtype=np.int64)
    n_batches = len(np.unique(batch_list))
    validate_harmonize_args(algorithm, combat_mode, resolved, block_size=block, n_batches=n_batches)
    logger.info(
        "Algorithm: %s%s | batches: %d | needed_values: %d",
        algorithm,
        f" mode {combat_mode}" if algorithm == "ComBat" else "",
        n_batches,
        resolved,
    )
    return PreparedInputs(data, batch_list, resolved, n_batches)


@dataclass(frozen=True)
class AffiliationPlan:
    """The groups a run corrects, after sorting, blocking, and unique removal."""

    data: pd.DataFrame
    data_np: npt.NDArray[np.float64]
    batch_list: npt.NDArray[np.integer]
    block_list: npt.NDArray[np.integer]
    col_order: npt.NDArray[np.intp] | None
    affiliation_list: list[tuple[int, ...]]
    n_empty: int

    @property
    def n_groups(self) -> int:
        """Number of distinct non-empty affiliations, one sub-matrix each."""
        return len({a for a in self.affiliation_list if a})


def plan_affiliations(
    data: pd.DataFrame,
    batch_list: npt.NDArray[np.int64],
    *,
    sort: str | None,
    block: int | None,
    needed_values: int,
    unique_removal: bool,
) -> AffiliationPlan:
    """Sort batches, build blocks, find each feature's affiliation, and apply unique removal.

    Shared by ``harmonize()`` and the command line dry run, so that the dry run
    reports the groups the correction uses.
    """
    n_features = data.shape[0]
    n_batches = len(np.unique(batch_list))
    col_order: npt.NDArray[np.intp] | None = None
    sorted_batches: npt.NDArray[np.integer] = batch_list
    if sort is not None:
        logger.info("Sorting %d batches by '%s'", n_batches, sort)
        data, sorted_batches, col_order = sort_batches(
            data, batch_list, strategy=sort, needed_values=needed_values
        )

    if block is not None:
        logger.info("Blocking: %d batches in blocks of %d", n_batches, block)
        block_list = build_block_list(sorted_batches, block_size=block)
    else:
        block_list = sorted_batches

    data_np = data.to_numpy(dtype=np.float64)
    affiliation_list = build_affiliation_list(
        data, sorted_batches, block_list, needed_values, data_np=data_np
    )
    n_empty = sum(1 for a in affiliation_list if not a)
    logger.debug(
        "Spotting: %d features with data, %d features without a usable batch",
        n_features - n_empty,
        n_empty,
    )

    if unique_removal:
        cropped = remove_unique_combinations(affiliation_list)
        n_cropped = sum(1 for old, new in zip(affiliation_list, cropped, strict=True) if old != new)
        affiliation_list = cropped
        if n_cropped:
            logger.info(
                "Unique removal moved %d feature(s) into a shared pattern (%.1f%%)",
                n_cropped,
                100.0 * n_cropped / n_features,
            )
        else:
            logger.debug("Unique removal: no feature had a pattern of its own")

    return AffiliationPlan(
        data, data_np, sorted_batches, block_list, col_order, affiliation_list, n_empty
    )
