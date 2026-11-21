"""The ``harmonize()`` pipeline and the steps it shares with the command line.

A run reads and validates the inputs, maps samples to batches, optionally
sorts the batches and groups neighbouring batches into blocks, finds each
feature's affiliation (the blocks in which it has enough values), applies
unique removal, corrects each group of features that share an affiliation
with ComBat or limma, and reassembles the matrix in the input column order::

    read -> validate -> [sort] -> [block] -> affiliation -> [unique removal]
         -> correct each group -> reassemble -> [write]

The pipeline follows R ``HarmonizR::harmonizR()`` from HarmonizR 1.10.0. The
differences are listed in ``assets/FEATURE_PARITY.md``.
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
    check_input_scale,
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
    """Correct batch effects in a matrix with missing values.

    Parameters
    ----------
    data : DataFrame, str, or Path
        Features x samples matrix, feature names as the index. Missing cells
        are allowed and stay missing. A path is read with ``read_main_data``:
        ``.csv`` as comma-separated, ``.parquet`` or ``.pq`` as Parquet, and
        anything else as tab-separated.
    description : DataFrame, str, or Path
        Batch description, one row per sample. The columns named ``ID`` and
        ``batch`` hold the sample names and batch labels; without those
        names, the first and third columns are used. Batch labels must be
        whole numbers. A path is read as CSV.
    config : HarmonizeConfig or None
        Settings for the run. When given, it replaces ``algorithm``,
        ``combat_mode``, ``needed_values``, ``sort``, ``block``, and
        ``unique_removal``.
    algorithm : {"ComBat", "limma"}
        Correction engine.
    combat_mode : {1, 2, 3, 4}
        ComBat mode: 1 parametric location and scale, 2 parametric location
        only, 3 non-parametric location and scale, 4 non-parametric location
        only. Ignored for limma.
    needed_values : int or None
        Minimum observed values a feature needs in a batch for that batch to
        count in its affiliation. ``None`` chooses 2 for limma and modes 1 and
        3, and 1 for modes 2 and 4.
    sort : {"sparsity", "jaccard", "seriation"} or None
        Batch order used to form blocks. It changes the output only together
        with ``block``.
    block : int or None
        Number of neighbouring batches per block, at least 2 and fewer than
        the number of batches. ``None`` treats every batch as its own block.
    unique_removal : bool
        When ``True``, a feature whose affiliation no other feature shares is
        moved to the largest shared affiliation within its own, as R
        ``ur = TRUE`` does.
    output_file : str, Path, or None
        If given, the corrected matrix is also written here, in the format of
        the suffix (``.csv``, ``.parquet`` or ``.pq``, otherwise TSV).

    Returns
    -------
    DataFrame
        Corrected matrix with the input's features and columns, in the input
        order, except that a feature repeated with the same name and values
        appears once. A feature with no usable batch is all missing. A feature
        alone in its affiliation, one whose affiliation spans a single batch,
        and, in the parametric ComBat modes, one whose variance is too small
        to estimate keep their raw values; the log counts each kind.

    Raises
    ------
    ValueError
        On an invalid argument, a matrix or description that fails
        validation, or a feature name repeated with different values.
    TypeError
        If ``unique_removal`` is not a bool.
    OSError
        If a path cannot be read or ``output_file`` cannot be written.
    ImportError
        If a Parquet file is read or written without ``pyarrow``.

    Warns
    -----
    InputScaleWarning
        When the values do not look log-transformed; see ``check_input_scale``.

    Examples
    --------
    >>> from harmonizepy import HarmonizeConfig, harmonize
    >>> result = harmonize("data.tsv", "batch.csv")  # doctest: +SKIP
    >>> result = harmonize(df, desc, algorithm="limma")  # doctest: +SKIP
    >>> cfg = HarmonizeConfig(sort_strategy="sparsity", block_size=2)
    >>> result = harmonize(df, desc, config=cfg)  # doctest: +SKIP

    Notes
    -----
    The inputs are not modified. Progress is logged at INFO level on the
    ``harmonizepy`` loggers.
    """
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

    Parameters
    ----------
    data, description
        As in ``harmonize()``.
    algorithm, combat_mode, needed_values, sort, block, unique_removal
        As in ``harmonize()``; ``needed_values`` may be ``None``.

    Returns
    -------
    PreparedInputs
        The cleaned matrix, the batch label of each of its columns, the
        resolved ``needed_values``, and the number of batches.

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
    check_input_scale(data.to_numpy(dtype=np.float64))
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

    Parameters
    ----------
    data : DataFrame
        Cleaned features x samples matrix.
    batch_list : ndarray of int64, shape (n_samples,)
        Batch label of each column of *data*.
    sort, block, needed_values, unique_removal
        As in ``harmonize()``, with ``needed_values`` resolved.

    Returns
    -------
    AffiliationPlan
        The matrix and labels in sorted column order, the block of each
        column, the permutation back to the input order (``None`` without
        sorting), and each feature's affiliation.
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
