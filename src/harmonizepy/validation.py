"""Centralised input validation for HarmonizePy.

Every public-facing validation check lives here so error messages are
consistent and the logic is not duplicated across modules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

_Array = npt.NDArray[np.floating[Any]]


def validate_data_matrix(df: pd.DataFrame) -> None:
    """Check that *df* is a valid features-x-samples matrix.

    Parameters
    ----------
    df : DataFrame
        Features x samples matrix to validate.

    Raises
    ------
    ValueError
        If *df* has duplicate row indices, all-NaN rows, fewer than
        2 columns, or non-numeric dtype.

    Examples
    --------
    >>> import pandas as pd
    >>> from harmonizepy.validation import validate_data_matrix
    >>> df = pd.DataFrame({"s1": [1.0, 2.0], "s2": [3.0, 4.0]})
    >>> validate_data_matrix(df)  # no error
    """
    dups = df.index[df.index.duplicated(keep=False)].unique().tolist()
    if dups:
        raise ValueError(
            f"data contains duplicate feature names (first {len(dups[:5])}: "
            f"{dups[:5]}). Each row must have a unique identifier."
        )

    bad = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(
            f"data contains non-numeric columns: {bad[:5]}. "
            f"All sample columns must be numeric (float or int)."
        )

    if df.shape[1] < 2:
        raise ValueError(
            f"data must have at least 2 sample columns (got {df.shape[1]}). "
            f"Batch correction requires samples from multiple batches."
        )


def validate_description(desc: pd.DataFrame, data: pd.DataFrame) -> None:
    """Check that a batch description matches a data matrix.

    Parameters
    ----------
    desc : DataFrame
        Batch description with at least 3 columns (ID, sample, batch).
    data : DataFrame
        Data matrix whose column names must match the IDs in *desc*.

    Raises
    ------
    ValueError
        If sample IDs do not match data columns or the description has
        fewer than 3 columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from harmonizepy.validation import validate_description
    >>> data = pd.DataFrame({"s1": [1.0], "s2": [2.0]})
    >>> desc = pd.DataFrame({"ID": ["s1", "s2"], "sample": [1, 2], "batch": [1, 2]})
    >>> validate_description(desc, data)  # no error
    """
    if desc.shape[1] < 3:
        raise ValueError(
            f"description must have at least 3 columns (ID, sample, batch), "
            f"got {desc.shape[1]}. Check the description file format."
        )

    if "ID" in desc.columns:
        id_series = desc["ID"].astype(str)
    else:
        id_series = desc.iloc[:, 0].astype(str)
    desc_ids = set(id_series)
    data_ids = set(data.columns.astype(str))

    if desc_ids != data_ids:
        extra_desc = desc_ids - data_ids
        extra_data = data_ids - desc_ids
        parts = []
        if extra_desc:
            parts.append(f"in description but not data: {sorted(extra_desc)[:5]}")
        if extra_data:
            parts.append(f"in data but not description: {sorted(extra_data)[:5]}")
        raise ValueError(
            f"Sample IDs in description do not match data columns - {'; '.join(parts)}. "
            f"The first column (or a column named 'ID') of description must list the exact column names of data."
        )


def validate_combat_input(data: _Array, batch: npt.NDArray[np.integer[Any]]) -> None:
    """Check ndarray inputs for the ComBat engine.

    Parameters
    ----------
    data : ndarray, shape (n_features, n_samples)
        Features x samples matrix. Per-cell NaN is allowed and handled
        inside ``combat()`` via per-feature computations that omit only
        the NaN observations for the feature being fitted. NaN positions
        are preserved in the output.
    batch : ndarray, shape (n_samples,)
        Integer batch labels; length must equal the number of samples.

    Raises
    ------
    ValueError
        On wrong dimensionality or batch length mismatch.

    Examples
    --------
    >>> import numpy as np
    >>> from harmonizepy.validation import validate_combat_input
    >>> data = np.ones((5, 4))
    >>> validate_combat_input(data, np.array([0, 0, 1, 1]))  # no error
    """
    if data.ndim != 2:
        raise ValueError(f"data must be a 2-D array (features x samples), got {data.ndim}-D.")

    _, n_samples = data.shape

    if batch.shape[0] != n_samples:
        raise ValueError(
            f"batch length ({batch.shape[0]}) does not match the number of "
            f"sample columns in data ({n_samples})."
        )


def validate_limma_input(data: _Array, batch: _Array) -> None:
    """Check ndarray inputs for the limma engine.

    Parameters
    ----------
    data : ndarray, shape (n_features, n_samples)
        Features x samples matrix. Per-cell NaN is allowed and handled
        inside ``remove_batch_effect()`` via per-feature fits that omit
        only the NaN observations for the feature being fitted. NaN
        positions are preserved in the output.
    batch : ndarray, shape (n_samples,)
        Integer batch labels; length must equal the number of samples.

    Raises
    ------
    ValueError
        On wrong dimensionality or batch length mismatch.

    Examples
    --------
    >>> import numpy as np
    >>> from harmonizepy.validation import validate_limma_input
    >>> data = np.ones((5, 4))
    >>> validate_limma_input(data, np.array([0, 0, 1, 1]))  # no error
    """
    if data.ndim != 2:
        raise ValueError(f"data must be a 2-D array (features x samples), got {data.ndim}-D.")

    _, n_samples = data.shape

    if batch.shape[0] != n_samples:
        raise ValueError(
            f"batch length ({batch.shape[0]}) does not match the number of "
            f"sample columns in data ({n_samples})."
        )


_VALID_SORT_STRATEGIES: frozenset[str] = frozenset({"sparsity", "jaccard", "seriation"})


def _validate_core_args(
    algorithm: str,
    combat_mode: int,
    needed_values: int | None,
    sort_strategy: str | None = None,
    block_size: int | None = None,
    unique_removal: bool = True,
) -> None:
    """Validate core parameter constraints shared by HarmonizeConfig and harmonize()."""
    if algorithm not in ("ComBat", "limma"):
        raise ValueError(f"algorithm must be 'ComBat' or 'limma', got {algorithm!r}.")
    if combat_mode not in (1, 2, 3, 4):
        raise ValueError(
            f"combat_mode must be 1, 2, 3, or 4, got {combat_mode}. "
            f"Modes 1/2 are parametric; 3/4 are non-parametric. "
            f"Modes 1/3 adjust location+scale; 2/4 adjust location only."
        )
    if needed_values is not None and needed_values < 1:
        raise ValueError(
            f"needed_values must be >= 1 or None, got {needed_values}. "
            f"Use needed_values=None to auto-select based on algorithm."
        )
    if sort_strategy is not None and sort_strategy not in _VALID_SORT_STRATEGIES:
        raise ValueError(
            f"sort must be one of {sorted(_VALID_SORT_STRATEGIES)!r} or None, "
            f"got {sort_strategy!r}."
        )
    if block_size is not None and (not isinstance(block_size, int) or block_size < 2):
        raise ValueError(
            f"block must be an integer >= 2 or None, got {block_size!r}. "
            f"Use block=None to disable blocking."
        )
    if not isinstance(unique_removal, bool):
        raise TypeError(
            f"unique_removal must be True or False, got {type(unique_removal).__name__!r}."
        )


def validate_harmonize_args(
    algorithm: str,
    combat_mode: int,
    needed_values: int,
    sort_strategy: str | None = None,
    block_size: int | None = None,
    unique_removal: bool = True,
    n_batches: int | None = None,
) -> None:
    """Validate top-level ``harmonize()`` keyword arguments.

    Parameters
    ----------
    algorithm : str
        Adjustment algorithm; must be ``"ComBat"`` or ``"limma"``.
    combat_mode : int
        ComBat variant; must be 1, 2, 3, or 4.
    needed_values : int
        Minimum non-missing observations per batch; must be >= 1.
    sort_strategy : str or None
        Batch sorting strategy; must be ``"sparsity"``, ``"jaccard"``,
        ``"seriation"``, or ``None``.
    block_size : int or None
        Block size; must be ``None`` or an integer >= 2.  If *n_batches*
        is provided it must also be strictly less than *n_batches*.
    unique_removal : bool
        Singleton-rescue toggle; must be a ``bool``.
    n_batches : int or None
        Number of unique batches in the data.  When provided, validates
        ``block_size < n_batches``.

    Raises
    ------
    ValueError
        On invalid algorithm, combat_mode, needed_values, sort_strategy,
        or block_size.
    TypeError
        If *unique_removal* is not a bool.

    Examples
    --------
    >>> from harmonizepy.validation import validate_harmonize_args
    >>> validate_harmonize_args("ComBat", 2, 2)  # no error
    >>> validate_harmonize_args("ComBat", 1, 2, sort_strategy="sparsity",
    ...                         block_size=2, n_batches=4)  # no error
    """
    _validate_core_args(algorithm, combat_mode, needed_values, sort_strategy, block_size, unique_removal)
    if n_batches is not None and block_size is not None and block_size >= n_batches:
        raise ValueError(
            f"block ({block_size}) must be less than the number of unique batches "
            f"({n_batches}). Each block needs at least two batches to be meaningful."
        )
