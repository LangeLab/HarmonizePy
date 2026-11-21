"""Centralised input validation for HarmonizePy.

Every public-facing validation check lives here so error messages are
consistent and the logic is not duplicated across modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

_Array = npt.NDArray[np.floating[Any]]


def _reject_infinite(
    values: npt.NDArray[np.floating[Any]], names: Sequence[object] | pd.Index
) -> None:
    """Raise ``ValueError`` if any value is infinite, naming the first features."""
    infinite_rows = np.flatnonzero(np.isinf(values).any(axis=1))
    if infinite_rows.size:
        shown = [str(names[i]) for i in infinite_rows[:5]]
        raise ValueError(
            f"data contains infinite values in {infinite_rows.size} feature(s) "
            f"(first: {shown}). Replace them with NaN; log2 of zero, for example, "
            "gives -inf."
        )


def validate_data_matrix(df: pd.DataFrame) -> None:
    """Check that *df* is a valid features-x-samples matrix.

    Parameters
    ----------
    df : DataFrame
        Features x samples matrix to validate.

    Raises
    ------
    ValueError
        If *df* has duplicate feature or sample names, fewer than 2 columns,
        a non-numeric column, or infinite values.

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

    repeated_samples = df.columns[df.columns.duplicated()].unique().tolist()
    if repeated_samples:
        raise ValueError(
            f"data contains duplicate sample names: {repeated_samples[:5]}. "
            "Each column must have a unique name."
        )

    bad = df.dtypes[~df.dtypes.map(pd.api.types.is_numeric_dtype)].index.tolist()
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

    _reject_infinite(df.to_numpy(dtype=np.float64, na_value=np.nan), df.index)


def validate_description(desc: pd.DataFrame, data: pd.DataFrame) -> None:
    """Check that a batch description matches a data matrix.

    Parameters
    ----------
    desc : DataFrame
        Batch description with at least 3 columns (ID, sample, batch). The ID
        and batch columns are found by name when present, otherwise by
        position (first and third).
    data : DataFrame
        Data matrix whose column names must match the IDs in *desc*.

    Raises
    ------
    ValueError
        If the description has fewer than 3 columns, its ID or batch column
        cannot be found, it lists a sample ID more than once, its IDs do not
        match the data columns, or a batch label is not a whole number.

    Examples
    --------
    >>> import pandas as pd
    >>> from harmonizepy.validation import validate_description
    >>> data = pd.DataFrame({"s1": [1.0], "s2": [2.0]})
    >>> desc = pd.DataFrame({"ID": ["s1", "s2"], "sample": [1, 2], "batch": [1, 2]})
    >>> validate_description(desc, data)  # no error
    """
    ids, batches = _description_columns(desc)
    id_series = ids.astype(str)
    repeated_ids = id_series[id_series.duplicated()]
    if not repeated_ids.empty:
        raise ValueError(
            f"sample IDs listed more than once in the description: "
            f"{sorted(set(repeated_ids))[:5]}. Each sample must have one batch."
        )
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
            f"The column named 'ID' (or else the first column) of description must list the "
            f"exact column names of data."
        )

    _integer_labels(batches)


def _description_columns(desc: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the sample ID and batch columns of a batch description.

    Each is found by name (``ID``, ``batch``) when present, otherwise by
    position: the first column for IDs and the third for batches.

    Raises
    ------
    ValueError
        If the description has fewer than 3 columns, or the ID and batch rules
        point at the same column.
    """
    if desc.shape[1] < 3:
        raise ValueError(
            f"description must have at least 3 columns (ID, sample, batch), "
            f"got {desc.shape[1]}. Check the description file format."
        )
    id_name = "ID" if "ID" in desc.columns else desc.columns[0]
    batch_name = "batch" if "batch" in desc.columns else desc.columns[2]
    if batch_name == id_name:
        if id_name == "batch":
            raise ValueError(
                "the ID column cannot be found: the first column is the batch column. "
                "Name the ID column 'ID'."
            )
        raise ValueError(
            "the batch column cannot be found: the third column is the ID column. "
            "Name the batch column 'batch'."
        )
    return desc[id_name], desc[batch_name]


# Every integer up to 2**53 is exact in float64; a float label at or beyond it
# may already have been rounded to a neighbouring integer.
_MAX_FLOAT_LABEL = 2.0**53


def _not_whole(values: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Mark labels that are missing, infinite, fractional, or too large to be exact.

    Float labels must lie below 2**53 in magnitude, the range in which float64
    holds every integer exactly.
    """
    bad: npt.NDArray[np.bool_] = (
        ~np.isfinite(values) | (values != np.round(values)) | (np.abs(values) >= _MAX_FLOAT_LABEL)
    )
    return bad


def _shown_labels(labels: Sequence[object]) -> str:
    """Return up to five distinct labels, quoted, for an error message."""
    distinct = list(dict.fromkeys(repr(label) for label in labels))
    return ", ".join(distinct[:5]) + (", ..." if len(distinct) > 5 else "")


def _integer_labels(batches: pd.Series) -> npt.NDArray[np.int64]:
    """Return a description's batch labels as integers.

    Integer columns are used as they are. Other columns are read as numbers and
    must hold whole numbers below 2**53 in magnitude.

    Raises
    ------
    ValueError
        If a label is missing, not numeric, not a whole number, or out of range,
        naming the column and the offending values.
    """
    numeric = pd.to_numeric(batches, errors="coerce")
    if pd.api.types.is_signed_integer_dtype(numeric.dtype) and not numeric.isna().any():
        return numeric.to_numpy(dtype=np.int64)
    values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    bad = _not_whole(values)
    if bad.any():
        raise ValueError(
            f"batch labels in column {batches.name!r} must be whole numbers below 2**53 "
            f"in magnitude; found {_shown_labels(batches[bad].tolist())}."
        )
    return values.astype(np.int64)


def batch_codes(batch: npt.ArrayLike) -> npt.NDArray[np.intp]:
    """Return engine batch labels as integers, rejecting labels that are not whole numbers.

    Parameters
    ----------
    batch : array_like, shape (n_samples,)
        One label per sample: integers of any width, booleans, or floats and
        numeric strings with whole values below 2**53 in magnitude, such as 1.0.

    Returns
    -------
    ndarray of intp, shape (n_samples,)
        The labels as integers, in the input order.

    Raises
    ------
    ValueError
        If a label is text that is not a number, complex, missing, infinite,
        fractional, or a float at least 2**53 in magnitude, naming the offending
        labels.
    """
    labels = np.asarray(batch).ravel()
    if labels.dtype.kind in "iub":
        return labels.astype(np.intp)
    if labels.dtype.kind == "c":
        values = np.full(labels.shape, np.nan)
    else:
        try:
            values = labels.astype(np.float64)
        except (TypeError, ValueError):
            values = np.full(labels.shape, np.nan)
    bad = _not_whole(values)
    if bad.any():
        raise ValueError(
            "batch labels must be whole numbers below 2**53 in magnitude; "
            f"found {_shown_labels(labels[bad].tolist())}."
        )
    return values.astype(np.intp)


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
        On wrong dimensionality, batch length mismatch, or infinite values.

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

    _reject_infinite(data, range(data.shape[0]))


def validate_limma_input(data: _Array, batch: npt.NDArray[np.integer[Any]]) -> None:
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
        On wrong dimensionality, batch length mismatch, or infinite values.

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

    _reject_infinite(data, range(data.shape[0]))


def sample_batch_map(desc: pd.DataFrame) -> dict[str, int]:
    """Map each sample ID to its integer batch label.

    The ID and batch columns are found as in ``validate_description``: by the
    names ``ID`` and ``batch`` when present, otherwise the first column holds
    the IDs and the third holds the batch labels.

    Parameters
    ----------
    desc : DataFrame
        Batch description with at least three columns.

    Returns
    -------
    dict[str, int]
        Sample ID to batch label.

    Raises
    ------
    ValueError
        If the description has fewer than 3 columns, its ID or batch column
        cannot be found, it lists a sample ID more than once, or a batch label
        is not a whole number.
    """
    ids, batches = _description_columns(desc)
    id_series = ids.astype(str)
    if id_series.duplicated().any():
        raise ValueError("sample IDs listed more than once in the description.")
    return dict(zip(id_series, _integer_labels(batches).tolist(), strict=True))


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
    _validate_core_args(
        algorithm, combat_mode, needed_values, sort_strategy, block_size, unique_removal
    )
    if n_batches is not None and block_size is not None and block_size >= n_batches:
        raise ValueError(
            f"block ({block_size}) must be less than the number of unique batches "
            f"({n_batches}). Each block needs at least two batches to be meaningful."
        )
