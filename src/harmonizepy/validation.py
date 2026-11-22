"""Checks on the inputs of a run and of the engines.

The data matrix, the batch description and its column rule, batch labels,
the ComBat mode, the reference batch, and the arguments of ``harmonize()``
and ``HarmonizeConfig`` are checked here, so that each rule has one place and
one message. ``build_block_list`` and ``sort_batches`` repeat their argument
checks because they can be called on their own. ``check_input_scale`` warns
when the data do not look log-transformed.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from .types import FloatArray

# Every integer up to 2**53 is exact in float64; a float label at or beyond it
# may already have been rounded to a neighbouring integer.
_MAX_FLOAT_LABEL = 2.0**53
_VALID_SORT_STRATEGIES: frozenset[str] = frozenset({"sparsity", "jaccard", "seriation"})
_RAW_SCALE_P95 = 100.0
_SCALE_SAMPLE = 100_000


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


# ComBat squares residuals and sums them per feature; values at or beyond this
# magnitude overflow float64 in those sums and turn the output non-finite.
COMBAT_MAX_ABS_VALUE = 1e150


def reject_combat_overflow(
    values: npt.NDArray[np.floating[Any]], names: Sequence[object] | pd.Index
) -> None:
    """Raise ``ValueError`` if a value is too large for ComBat, naming the first features.

    ComBat's variance sums overflow float64 for magnitudes of about 1e154, which
    makes its output non-finite; values at or beyond ``COMBAT_MAX_ABS_VALUE``
    are rejected. Log-scale data are many orders of magnitude below it.
    """
    with np.errstate(invalid="ignore"):
        large_rows = np.flatnonzero((np.abs(values) >= COMBAT_MAX_ABS_VALUE).any(axis=1))
    if large_rows.size:
        shown = [str(names[i]) for i in large_rows[:5]]
        raise ValueError(
            f"data contains values of magnitude 1e150 or more in {large_rows.size} "
            f"feature(s) (first: {shown}); ComBat's variance sums overflow at such values. "
            "Log-transform the data, or use limma."
        )


def validate_data_matrix(df: pd.DataFrame) -> None:
    """Check that *df* is a features x samples matrix the engines can use.

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
    >>> validate_data_matrix(df)
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
    >>> validate_description(desc, data)
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


def validate_engine_input(data: FloatArray, batch: npt.NDArray[np.integer[Any]]) -> None:
    """Check the matrix and batch labels passed to ``combat()`` or ``remove_batch_effect()``.

    Missing cells are allowed: both engines fit each feature on its observed
    cells and keep missing cells missing.

    Parameters
    ----------
    data : ndarray, shape (n_features, n_samples)
        Features x samples matrix.
    batch : ndarray, shape (n_samples,)
        Integer batch labels, as ``batch_codes`` returns them.

    Raises
    ------
    ValueError
        If *data* is not 2-D, *batch* does not have one label per sample, or
        *data* holds infinite values.

    Examples
    --------
    >>> import numpy as np
    >>> from harmonizepy.validation import validate_engine_input
    >>> validate_engine_input(np.ones((5, 4)), np.array([0, 0, 1, 1]))
    """
    if data.ndim != 2:
        raise ValueError(f"data must be a 2-D array (features x samples), got {data.ndim}-D.")
    n_samples = data.shape[1]
    if batch.shape[0] != n_samples:
        raise ValueError(
            f"batch length ({batch.shape[0]}) does not match the number of "
            f"sample columns in data ({n_samples})."
        )
    _reject_infinite(data, range(data.shape[0]))


def validate_combat_mode(mode: object) -> None:
    """Check that *mode* is one of the ComBat modes, an integer from 1 to 4.

    Modes 1 and 2 are parametric, 3 and 4 non-parametric; 1 and 3 adjust
    location and scale, 2 and 4 location only. NumPy integers are accepted;
    booleans and floats are not, even when they compare equal to a mode.

    Raises
    ------
    ValueError
        If *mode* is not an integer from 1 to 4.
    """
    is_integer = isinstance(mode, (int, np.integer)) and not isinstance(mode, (bool, np.bool_))
    if not is_integer or mode not in (1, 2, 3, 4):
        raise ValueError(
            f"combat_mode must be 1, 2, 3, or 4, got {mode!r}. "
            "Modes 1/2 are parametric; 3/4 are non-parametric. "
            "Modes 1/3 adjust location+scale; 2/4 adjust location only."
        )


def validate_ref_batch(batch: npt.NDArray[np.integer[Any]], ref_batch: int | None) -> None:
    """Check that a reference batch is one of the labels and has at least two samples.

    The pooled variance comes from the reference batch alone, so a single
    sample leaves nothing to pool.

    Raises
    ------
    ValueError
        If *ref_batch* is not among the labels, or labels a single sample.
    """
    if ref_batch is None:
        return
    labels = sorted(set(batch.tolist()))
    if ref_batch not in labels:
        raise ValueError(f"ref_batch={ref_batch!r} not found in batch labels {labels}")
    if int(np.sum(batch == ref_batch)) < 2:
        raise ValueError(
            f"ref_batch={ref_batch!r} has a single sample; a reference batch needs at least two."
        )


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


def validate_core_args(
    algorithm: str,
    combat_mode: int,
    needed_values: int | None,
    sort_strategy: str | None = None,
    block_size: int | None = None,
    unique_removal: bool = True,
) -> None:
    """Check the arguments shared by ``HarmonizeConfig`` and ``harmonize()``.

    Raises
    ------
    ValueError
        On an unknown algorithm, ComBat mode, or sort strategy, a
        ``needed_values`` that is not a whole number of at least 1 (booleans
        and floats are rejected), or a block size that is not an integer of at
        least 2.
    TypeError
        If *unique_removal* is not a bool.
    """
    if algorithm not in ("ComBat", "limma"):
        raise ValueError(f"algorithm must be 'ComBat' or 'limma', got {algorithm!r}.")
    validate_combat_mode(combat_mode)
    if needed_values is not None:
        is_integer = isinstance(needed_values, (int, np.integer)) and not isinstance(
            needed_values, (bool, np.bool_)
        )
        if not is_integer or needed_values < 1:
            raise ValueError(
                f"needed_values must be a whole number of at least 1, or None, got "
                f"{needed_values!r}. Use needed_values=None to auto-select based on algorithm."
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
        Whether unique removal runs; must be a ``bool``.
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
    >>> validate_harmonize_args("ComBat", 2, 2)
    >>> validate_harmonize_args("ComBat", 1, 2, sort_strategy="sparsity",
    ...                         block_size=2, n_batches=4)
    """
    validate_core_args(
        algorithm, combat_mode, needed_values, sort_strategy, block_size, unique_removal
    )
    if n_batches is not None and block_size is not None and block_size >= n_batches:
        raise ValueError(
            f"block ({block_size}) must be less than the number of unique batches "
            f"({n_batches}). Each block needs at least two batches to be meaningful."
        )


class InputScaleWarning(UserWarning):
    """The data matrix looks like it is not on the log scale ComBat and limma expect.

    Silence it with ``warnings.filterwarnings("ignore", category=InputScaleWarning)``.
    """


def check_input_scale(values: FloatArray) -> None:
    """Warn when the observed values look untransformed.

    ComBat and limma model batch effects as additive on the scale they receive,
    so they expect log-transformed values, usually log2. This is a heuristic on
    the observed cells only:

    - every value is non-negative and the 95th percentile is above
      ``_RAW_SCALE_P95`` (100), as for raw intensities or counts. Log2
      intensities and log2 counts stay below about 35;
    - every value lies in [0, 1] with more than two distinct values, as for
      proportions such as methylation beta values. Binary and constant data are
      left alone.

    The percentile is taken on whole rows at a fixed step, about
    ``_SCALE_SAMPLE`` values in total, so that every column is represented and
    the check stays cheap on large matrices. Percentages from 0 to 100 and
    data with a single negative value are not detected.

    Parameters
    ----------
    values : ndarray, shape (n_features, n_samples)
        The data matrix. Missing cells are ignored.

    Warns
    -----
    InputScaleWarning
        Once, naming the pattern found and the transform to apply.
    """
    if values.size == 0:
        return
    flat = values.ravel(order="K")
    low = float(np.fmin.reduce(flat))
    if np.isnan(low) or low < 0.0:
        return
    high = float(np.fmax.reduce(flat))
    n_rows, n_cols = values.shape
    step = max(1, n_rows // max(1, _SCALE_SAMPLE // max(n_cols, 1)))
    sample = values[::step].ravel()
    sample = sample[~np.isnan(sample)]
    if sample.size and float(np.percentile(sample, 95)) > _RAW_SCALE_P95:
        warnings.warn(
            "the data look untransformed: every value is non-negative and the 95th "
            f"percentile is above {_RAW_SCALE_P95:g}. ComBat and limma expect "
            "log-transformed values: apply log2(x + 1) when the data contain zeros, "
            "log2(x) otherwise. This check is a heuristic.",
            InputScaleWarning,
            stacklevel=4,
        )
    elif high <= 1.0 and np.unique(sample).size > 2:
        warnings.warn(
            "the data look like proportions: every value lies between 0 and 1. ComBat "
            "and limma assume values that can vary freely; for methylation beta "
            "values, run the correction on M-values, log2(beta / (1 - beta)), and "
            "convert back to beta. This check is a heuristic.",
            InputScaleWarning,
            stacklevel=4,
        )
