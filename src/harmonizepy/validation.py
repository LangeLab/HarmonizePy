"""Centralised input validation for HarmonizePy.

Every public-facing validation check lives here so error messages are
consistent and the logic is not duplicated across modules.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

_Array = npt.NDArray[np.floating]


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
    if df.index.duplicated().any():
        dups = df.index[df.index.duplicated(keep=False)].unique().tolist()
        raise ValueError(f"Duplicate feature names: {dups[:5]}")

    bad = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"Non-numeric columns: {bad[:5]}")

    if df.shape[1] < 2:
        raise ValueError(
            f"Data must have >= 2 sample columns, got {df.shape[1]}"
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
        If sample IDs do not match data columns or the batch column has
        fewer than 2 unique values.

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
            f"Description must have >= 3 columns (ID, sample, batch), "
            f"got {desc.shape[1]}"
        )

    desc_ids = set(desc.iloc[:, 0].astype(str))
    data_ids = set(data.columns.astype(str))

    if desc_ids != data_ids:
        extra_desc = desc_ids - data_ids
        extra_data = data_ids - desc_ids
        parts = []
        if extra_desc:
            parts.append(f"in description but not data: {sorted(extra_desc)[:5]}")
        if extra_data:
            parts.append(f"in data but not description: {sorted(extra_data)[:5]}")
        raise ValueError(f"Sample ID mismatch: {'; '.join(parts)}")




def validate_combat_input(data: _Array, batch: _Array) -> None:
    """Check ndarray inputs for the ComBat engine.

    Parameters
    ----------
    data : ndarray, shape (n_features, n_samples)
        Features x samples matrix.  Must be 2-D and NaN-free.
    batch : ndarray, shape (n_samples,)
        Integer batch labels; length must equal the number of samples.

    Raises
    ------
    ValueError
        On NaN in *data*, wrong dimensionality, < 2 features, or
        batch length mismatch.

    Examples
    --------
    >>> import numpy as np
    >>> from harmonizepy.validation import validate_combat_input
    >>> data = np.ones((5, 4))
    >>> validate_combat_input(data, np.array([0, 0, 1, 1]))  # no error
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got {data.ndim}-D")

    n_features, n_samples = data.shape

    if n_features < 2:
        raise ValueError("ComBat requires >= 2 features (rows)")

    if np.isnan(data).any():
        raise ValueError("data must not contain NaN")

    if batch.shape[0] != n_samples:
        raise ValueError(
            f"batch length ({batch.shape[0]}) != number of samples ({n_samples})"
        )


def validate_limma_input(data: _Array, batch: _Array) -> None:
    """Check ndarray inputs for the limma engine.

    Parameters
    ----------
    data : ndarray, shape (n_features, n_samples)
        Features x samples matrix.  Must be 2-D and NaN-free.
    batch : ndarray, shape (n_samples,)
        Integer batch labels; length must equal the number of samples.

    Raises
    ------
    ValueError
        On NaN in *data*, wrong dimensionality, or batch length mismatch.

    Examples
    --------
    >>> import numpy as np
    >>> from harmonizepy.validation import validate_limma_input
    >>> data = np.ones((5, 4))
    >>> validate_limma_input(data, np.array([0, 0, 1, 1]))  # no error
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D, got {data.ndim}-D")

    if np.isnan(data).any():
        raise ValueError("data must not contain NaN")

    n_features, n_samples = data.shape

    if batch.shape[0] != n_samples:
        raise ValueError(
            f"batch length ({batch.shape[0]}) != number of samples ({n_samples})"
        )


def validate_harmonize_args(
    algorithm: str, combat_mode: int, needed_values: int,
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

    Raises
    ------
    ValueError
        On invalid algorithm, combat_mode, or needed_values.

    Examples
    --------
    >>> from harmonizepy.validation import validate_harmonize_args
    >>> validate_harmonize_args("ComBat", 2, 2)  # no error
    """
    if algorithm not in ("ComBat", "limma"):
        raise ValueError(
            f"algorithm must be 'ComBat' or 'limma', got {algorithm!r}"
        )
    if combat_mode not in (1, 2, 3, 4):
        raise ValueError(f"combat_mode must be 1-4, got {combat_mode}")
    if needed_values < 1:
        raise ValueError(f"needed_values must be >= 1, got {needed_values}")
