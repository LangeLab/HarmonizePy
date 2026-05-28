"""Validity checks for benchmark results.

The validity pass runs before any performance timing and is never skipped.
It uses the warmup run result to verify that the corrected output satisfies
the core contract: shape preserved, NaN positions preserved, no Inf values.

Usage::

    from benchmarks.validity import validate_result, compute_concordance, ValidityResult

    vr = validate_result(data_df, result_df, scenario_id)
    print(vr.shape_preserved, vr.nan_preserved)

    if r_result_path is not None:
        cr = compute_concordance(result_df, str(r_result_path))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ValidityResult:
    """Results of a single validity check on a benchmark run.

    The first five fields are always populated.  The concordance fields
    are populated only when R reference output is available.
    """

    scenario_id: str
    shape_preserved: bool = True
    nan_preserved: bool = True
    no_inf: bool = True
    row_count_match: bool = True
    error: str | None = None

    # Concordance (only when R results are available)
    concordance_max_rel: float | None = None
    concordance_mean_rel: float | None = None
    concordance_p95_rel: float | None = None
    concordance_nan_match: bool | None = None
    concordance_shared_features: int | None = None
    concordance_py_only_features: int | None = None
    concordance_r_only_features: int | None = None
    concordance_shared_nonnan_cells: int | None = None


def validate_result(
    data: pd.DataFrame,
    result: pd.DataFrame,
    scenario_id: str,
) -> ValidityResult:
    """Check that a corrected output satisfies the core contract.

    Checks:
    - Output shape matches input shape exactly.
    - Every input NaN is still NaN in output.
    - No Inf or -Inf in output.
    - Output row count == input row count (handles all-NaN rows from
      empty affiliations).

    Parameters
    ----------
    data : DataFrame
        Original input matrix.
    result : DataFrame
        Corrected output matrix.
    scenario_id : str
        Identifier for the scenario being validated.

    Returns
    -------
    ValidityResult
        One field per check; ``error`` is set if the pipeline raised.
    """
    vr = ValidityResult(scenario_id=scenario_id)

    # Shape
    if data.shape != result.shape:
        vr.shape_preserved = False
        vr.error = (
            f"Shape mismatch: input {data.shape}, output {result.shape}"
        )

    # Row count (separate from shape for explicit contract check)
    if data.shape[0] != result.shape[0]:
        vr.row_count_match = False

    # NaN positions preserved
    if vr.shape_preserved:
        input_nan = np.isnan(data.to_numpy())
        output_nan = np.isnan(result.to_numpy())
        if not (input_nan == output_nan).all():
            # Input NaN present but some shifted to non-NaN:
            # is every input NaN still NaN in output?
            if not (output_nan[input_nan]).all():
                vr.nan_preserved = False
                vr.error = (
                    "Some input NaN values became non-NaN in output "
                    "(imputation detected)"
                )

    # No Inf
    result_arr = result.to_numpy(dtype=np.float64)
    if np.isinf(result_arr).any():
        vr.no_inf = False

    return vr


def compute_concordance(
    py_result: pd.DataFrame | str | Path,
    r_result_path: str | Path,
) -> ValidityResult:
    """Compute concordance metrics between Python and R corrected outputs.

    Parameters
    ----------
    py_result : DataFrame or str or Path
        Python corrected output, either as DataFrame or file path.
    r_result_path : str or Path
        Path to R corrected output TSV file.

    Returns
    -------
    ValidityResult
        Populated concordance fields; other fields are defaults.
    """
    if isinstance(py_result, (str, Path)):
        py_result = pd.read_csv(py_result, sep="\t", index_col=0)

    r_df = pd.read_csv(r_result_path, sep="\t", index_col=0)

    common_idx = py_result.index.intersection(r_df.index)
    common_cols = py_result.columns.intersection(r_df.columns)

    py_only = len(py_result.index.difference(r_df.index))
    r_only = len(r_df.index.difference(py_result.index))

    vr = ValidityResult(scenario_id="")
    vr.concordance_py_only_features = py_only
    vr.concordance_r_only_features = r_only

    if len(common_idx) == 0 or len(common_cols) == 0:
        return vr

    vr.concordance_shared_features = len(common_idx)

    p = py_result.loc[common_idx, common_cols].to_numpy(dtype=np.float64)
    r = r_df.loc[common_idx, common_cols].to_numpy(dtype=np.float64)

    vr.concordance_nan_match = bool((np.isnan(p) == np.isnan(r)).all())

    mask = ~(np.isnan(p) | np.isnan(r))
    n_nonnan = int(mask.sum())
    vr.concordance_shared_nonnan_cells = n_nonnan

    if n_nonnan == 0:
        return vr

    rel_diff = np.abs(p[mask] - r[mask]) / np.maximum(np.abs(r[mask]), 1e-12)
    vr.concordance_max_rel = float(rel_diff.max())
    vr.concordance_mean_rel = float(rel_diff.mean())
    vr.concordance_p95_rel = float(np.percentile(rel_diff, 95))

    return vr
