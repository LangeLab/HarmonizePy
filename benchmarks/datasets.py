"""Dataset definitions, generation, and path resolution for benchmarks.

Absorbs the data generation logic from the old ``generate_data.py`` and
adds a lookup layer that resolves dataset short names to file paths
using the config.yaml ``datasets`` section.

Usage::

    from benchmarks.datasets import generate_dataset, resolve_dataset_paths

    # Generate a synthetic dataset
    data_df, desc_df = generate_dataset("medium", seed=42)

    # Resolve paths from config
    paths = resolve_dataset_paths(cfg, "medium")
    # paths.input, paths.desc, paths.features, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .scenarios import Config

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_BENCHMARKS_DIR = Path(__file__).parent


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved file paths and metadata for a dataset."""

    name: str
    input_path: Path
    desc_path: Path
    features: int
    samples: int
    batches: int
    input_format: str
    r_eligible: bool


def resolve_dataset_paths(cfg: Config, dataset_name: str) -> DatasetPaths:
    """Resolve a dataset short name to absolute file paths.

    Parameters
    ----------
    cfg : Config
        Parsed configuration from ``load_config``.
    dataset_name : str
        Short dataset name (e.g., ``"medium"``, ``"murine"``).

    Returns
    -------
    DatasetPaths
        Resolved paths and metadata.

    Raises
    ------
    KeyError
        If *dataset_name* is not in the config datasets.
    """
    ds = cfg.datasets[dataset_name]
    return DatasetPaths(
        name=dataset_name,
        input_path=_BENCHMARKS_DIR / ds.input,
        desc_path=_BENCHMARKS_DIR / ds.desc,
        features=ds.features,
        samples=ds.samples,
        batches=ds.batches,
        input_format=ds.input_format,
        r_eligible=ds.r_eligible,
    )


def assert_dataset_exists(paths: DatasetPaths) -> None:
    """Verify that dataset files exist on disk.

    Raises
    ------
    FileNotFoundError
        If either the input or description file is missing.
    """
    if not paths.input_path.is_file():
        raise FileNotFoundError(f"Dataset input file not found: {paths.input_path}")
    if not paths.desc_path.is_file():
        raise FileNotFoundError(f"Dataset description file not found: {paths.desc_path}")


def load_dataset(paths: DatasetPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a benchmark dataset using HarmonizePy's production IO layer.

    This keeps benchmark dataset parsing aligned with the actual application
    behavior, including extension-based format detection and input cleanup.
    """
    from harmonizepy.io import read_description, read_main_data

    assert_dataset_exists(paths)
    data_df = read_main_data(str(paths.input_path))
    desc_df = read_description(str(paths.desc_path))
    return data_df, desc_df


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

_SEED = 42
_BATCH_EFFECT_STRENGTH = 2.0
_BASELINE_MEAN = 10.0
_BASELINE_STD = 2.0
_MIN_SAMPLES_PER_BATCH = 4


def _zipfian_batch_sizes(n_batches: int, n_samples: int, rng: np.random.Generator) -> list[int]:
    """Distribute *n_samples* across *n_batches* using a Zipf-like distribution.

    Guarantees each batch gets at least ``_MIN_SAMPLES_PER_BATCH`` samples
    so that ``needed_values=2`` can be satisfied even after structural
    missingness.  Returns a list of integers summing to *n_samples*.
    """
    min_total = n_batches * _MIN_SAMPLES_PER_BATCH
    if min_total > n_samples:
        raise ValueError(
            f"n_batches ({n_batches}) x min_samples_per_batch ({_MIN_SAMPLES_PER_BATCH}) "
            f"= {min_total} exceeds n_samples ({n_samples}). "
            f"Increase n_samples or reduce n_batches."
        )

    sizes = [_MIN_SAMPLES_PER_BATCH] * n_batches
    remaining = n_samples - min_total

    if remaining > 0:
        raw = rng.zipf(1.5, size=n_batches)
        proportions = raw / raw.sum()
        extra = (proportions * remaining).astype(int)
        extra[-1] += remaining - extra.sum()
        for i in range(n_batches):
            sizes[i] += int(extra[i])

    return sizes


def _assign_batch_labels(batch_sizes: list[int]) -> np.ndarray:
    """Create a batch label array where label ``i`` appears ``batch_sizes[i]`` times."""
    labels: list[int] = []
    for i, size in enumerate(batch_sizes):
        labels.extend([i + 1] * size)
    return np.array(labels, dtype=np.int64)


def _apply_abundance_missingness(
    data: np.ndarray,
    batch_labels: np.ndarray,
    missing_frac: float,
    rng: np.random.Generator,
) -> None:
    """Apply abundance-dependent structural missingness in-place.

    Features are assigned to a small set of detection profiles (shared
    batch-presence patterns).  Features in the same profile share an
    identical set of batches where they have non-NA data, forming
    correction groups for ComBat/limma.
    """
    n_features = data.shape[0]
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)

    n_profiles = max(20, n_features // 40)
    target_present = 1.0 - missing_frac
    profile_patterns = rng.random((n_profiles, n_batches)) < target_present

    for i in range(n_profiles):
        if profile_patterns[i].sum() < 2:
            profile_patterns[i, rng.choice(n_batches, 2, replace=False)] = True

    for j in range(n_batches):
        if not profile_patterns[:, j].any():
            profile_patterns[rng.choice(n_profiles), j] = True

    feature_profile = rng.choice(n_profiles, size=n_features)
    profile_in_batch = profile_patterns[feature_profile]

    for j, b in enumerate(unique_batches):
        mask = batch_labels == b
        absent = np.where(profile_in_batch[:, j] == 0)[0]
        smp = np.where(mask)[0]
        if len(absent) and len(smp):
            data[np.ix_(absent, smp)] = np.nan


def _apply_per_batch_missingness(
    data: np.ndarray,
    batch_labels: np.ndarray,
    missing_frac: float,
    rng: np.random.Generator,
) -> None:
    """Apply per-batch random structural missingness in-place (legacy model).

    Each batch independently picks ``missing_frac * n_features`` random
    features and marks them as entirely absent.
    """
    n_features = data.shape[0]
    n_present = max(1, n_features - int(n_features * missing_frac))
    unique_batches = np.unique(batch_labels)
    for b in unique_batches:
        absent = rng.choice(n_features, size=n_features - n_present, replace=False)
        mask = batch_labels == b
        data[np.ix_(absent, mask)] = np.nan


def generate_dataset(
    name: str,
    n_features: int,
    n_samples: int,
    n_batches: int,
    missing_frac: float,
    *,
    missing_mode: str = "per_batch",
    seed: int = _SEED,
    batch_effect_strength: float = _BATCH_EFFECT_STRENGTH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic dataset with batch effects and structural missingness.

    Parameters
    ----------
    name : str
        Dataset name, used as the filename stem.
    n_features : int
        Number of features (rows).
    n_samples : int
        Number of samples (columns).
    n_batches : int
        Number of batches.
    missing_frac : float
        Fraction of features that are structurally absent (entirely NaN) per batch.
    missing_mode : str
        ``"per_batch"`` (default): each batch drops random independent features.
        ``"abundance"``: abundance-dependent dropout where features with higher
        simulated abundance are detected more consistently across batches.
    seed : int
        RNG seed for reproducibility.
    batch_effect_strength : float
        Standard deviation of the per-batch additive shift.

    Returns
    -------
    data_df : DataFrame
        Features x samples matrix (float64), index = ``f0..fN``, columns = ``s0..sM``.
    desc_df : DataFrame
        Description with columns ``ID``, ``sample``, ``batch``.
    """
    rng = np.random.default_rng(seed)

    data = rng.normal(_BASELINE_MEAN, _BASELINE_STD, size=(n_features, n_samples))

    batch_sizes = _zipfian_batch_sizes(n_batches, n_samples, rng)
    batch_labels = _assign_batch_labels(batch_sizes)
    assert len(batch_labels) == n_samples

    for b in range(n_batches):
        mask = batch_labels == (b + 1)
        shift = rng.normal(0, batch_effect_strength)
        data[:, mask] += shift

    if missing_mode == "abundance":
        _apply_abundance_missingness(data, batch_labels, missing_frac, rng)
    else:
        _apply_per_batch_missingness(data, batch_labels, missing_frac, rng)

    data_df = pd.DataFrame(
        data,
        index=[f"f{i}" for i in range(n_features)],
        columns=[f"s{j}" for j in range(n_samples)],
    )

    desc_df = pd.DataFrame(
        {
            "ID": data_df.columns.tolist(),
            "sample": list(range(1, n_samples + 1)),
            "batch": batch_labels.tolist(),
        }
    )

    return data_df, desc_df


def write_dataset(
    data_df: pd.DataFrame,
    desc_df: pd.DataFrame,
    name: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write dataset files to *output_dir*.

    Returns
    -------
    input_path, desc_path
        Paths to the written files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / f"{name}_input.tsv"
    desc_path = output_dir / f"{name}_batch.csv"
    data_df.to_csv(input_path, sep="\t")
    desc_df.to_csv(desc_path, index=False)
    return input_path, desc_path


def generate_from_config(
    cfg: Config,
    dataset_name: str,
    *,
    output_dir: Path | None = None,
    seed: int = _SEED,
) -> tuple[Path, Path]:
    """Generate a dataset using config.yaml parameters.

    Parameters
    ----------
    cfg : Config
        Parsed configuration.
    dataset_name : str
        Short dataset name from config.
    output_dir : Path or None
        Where to write files.  Defaults to ``benchmarks/data/``.
    seed : int
        RNG seed.

    Returns
    -------
    input_path, desc_path
        Paths to the written files.

    Raises
    ------
    KeyError
        If *dataset_name* is not in the config.
    """
    ds = cfg.datasets[dataset_name]
    if output_dir is None:
        output_dir = _BENCHMARKS_DIR / cfg.paths_data_dir

    data_df, desc_df = generate_dataset(
        name=dataset_name,
        n_features=ds.features,
        n_samples=ds.samples,
        n_batches=ds.batches,
        missing_frac=ds.missing_frac or 0.0,
        missing_mode=ds.missing_mode,
        seed=seed,
    )
    return write_dataset(data_df, desc_df, dataset_name, output_dir)
