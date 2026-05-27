"""Synthetic data generator for the HarmonizePy benchmark suite.

Generates features x samples matrices with known batch effects and
structural missingness, matching the format expected by both
HarmonizePy and R HarmonizR (TSV data + CSV description).

Datasets are deterministic given a seed. Three standard sizes are
pre-configured: small (1K x 20), medium (5K x 60), large (10K x 100).
Custom sizes are supported via explicit flags.

Usage::

    # Generate all standard datasets
    python benchmarks/generate_data.py --dataset all

    # Generate just the medium dataset
    python benchmarks/generate_data.py --dataset medium

    # Custom size
    python benchmarks/generate_data.py --features 2000 --samples 30 --batches 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_OUTPUT_DIR = Path(__file__).parent / "data"

_DATASETS: dict[str, dict[str, int | float | str]] = {
    "small": {
        "n_features": 1000,
        "n_samples": 20,
        "n_batches": 5,
        "missing_frac": 0.30,
        "missing_mode": "per_batch",
    },
    "medium": {
        "n_features": 5000,
        "n_samples": 60,
        "n_batches": 10,
        "missing_frac": 0.20,
        "missing_mode": "per_batch",
    },
    "large": {
        "n_features": 10000,
        "n_samples": 100,
        "n_batches": 20,
        "missing_frac": 0.05,
        "missing_mode": "per_batch",
    },
    # Single-cell proteomics cohorts: abundance-dependent missingness so
    # features share detection patterns and form correction groups.
    "scp_small": {
        "n_features": 3000,
        "n_samples": 1000,
        "n_batches": 20,
        "missing_frac": 0.50,
        "missing_mode": "abundance",
    },
    "scp_large": {
        "n_features": 5000,
        "n_samples": 10000,
        "n_batches": 100,
        "missing_frac": 0.60,
        "missing_mode": "abundance",
    },
}

_SEED = 42
_BATCH_EFFECT_STRENGTH = 2.0
_BASELINE_MEAN = 10.0
_BASELINE_STD = 2.0


_MIN_SAMPLES_PER_BATCH = 4


def _zipfian_batch_sizes(n_batches: int, n_samples: int, rng: np.random.Generator) -> list[int]:
    """Distribute *n_samples* across *n_batches* using a Zipf-like distribution.

    Guarantees each batch gets at least ``_MIN_SAMPLES_PER_BATCH`` samples
    (default 4) so that ``needed_values=2`` can be satisfied even after
    structural missingness. Returns a list of integers summing to *n_samples*.
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

    High-abundance profiles are present in most batches; low-abundance
    profiles are present in few batches.  The number of profiles is
    ``max(20, n_features // 40)`` so typical group size is ~40 features.
    """
    n_features = data.shape[0]
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)

    # Number of shared detection profiles
    n_profiles = max(20, n_features // 40)

    # Generate profile patterns: each profile is present in a random subset
    # of batches.  The presence probability per batch is calibrated so the
    # overall missing rate matches *missing_frac*.
    target_present = 1.0 - missing_frac
    profile_patterns = rng.random((n_profiles, n_batches)) < target_present

    # Ensure every profile has >= 2 batches (minimum for correction)
    for i in range(n_profiles):
        if profile_patterns[i].sum() < 2:
            profile_patterns[i, rng.choice(n_batches, 2, replace=False)] = True

    # Ensure every batch is covered by at least one profile
    for j in range(n_batches):
        if not profile_patterns[:, j].any():
            profile_patterns[rng.choice(n_profiles), j] = True

    # Assign each feature to a random profile
    feature_profile = rng.choice(n_profiles, size=n_features)

    # Vectorised missingness: for each batch, features whose profile does
    # not include that batch are set to NaN.
    # profile_in_batch[f, j] = profile_patterns[feature_profile[f], j]
    profile_in_batch = profile_patterns[feature_profile]  # (n_features, n_batches)
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
    features and marks them as entirely absent.  This creates mostly
    unique detection patterns per feature when ``missing_frac`` is high
    and the number of batches is large.
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


def _write_dataset(
    data_df: pd.DataFrame, desc_df: pd.DataFrame, name: str, output_dir: Path
) -> None:
    """Write dataset files to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"{name}_input.tsv"
    desc_path = output_dir / f"{name}_batch.csv"
    data_df.to_csv(data_path, sep="\t")
    desc_df.to_csv(desc_path, index=False)
    print(f"  {data_path}  ({data_df.shape[0]} x {data_df.shape[1]})")
    print(f"  {desc_path}  ({desc_df.shape[0]} rows)")


def generate_all_datasets(output_dir: Path = _OUTPUT_DIR, seed: int = _SEED) -> None:
    """Generate all three standard datasets."""
    for name, params in _DATASETS.items():
        print(f"Generating '{name}' dataset...")
        data_df, desc_df = generate_dataset(
            name=name,
            n_features=int(params["n_features"]),
            n_samples=int(params["n_samples"]),
            n_batches=int(params["n_batches"]),
            missing_frac=float(params["missing_frac"]),
            missing_mode=str(params.get("missing_mode", "per_batch")),
            seed=seed,
        )
        _write_dataset(data_df, desc_df, name, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark datasets.")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["small", "medium", "large", "scp_small", "scp_large", "all"],
        help="Dataset to generate (default: all).",
    )
    parser.add_argument("--features", type=int, default=None, help="Number of features (custom).")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples (custom).")
    parser.add_argument("--batches", type=int, default=None, help="Number of batches (custom).")
    parser.add_argument("--missing", type=float, default=0.3, help="Missingness fraction (custom).")
    parser.add_argument(
        "--missing-mode",
        default=None,
        choices=["per_batch", "abundance"],
        help="Missingness model: per_batch (default, random independent dropout) or "
        "abundance (abundance-dependent, more realistic for SCP).",
    )
    parser.add_argument("--seed", type=int, default=_SEED, help="RNG seed.")
    parser.add_argument("--output-dir", type=Path, default=_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args()

    if args.features is not None or args.samples is not None or args.batches is not None:
        print("Generating custom dataset...")
        missing_mode = args.missing_mode or "per_batch"
        data_df, desc_df = generate_dataset(
            name="custom",
            n_features=args.features or 1000,
            n_samples=args.samples or 20,
            n_batches=args.batches or 5,
            missing_frac=args.missing,
            missing_mode=missing_mode,
            seed=args.seed,
        )
        _write_dataset(data_df, desc_df, "custom", args.output_dir)
    elif args.dataset == "all":
        generate_all_datasets(output_dir=args.output_dir, seed=args.seed)
    else:
        name = args.dataset
        params = _DATASETS[name]
        print(f"Generating '{name}' dataset...")
        data_df, desc_df = generate_dataset(
            name=name,
            n_features=int(params["n_features"]),
            n_samples=int(params["n_samples"]),
            n_batches=int(params["n_batches"]),
            missing_frac=float(params["missing_frac"]),
            missing_mode=str(params.get("missing_mode", "per_batch")),
            seed=args.seed,
        )
        _write_dataset(data_df, desc_df, name, args.output_dir)


if __name__ == "__main__":
    main()
