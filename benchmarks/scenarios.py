"""Benchmark scenario definitions and registry builder.

A ``Scenario`` is a single benchmark run: one dataset, one algorithm,
one set of parameters.  The registry is built from ``config.yaml`` by
expanding the scenario matrix against dataset group memberships.

Usage::

    from benchmarks.scenarios import load_config, build_registry, filter_registry

    cfg = load_config("benchmarks/config.yaml")
    registry = build_registry(cfg)
    filtered = filter_registry(registry, datasets=["medium"], tags=["bulk"])
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for a single dataset entry from config.yaml."""

    input: str
    desc: str
    tags: list[str]
    r_eligible: bool
    features: int
    samples: int
    batches: int
    scenarios: list[str]
    input_format: str = "tsv"
    missing_frac: float | None = None
    missing_mode: str = "per_batch"


@dataclass(frozen=True)
class ScenarioTemplate:
    """One row from a scenario_matrix group, before mode expansion."""

    algorithm: str
    modes: list[int] | None = None
    block: int | None = None
    sort: str | None = None


@dataclass(frozen=True)
class Config:
    """Parsed config.yaml contents."""

    datasets: dict[str, DatasetSpec]
    scenario_matrix: dict[str, list[ScenarioTemplate]]
    r_cache_cores: list[int]
    r_cache_default_cores: int
    r_cache_timeout_s: int
    python_budget_s: int
    python_min_reps: int
    python_max_reps: int
    python_warmup: bool
    r_budget_s: int
    r_min_reps: int
    r_max_reps: int
    paths_data_dir: str
    paths_results_dir: str
    paths_tmp_dir: str
    paths_cache_dir: str


def load_config(path: str | Path) -> Config:
    """Load and validate a ``config.yaml`` file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    Config
        Parsed configuration with all sections populated.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ImportError
        If ``pyyaml`` is not installed.
    ValueError
        If required sections are missing or malformed.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "Benchmark config requires pyyaml: pip install harmonizepy[config]"
        ) from None

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open() as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}")

    # -- Datasets --
    raw_ds = raw.get("datasets")
    if not isinstance(raw_ds, dict):
        raise ValueError("Config missing required 'datasets' section")

    datasets: dict[str, DatasetSpec] = {}
    for name, entry in raw_ds.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Dataset '{name}' must be a mapping")
        datasets[name] = DatasetSpec(
            input=entry["input"],
            desc=entry["desc"],
            tags=entry.get("tags", []),
            r_eligible=bool(entry.get("r_eligible", False)),
            features=int(entry["features"]),
            samples=int(entry["samples"]),
            batches=int(entry["batches"]),
            scenarios=entry.get("scenarios", []),
            input_format=entry.get("input_format", "tsv"),
            missing_frac=entry.get("missing_frac"),
            missing_mode=entry.get("missing_mode", "per_batch"),
        )

    # -- Scenario matrix --
    raw_sm = raw.get("scenario_matrix")
    if not isinstance(raw_sm, dict):
        raise ValueError("Config missing required 'scenario_matrix' section")

    scenario_matrix: dict[str, list[ScenarioTemplate]] = {}
    for group_name, templates in raw_sm.items():
        if not isinstance(templates, list):
            raise ValueError(f"Scenario group '{group_name}' must be a list")
        parsed: list[ScenarioTemplate] = []
        for t in templates:
            if not isinstance(t, dict):
                raise ValueError(f"Scenario template in '{group_name}' must be a mapping")
            parsed.append(
                ScenarioTemplate(
                    algorithm=t["algorithm"],
                    modes=t.get("modes"),
                    block=t.get("block"),
                    sort=t.get("sort"),
                )
            )
        scenario_matrix[group_name] = parsed

    # -- R cache --
    raw_rc = raw.get("r_cache", {})
    r_cache_cores = raw_rc.get("cores_variants", [16, 1])
    r_cache_default_cores = raw_rc.get("default_cores", 16)
    r_cache_timeout_s = raw_rc.get("timeout_s", 300)

    # -- Python runner --
    raw_py = raw.get("python_runner", {})
    python_budget_s = raw_py.get("budget_s", 30)
    python_min_reps = raw_py.get("min_reps", 3)
    python_max_reps = raw_py.get("max_reps", 10)
    python_warmup = raw_py.get("warmup", True)

    # -- R runner --
    raw_r = raw.get("r_runner", {})
    r_budget_s = raw_r.get("budget_s", 180)
    r_min_reps = raw_r.get("min_reps", 2)
    r_max_reps = raw_r.get("max_reps", 5)

    # -- Paths (resolved relative to config file location) --
    config_dir = p.parent.resolve()
    raw_paths = raw.get("paths", {})
    paths_data_dir = str(config_dir / raw_paths.get("data_dir", "data"))
    paths_results_dir = str(config_dir / raw_paths.get("results_dir", "results"))
    paths_tmp_dir = str(config_dir / raw_paths.get("tmp_dir", "results/tmp"))
    paths_cache_dir = str(config_dir / raw_paths.get("cache_dir", "results/r_cache"))

    return Config(
        datasets=datasets,
        scenario_matrix=scenario_matrix,
        r_cache_cores=r_cache_cores,
        r_cache_default_cores=r_cache_default_cores,
        r_cache_timeout_s=r_cache_timeout_s,
        python_budget_s=python_budget_s,
        python_min_reps=python_min_reps,
        python_max_reps=python_max_reps,
        python_warmup=python_warmup,
        r_budget_s=r_budget_s,
        r_min_reps=r_min_reps,
        r_max_reps=r_max_reps,
        paths_data_dir=paths_data_dir,
        paths_results_dir=paths_results_dir,
        paths_tmp_dir=paths_tmp_dir,
        paths_cache_dir=paths_cache_dir,
    )


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """A single benchmark run: dataset + algorithm + parameter combination.

    The ``id`` property produces a unique, filesystem-safe string key
    like ``medium_ComBat_m3_b2_sparsity``.
    """

    dataset: str
    algorithm: str
    combat_mode: int | None = None
    block: int | None = None
    sort: str | None = None
    tags: frozenset[str] = field(default_factory=frozenset)

    @property
    def id(self) -> str:
        """Unique scenario identifier built from parameter components."""
        parts = [self.dataset, self.algorithm]
        if self.combat_mode is not None:
            parts.append(f"m{self.combat_mode}")
        if self.block is not None:
            parts.append(f"b{self.block}")
        if self.sort is not None:
            parts.append(self.sort)
        return "_".join(parts)

    @property
    def cache_key(self) -> str:
        """Short hash for R cache directory naming.

        The key is a SHA-256 truncated to 12 hex chars, computed from
        the scenario id.  When full cache invalidation is needed (e.g.,
        dataset file changed), callers should incorporate file hashes
        externally.
        """
        h = hashlib.sha256(self.id.encode()).hexdigest()[:12]
        return h


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------


def build_registry(cfg: Config) -> list[Scenario]:
    """Build the full benchmark scenario matrix from a loaded config.

    For each dataset, iterates over its declared scenario groups, expands
    mode combinations from the scenario matrix, and yields one ``Scenario``
    per (dataset, algorithm, combat_mode, block, sort).

    Parameters
    ----------
    cfg : Config
        Parsed configuration from ``load_config``.

    Returns
    -------
    list[Scenario]
        All scenarios, with duplicates rejected.

    Raises
    ------
    ValueError
        If a dataset references a scenario group not in the matrix,
        or if two scenarios produce the same ``id``.
    """
    seen_ids: dict[str, str] = {}
    registry: list[Scenario] = []

    for ds_name, ds_spec in cfg.datasets.items():
        for group_name in ds_spec.scenarios:
            if group_name not in cfg.scenario_matrix:
                raise ValueError(
                    f"Dataset '{ds_name}' references scenario group '{group_name}' "
                    f"not found in scenario_matrix"
                )

            templates = cfg.scenario_matrix[group_name]
            for tmpl in templates:
                # Expand mode combinations
                if tmpl.algorithm == "limma":
                    # limma has no mode parameter; emit one scenario
                    modes_to_emit: list[int] | list[None] = [None]
                elif tmpl.modes is not None:
                    modes_to_emit = tmpl.modes
                else:
                    # Default: all 4 ComBat modes
                    modes_to_emit = [1, 2, 3, 4]

                for mode in modes_to_emit:
                    # Build scenario-level tags
                    extra_tags: set[str] = set()
                    if tmpl.algorithm == "ComBat":
                        if mode in (1, 2):
                            extra_tags.add("parametric")
                        else:
                            extra_tags.add("nonparametric")
                    if tmpl.block is not None:
                        extra_tags.add("blocking")
                    if tmpl.sort is not None:
                        extra_tags.add("sorted")
                    if not ds_spec.r_eligible:
                        extra_tags.add("py_only")

                    scenario = Scenario(
                        dataset=ds_name,
                        algorithm=tmpl.algorithm,
                        combat_mode=mode,
                        block=tmpl.block,
                        sort=tmpl.sort,
                        tags=frozenset(ds_spec.tags) | extra_tags,
                    )

                    sid = scenario.id
                    if sid in seen_ids:
                        raise ValueError(
                            f"Duplicate scenario id '{sid}': "
                            f"from datasets '{seen_ids[sid]}' and '{ds_name}'"
                        )
                    seen_ids[sid] = ds_name
                    registry.append(scenario)

    return registry


def filter_registry(
    registry: list[Scenario],
    *,
    datasets: list[str] | None = None,
    algorithms: list[str] | None = None,
    tags: list[str] | None = None,
) -> list[Scenario]:
    """Filter the scenario registry by dataset, algorithm, or tags.

    Parameters
    ----------
    registry : list[Scenario]
        Full scenario registry from ``build_registry``.
    datasets : list[str] or None
        If given, only scenarios matching these dataset names.
    algorithms : list[str] or None
        If given, only scenarios matching these algorithms.
    tags : list[str] or None
        If given, only scenarios whose tags contain ALL specified tags.

    Returns
    -------
    list[Scenario]
        Filtered scenarios preserving original order.
    """
    result = registry

    if datasets is not None:
        ds_set = set(datasets)
        result = [s for s in result if s.dataset in ds_set]

    if algorithms is not None:
        algo_set = set(algorithms)
        result = [s for s in result if s.algorithm in algo_set]

    if tags is not None:
        tag_set = set(tags)
        result = [s for s in result if tag_set <= s.tags]

    return result
