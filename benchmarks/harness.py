"""Benchmark harness: dataset-at-a-time scenario orchestration.

The harness loads one dataset at a time, runs all its scenarios
(validity warmup + timed reps), then releases the dataset before
loading the next.  This caps peak memory to roughly one dataset.

Usage::

    from benchmarks.harness import BenchmarkHarness
    from benchmarks.scenarios import load_config, build_registry

    cfg = load_config("benchmarks/config.yaml")
    registry = build_registry(cfg)
    harness = BenchmarkHarness(cfg, registry)
    results = harness.run_all()
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from .datasets import load_dataset, resolve_dataset_paths
from .metrics import ScenarioMetrics, aggregate_metrics
from .runners.python_runner import run_once, run_scenario
from .scenarios import Config, Scenario
from .validity import ValidityResult, validate_result

logger = logging.getLogger(__name__)

# Type alias for a full scenario result
ScenarioResult = tuple[Scenario, ScenarioMetrics, ValidityResult | None, dict[str, Any] | None]


class BenchmarkHarness:
    """Orchestrates benchmark runs across datasets and scenarios.

    Parameters
    ----------
    config : Config
        Loaded benchmark configuration.
    scenarios : list[Scenario]
        Scenarios to run (from ``build_registry``, optionally filtered).
    budget_s : int
        Per-scenario time budget for adaptive repetition (default from config).
    min_reps : int
        Minimum timed repetitions per scenario (default from config).
    max_reps : int
        Maximum timed repetitions per scenario (default from config).
    warmup : bool
        Run an untimed warmup for validity (default True).
    """

    def __init__(
        self,
        config: Config,
        scenarios: list[Scenario],
        *,
        budget_s: int | None = None,
        min_reps: int | None = None,
        max_reps: int | None = None,
        warmup: bool = True,
    ) -> None:
        self.config = config
        self.scenarios = scenarios
        self.budget_s = budget_s or config.python_budget_s
        self.min_reps = min_reps or config.python_min_reps
        self.max_reps = max_reps or config.python_max_reps
        self.warmup = warmup

        # Group scenarios by dataset for dataset-at-a-time loading
        self._by_dataset: dict[str, list[Scenario]] = defaultdict(list)
        for s in scenarios:
            self._by_dataset[s.dataset].append(s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> list[ScenarioResult]:
        """Run all scenarios across all datasets.

        Returns
        -------
        list[ScenarioResult]
            One entry per scenario, each containing the scenario,
            aggregated metrics, validity result, and optional R
            cache entry.
        """
        results: list[ScenarioResult] = []

        for dataset_name, dataset_scenarios in sorted(self._by_dataset.items()):
            logger.info("Loading dataset '%s' (%d scenarios)", dataset_name, len(dataset_scenarios))

            # Resolve paths and load data
            paths = resolve_dataset_paths(self.config, dataset_name)
            data_df, desc_df = self._load_dataset(paths)

            # Assert dataset dimensions match config (catch stale config)
            if data_df.shape[0] > paths.features:
                raise ValueError(
                    f"Dataset '{dataset_name}': expected {paths.features} features, "
                    f"got {data_df.shape[0]}. Config may be stale."
                )
            if data_df.shape[0] < paths.features:
                logger.info(
                    "Dataset '%s' loaded %d/%d configured features after IO normalization",
                    dataset_name,
                    data_df.shape[0],
                    paths.features,
                )
            if data_df.shape[1] != paths.samples:
                raise ValueError(
                    f"Dataset '{dataset_name}': expected {paths.samples} samples, "
                    f"got {data_df.shape[1]}. Config may be stale."
                )
            if "batch" in desc_df.columns:
                n_batches = int(desc_df["batch"].nunique())
            else:
                n_batches = int(desc_df.iloc[:, 2].nunique())
            if n_batches != paths.batches:
                raise ValueError(
                    f"Dataset '{dataset_name}': expected {paths.batches} batches, "
                    f"got {n_batches}. Config may be stale."
                )

            # --- Resolve R results once per dataset (shared across scenarios) ---
            r_entries: dict[str, dict[str, Any] | None] = {}
            if getattr(self.config, "r_cache_default_cores", None) is not None:
                try:
                    from .runners.r_runner import resolve_r_results
                    r_entries = resolve_r_results(
                        dataset_scenarios, self.config,
                        self.config.paths_cache_dir,
                    )
                except Exception as exc:
                    logger.debug("Could not resolve R results: %s", exc)

            for scenario in dataset_scenarios:
                logger.info("  Running %s ...", scenario.id)
                result = self._run_single_scenario(data_df, desc_df, scenario, r_entries)
                results.append(result)

            # Release dataset
            del data_df, desc_df
            logger.info("  Released dataset '%s'", dataset_name)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dataset(paths: Any) -> tuple[Any, Any]:
        """Load a dataset from resolved paths.

        Uses HarmonizePy's production IO layer so benchmark loading matches
        real application parsing behavior.
        """
        return load_dataset(paths)

    def _run_single_scenario(
        self,
        data_df: Any,
        desc_df: Any,
        scenario: Scenario,
        r_entries: dict[str, dict[str, Any] | None] | None = None,
    ) -> ScenarioResult:
        """Run warmup + timed reps for one scenario."""

        # --- Look up R results for this scenario ---
        r_entry: dict[str, Any] | None = None
        if r_entries is not None:
            r_entry = r_entries.get(scenario.id)

        # --- Warmup (untimed, for validity + concordance) ---
        validity: ValidityResult | None = None

        if self.warmup:
            warmup_metrics = run_once(data_df, desc_df, scenario)
            result_dict = warmup_metrics.result
            if result_dict is not None and "result_df" in result_dict:
                result_df = result_dict["result_df"]
                validity = validate_result(data_df, result_df, scenario.id)

                # Compute concordance if R output is available
                if r_entry is not None and r_entry.get("output_tsv"):
                    try:
                        from .validity import compute_concordance
                        cv = compute_concordance(result_df, r_entry["output_tsv"])
                        validity.concordance_max_rel = cv.concordance_max_rel
                        validity.concordance_mean_rel = cv.concordance_mean_rel
                        validity.concordance_nan_match = cv.concordance_nan_match
                        validity.concordance_shared_features = cv.concordance_shared_features
                        validity.concordance_py_only_features = cv.concordance_py_only_features
                        validity.concordance_r_only_features = cv.concordance_r_only_features
                        validity.concordance_shared_nonnan_cells = cv.concordance_shared_nonnan_cells
                        validity.concordance_p95_rel = cv.concordance_p95_rel
                    except Exception as exc:
                        logger.debug("Concordance failed: %s", exc)

                warmup_metrics.release_result()
            else:
                validity = ValidityResult(
                    scenario_id=scenario.id,
                    error="No result from warmup run",
                )

            if validity is not None and validity.error is not None:
                logger.warning("  Validity error: %s", validity.error)

        # --- Timed runs ---
        timed = run_scenario(
            data_df, desc_df, scenario,
            budget_s=self.budget_s,
            min_reps=self.min_reps,
            max_reps=self.max_reps,
        )

        # Extract feature counts from the first timed rep's result
        metrics = aggregate_metrics(timed)
        first_result = timed[0].result
        if isinstance(first_result, dict):
            metrics.features_out = first_result.get("n_total", 0)
            metrics.features_corrected = first_result.get("n_corrected", 0)
            metrics.features_passthrough = first_result.get("n_passthrough", 0)

        rss_stability = metrics.rss_stability
        if rss_stability.status == "growing":
            logger.warning(
                "  RSS stability warning for %s: %s (growth=%d KB, tail_span=%d KB)",
                scenario.id,
                rss_stability.reason,
                rss_stability.total_growth_kb,
                rss_stability.tail_span_kb,
            )

        # Release result DataFrames from timed runs
        for m in timed:
            m.release_result()

        return (scenario, metrics, validity, r_entry)
