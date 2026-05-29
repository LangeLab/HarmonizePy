"""Focused tests for benchmark runner and harness behavior.

These tests guard benchmark-integrity behavior rather than algorithm output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from benchmarks.harness import BenchmarkHarness
from benchmarks.metrics import SingleRunMetrics, aggregate_metrics, assess_rss_stability
from benchmarks.report import build_report_json
from benchmarks.runners import python_runner
from benchmarks.scenarios import Config, Scenario


def _make_input() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_df = pd.DataFrame(
        {
            "s1": [1.0, 2.0],
            "s2": [1.5, 2.5],
            "s3": [3.0, 4.0],
            "s4": [3.5, 4.5],
        },
        index=["f1", "f2"],
    )
    desc_df = pd.DataFrame(
        {
            "ID": ["s1", "s2", "s3", "s4"],
            "sample": [1, 2, 3, 4],
            "batch": [1, 1, 2, 2],
        }
    )
    return data_df, desc_df


def _make_config() -> Config:
    return Config(
        datasets={},
        scenario_matrix={},
        r_cache_cores=[1],
        r_cache_default_cores=1,
        r_cache_timeout_s=300,
        python_budget_s=1,
        python_min_reps=1,
        python_max_reps=1,
        python_warmup=True,
        r_budget_s=1,
        r_min_reps=1,
        r_max_reps=1,
        paths_data_dir="",
        paths_results_dir="",
        paths_tmp_dir="",
        paths_cache_dir="",
    )


class TestPythonRunner:
    def test_run_once_records_post_gc_rss(self, monkeypatch) -> None:
        """RSS metrics should distinguish raw end-of-call and retained post-GC RSS.

        Failure condition: the runner reports only a noisy before/after delta
        and cannot separate transient allocations from retained process RSS.
        """
        rss_reads = iter([100, 140, 110])

        monkeypatch.setattr(python_runner, "_read_proc_rss_kb", lambda: next(rss_reads))
        monkeypatch.setattr(python_runner.gc, "collect", lambda: 0)

        data_df, desc_df = _make_input()
        scenario = Scenario(dataset="tiny", algorithm="limma")

        metrics = python_runner.run_once(
            data_df,
            desc_df,
            scenario,
            keep_result_df=False,
        )

        assert metrics.rss_before_kb == 100
        assert metrics.rss_after_kb == 140
        assert metrics.rss_post_gc_kb == 110
        assert metrics.rss_delta_kb == 40
        assert metrics.rss_delta_post_gc_kb == 10


class TestBenchmarkMetrics:
    def test_assess_rss_stability_plateau_after_growth(self) -> None:
        """Warm-up growth followed by a stable tail should report plateau.

        Failure condition: repeated-run RSS that stabilizes is mislabeled as
        ongoing growth, making memory behavior look worse than it is.
        """
        stability = assess_rss_stability([100, 180, 182, 181])

        assert stability.status == "plateau"
        assert stability.total_growth_kb == 81
        assert stability.tail_span_kb == 2

    def test_assess_rss_stability_growing(self) -> None:
        """Monotonic repeated-run RSS growth without a stable tail is growing.

        Failure condition: the workflow hides obviously unbounded repeated-run
        growth and reports it as stable.
        """
        stability = assess_rss_stability([100, 2400, 4200])

        assert stability.status == "growing"
        assert stability.monotonic_non_decreasing is True

    def test_run_once_can_skip_result_frame(self) -> None:
        """Timed runs may drop the result DataFrame while keeping scalar counts.

        Failure condition: benchmark timing still retains a full ``result_df``
        even when only counts are needed for aggregation.
        """
        data_df, desc_df = _make_input()
        scenario = Scenario(dataset="tiny", algorithm="limma")

        metrics = python_runner.run_once(
            data_df,
            desc_df,
            scenario,
            keep_result_df=False,
        )

        assert isinstance(metrics.result, dict)
        assert "result_df" not in metrics.result
        assert metrics.result["n_total"] == len(data_df)
        assert metrics.result["n_corrected"] + metrics.result["n_passthrough"] == len(data_df)

    def test_run_scenario_requests_scalar_only_runs(self, monkeypatch) -> None:
        """Timed repetition scheduling must disable result-frame retention.

        Failure condition: timed repetitions call ``run_once`` with
        ``keep_result_df=True`` and keep full DataFrames alive across reps.
        """
        calls: list[bool] = []

        def fake_run_once(*args, keep_result_df: bool = True, **kwargs) -> SingleRunMetrics:
            calls.append(keep_result_df)
            return SingleRunMetrics(
                elapsed_s=0.6,
                cpu_pct=10.0,
                tracemalloc_peak_mb=1.0,
                rss_before_kb=0,
                rss_after_kb=0,
                rss_post_gc_kb=0,
                rss_delta_kb=0,
                rss_delta_post_gc_kb=0,
                result={"n_total": 2, "n_corrected": 2, "n_passthrough": 0},
            )

        monkeypatch.setattr(python_runner, "run_once", fake_run_once)

        runs = python_runner.run_scenario(
            pd.DataFrame(),
            pd.DataFrame(),
            Scenario(dataset="tiny", algorithm="limma"),
            budget_s=1,
            min_reps=2,
            max_reps=2,
        )

        assert len(runs) == 2
        assert calls == [False, False]


class TestBenchmarkHarness:
    def test_harness_uses_warmup_frame_but_timed_counts_only(self, monkeypatch) -> None:
        """Warmup keeps a result frame for validity, timed runs keep only counts.

        Failure condition: the harness requires timed ``result_df`` objects to
        compute feature counts, defeating the retention fix.
        """
        data_df, desc_df = _make_input()
        scenario = Scenario(dataset="tiny", algorithm="limma")
        cfg = _make_config()
        harness = BenchmarkHarness(cfg, [scenario], warmup=True)

        warmup_result = SingleRunMetrics(
            elapsed_s=0.1,
            cpu_pct=10.0,
            tracemalloc_peak_mb=1.0,
            rss_before_kb=0,
            rss_after_kb=0,
            rss_post_gc_kb=0,
            rss_delta_kb=0,
            rss_delta_post_gc_kb=0,
            result={
                "result_df": data_df.copy(),
                "n_total": 2,
                "n_corrected": 2,
                "n_passthrough": 0,
            },
        )
        timed_result = SingleRunMetrics(
            elapsed_s=0.2,
            cpu_pct=20.0,
            tracemalloc_peak_mb=2.0,
            rss_before_kb=100,
            rss_after_kb=140,
            rss_post_gc_kb=110,
            rss_delta_kb=1,
            rss_delta_post_gc_kb=10,
            result={"n_total": 2, "n_corrected": 2, "n_passthrough": 0},
        )

        def fake_run_once(*args, **kwargs) -> SingleRunMetrics:
            return warmup_result

        def fake_run_scenario(*args, **kwargs) -> list[SingleRunMetrics]:
            return [timed_result]

        monkeypatch.setattr("benchmarks.harness.run_once", fake_run_once)
        monkeypatch.setattr("benchmarks.harness.run_scenario", fake_run_scenario)

        _scenario, metrics, validity, _r_entry = harness._run_single_scenario(
            data_df,
            desc_df,
            scenario,
        )

        assert validity is not None
        assert validity.error is None
        assert metrics.features_out == 2
        assert metrics.features_corrected == 2
        assert metrics.features_passthrough == 0

    def test_harness_warns_on_growing_rss(self, monkeypatch, caplog) -> None:
        """The harness should warn when timed post-GC RSS keeps growing.

        Failure condition: repeated-run memory growth is only visible by manual
        inspection of raw RSS values and produces no workflow signal.
        """
        data_df, desc_df = _make_input()
        scenario = Scenario(dataset="tiny", algorithm="limma")
        cfg = _make_config()
        harness = BenchmarkHarness(cfg, [scenario], warmup=True)

        warmup_result = SingleRunMetrics(
            elapsed_s=0.1,
            cpu_pct=10.0,
            tracemalloc_peak_mb=1.0,
            rss_before_kb=0,
            rss_after_kb=0,
            rss_post_gc_kb=0,
            rss_delta_kb=0,
            rss_delta_post_gc_kb=0,
            result={
                "result_df": data_df.copy(),
                "n_total": 2,
                "n_corrected": 2,
                "n_passthrough": 0,
            },
        )

        def fake_run_once(*args, **kwargs) -> SingleRunMetrics:
            return warmup_result

        def fake_run_scenario(*args, **kwargs) -> list[SingleRunMetrics]:
            return [
                SingleRunMetrics(
                    elapsed_s=0.1,
                    cpu_pct=10.0,
                    tracemalloc_peak_mb=1.0,
                    rss_before_kb=100,
                    rss_after_kb=200,
                    rss_post_gc_kb=100,
                    rss_delta_kb=100,
                    rss_delta_post_gc_kb=0,
                    result={"n_total": 2, "n_corrected": 2, "n_passthrough": 0},
                ),
                SingleRunMetrics(
                    elapsed_s=0.1,
                    cpu_pct=10.0,
                    tracemalloc_peak_mb=1.0,
                    rss_before_kb=100,
                    rss_after_kb=2500,
                    rss_post_gc_kb=2400,
                    rss_delta_kb=240,
                    rss_delta_post_gc_kb=2300,
                    result={"n_total": 2, "n_corrected": 2, "n_passthrough": 0},
                ),
                SingleRunMetrics(
                    elapsed_s=0.1,
                    cpu_pct=10.0,
                    tracemalloc_peak_mb=1.0,
                    rss_before_kb=100,
                    rss_after_kb=4300,
                    rss_post_gc_kb=4200,
                    rss_delta_kb=420,
                    rss_delta_post_gc_kb=4100,
                    result={"n_total": 2, "n_corrected": 2, "n_passthrough": 0},
                ),
            ]

        monkeypatch.setattr("benchmarks.harness.run_once", fake_run_once)
        monkeypatch.setattr("benchmarks.harness.run_scenario", fake_run_scenario)

        with caplog.at_level(logging.WARNING):
            harness._run_single_scenario(data_df, desc_df, scenario)

        assert "RSS stability warning" in caplog.text

    def test_run_all_allows_feature_normalization(self, monkeypatch, caplog) -> None:
        """Feature-count validation should allow production IO row cleanup.

        Failure condition: benchmark runs fail when the production IO layer
        legitimately removes duplicate or all-NaN feature rows before timing.
        """
        from benchmarks.datasets import DatasetPaths

        scenario = Scenario(dataset="tiny", algorithm="limma")
        cfg = _make_config()
        harness = BenchmarkHarness(cfg, [scenario], warmup=False)
        data_df, desc_df = _make_input()

        monkeypatch.setattr(
            "benchmarks.harness.resolve_dataset_paths",
            lambda *_args, **_kwargs: DatasetPaths(
                name="tiny",
                input_path=Path(),
                desc_path=Path(),
                features=3,
                samples=4,
                batches=2,
                input_format="tsv",
                r_eligible=False,
            ),
        )
        monkeypatch.setattr(
            BenchmarkHarness,
            "_load_dataset",
            staticmethod(lambda _paths: (data_df, desc_df)),
        )
        monkeypatch.setattr(
            BenchmarkHarness,
            "_run_single_scenario",
            lambda self, *_args, **_kwargs: (scenario, aggregate_metrics([]), None, None),
        )

        with caplog.at_level(logging.INFO):
            results = harness.run_all()

        assert len(results) == 1
        assert "after IO normalization" in caplog.text


class TestBenchmarkReport:
    def test_report_uses_post_gc_python_rss_fields(self) -> None:
        """Report JSON should expose retained post-GC Python RSS metrics.

        Failure condition: the report only keeps the old raw delta and drops the
        normalized RSS values needed for repeated-run interpretation.
        """
        cfg = _make_config()
        scenario = Scenario(dataset="tiny", algorithm="limma")
        metrics = SingleRunMetrics(
            elapsed_s=0.2,
            cpu_pct=20.0,
            tracemalloc_peak_mb=2.0,
            rss_before_kb=100,
            rss_after_kb=140,
            rss_post_gc_kb=110,
            rss_delta_kb=40,
            rss_delta_post_gc_kb=10,
            result={"n_total": 2, "n_corrected": 2, "n_passthrough": 0},
        )

        report = build_report_json(
            [(scenario, aggregate_metrics([metrics]), None, None)],
            cfg,
        )

        py = report["results"][0]["python"]
        assert py["rss_post_gc_kbs"] == [110]
        assert py["rss_before_kb"] == 100
        assert py["rss_after_kb"] == 140
        assert py["rss_post_gc_kb"] == 110
        assert py["rss_delta_kb"] == 40
        assert py["rss_delta_post_gc_kb"] == 10
        assert py["rss_stability"]["status"] == "insufficient-data"


class TestBenchmarkLoading:
    def test_harness_load_dataset_delegates_to_production_io(self, monkeypatch) -> None:
        """Benchmark loading should go through the production IO layer.

        Failure condition: benchmark runs bypass HarmonizePy's real file loaders
        and measure a different parsing path than normal execution.
        """
        from benchmarks.datasets import DatasetPaths
        from benchmarks.harness import BenchmarkHarness

        expected_data, expected_desc = _make_input()

        def fake_load_dataset(paths):
            return expected_data, expected_desc

        monkeypatch.setattr("benchmarks.harness.load_dataset", fake_load_dataset)

        paths = DatasetPaths(
            name="tiny",
            input_path=Path(),
            desc_path=Path(),
            features=2,
            samples=4,
            batches=2,
            input_format="tsv",
            r_eligible=False,
        )

        data_df, desc_df = BenchmarkHarness._load_dataset(paths)

        assert data_df is expected_data
        assert desc_df is expected_desc
