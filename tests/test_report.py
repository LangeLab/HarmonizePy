"""Tests for the per-feature correction report of ``harmonize(..., return_report=True)``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harmonizepy import HarmonizeResult, harmonize
from harmonizepy.__main__ import main

_SAMPLES = [f"s{i}" for i in range(1, 10)]
_BATCHES = np.repeat([1, 2, 3], 3)
_DESC = pd.DataFrame({"ID": _SAMPLES, "sample": range(1, 10), "batch": _BATCHES})


def _known_outcomes() -> tuple[pd.DataFrame, dict[str, str]]:
    """Return a matrix in which every feature's outcome is known, and those outcomes.

    Batches 1, 2, 3 hold three samples each; ComBat mode 1 needs two values per
    batch.
    """
    rng = np.random.default_rng(40)
    shift = np.array([0.0, 1.0, -1.0])[_BATCHES - 1]
    rows: dict[str, np.ndarray] = {}
    expected: dict[str, str] = {}
    for i in range(20):
        rows[f"full_{i}"] = rng.normal(20, 1) + rng.normal(0, 0.3, 9) + shift
        expected[f"full_{i}"] = "corrected"
    rows["constant"] = np.array([5.0] * 3 + [6.0] * 3 + [7.0] * 3)
    expected["constant"] = "raw_zero_variance"
    for i in range(3):
        values = rng.normal(15, 1) + rng.normal(0, 0.3, 9) + shift
        values[_BATCHES == 3] = np.nan
        rows[f"first_two_{i}"] = values
        expected[f"first_two_{i}"] = "corrected"
    alone = rng.normal(12, 1) + rng.normal(0, 0.3, 9) + shift
    alone[_BATCHES == 1] = np.nan
    rows["alone"] = alone
    expected["alone"] = "raw_alone_in_pattern"
    for i in range(2):
        values = np.full(9, np.nan)
        values[_BATCHES == 1] = rng.normal(10, 1, 3)
        rows[f"batch_one_{i}"] = values
        expected[f"batch_one_{i}"] = "raw_single_batch"
    pair_varied = rng.normal(8, 1) + rng.normal(0, 0.3, 9) + shift
    pair_varied[_BATCHES == 2] = np.nan
    pair_constant = np.array([3.0] * 3 + [np.nan] * 3 + [4.0] * 3)
    rows["pair_varied"], rows["pair_constant"] = pair_varied, pair_constant
    expected["pair_varied"] = "raw_too_few_features"
    expected["pair_constant"] = "raw_zero_variance"
    sparse = np.full(9, np.nan)
    sparse[[0, 3, 6]] = [1.0, 2.0, 3.0]
    rows["sparse"] = sparse
    expected["sparse"] = "missing_no_usable_batch"
    data = pd.DataFrame(rows, index=_SAMPLES).T
    return data, expected


class TestReportStatuses:
    """Each status, and the columns that describe it."""

    def test_every_status_matches_the_known_outcome(self) -> None:
        """Each feature gets the status its construction implies.

        Failure condition: a feature left raw is reported as corrected, or the
        reason given is not the one that applies.
        """
        data, expected = _known_outcomes()
        _, report = harmonize(data, _DESC, unique_removal=False, return_report=True)
        assert report["status"].to_dict() == expected

    def test_batches_used_and_group_size(self) -> None:
        """Corrected features list their group's batches; raw features list none.

        Failure condition: the batches or group size do not match the group the
        feature was corrected in.
        """
        data, _ = _known_outcomes()
        _, report = harmonize(data, _DESC, unique_removal=False, return_report=True)
        assert report.loc["full_0", "batches_used"] == "1,2,3"
        assert report.loc["full_0", "group_size"] == 21
        assert report.loc["first_two_0", "batches_used"] == "1,2"
        assert report.loc["first_two_0", "group_size"] == 3
        assert report.loc["constant", "batches_used"] == ""
        assert report.loc["sparse", "group_size"] == 0

    def test_every_feature_appears_once_in_output_order(self) -> None:
        """The report has one row per output feature, in the output's order.

        Failure condition: a feature is missing from the report or listed twice.
        """
        data, _ = _known_outcomes()
        out, report = harmonize(data, _DESC, return_report=True)
        assert list(report.index) == list(out.index)

    def test_matrix_is_identical_with_and_without_the_report(self) -> None:
        """Asking for the report does not change the corrected matrix.

        Failure condition: the report path corrects differently.
        """
        data, _ = _known_outcomes()
        plain = harmonize(data, _DESC, sort="sparsity")
        result = harmonize(data, _DESC, sort="sparsity", return_report=True)
        assert isinstance(result, HarmonizeResult)
        pd.testing.assert_frame_equal(result.data, plain)

    def test_cropped_and_repeats_are_recorded(self) -> None:
        """A feature moved by unique removal is marked cropped, and removed repeats are counted.

        Failure condition: the report hides that a feature lost the values of the
        dropped batch, or that repeats of it were removed.
        """
        rows = [
            [1.0 + i, 1.5 + i, 1.2 + i, 2.1 + i, 2.4 + i, 2.2 + i, np.nan, np.nan, np.nan]
            for i in range(4)
        ]
        rows.append([1.0, 1.4, 1.2, 2.0, 2.6, 2.3, 3.1, 3.5, 3.3])
        data = pd.DataFrame(rows, index=[f"p{i}" for i in range(5)], columns=_SAMPLES)
        data = pd.concat([data, data.iloc[[0, 0]]])
        _, report = harmonize(data, _DESC, return_report=True)
        assert report.loc["p4", "cropped"]
        assert report.loc["p4", "batches_used"] == "1,2"
        assert not report.loc["p0", "cropped"]
        assert report.loc["p0", "repeats_removed"] == 2
        assert report["repeats_removed"].sum() == 2

    def test_repeats_of_a_feature_without_a_name_are_counted(self) -> None:
        """Repeats of a feature whose name is missing are counted like any other.

        Failure condition: the missing name is dropped from the count, so the
        report says 0 although two rows were removed.
        """
        data, _ = _known_outcomes()
        data = data.rename(index={"full_0": np.nan})
        data = pd.concat([data, data.iloc[[0, 0]]])
        _, report = harmonize(data, _DESC, return_report=True)
        assert report["repeats_removed"].sum() == 2

    def test_limma_reports_its_groups(self) -> None:
        """With limma, features in a group spanning several batches are corrected.

        Failure condition: limma groups get a ComBat-only status.
        """
        data, expected = _known_outcomes()
        _, report = harmonize(
            data, _DESC, algorithm="limma", unique_removal=False, return_report=True
        )
        assert report.loc["full_0", "status"] == "corrected"
        assert report.loc["constant", "status"] == "corrected"
        assert report.loc["batch_one_0", "status"] == expected["batch_one_0"]


class TestReportCommandLine:
    """The report written by ``--report``."""

    def _inputs(self, tmp_path: Path) -> tuple[str, str]:
        data, _ = _known_outcomes()
        data_path = tmp_path / "data.tsv"
        data.to_csv(data_path, sep="\t")
        desc_path = tmp_path / "desc.csv"
        _DESC.to_csv(desc_path, index=False)
        return str(data_path), str(desc_path)

    @pytest.mark.parametrize("suffix", ["tsv", "csv"])
    def test_report_file_matches_the_api(self, tmp_path: Path, suffix: str) -> None:
        """``--report`` writes the same table as ``return_report=True``.

        Failure condition: the file and the API disagree on a status or a column.
        """
        data_path, desc_path = self._inputs(tmp_path)
        report_path = tmp_path / f"report.{suffix}"
        main([data_path, desc_path, "--no-log", "-q", "--report", str(report_path)])
        written = pd.read_csv(
            report_path, sep="," if suffix == "csv" else "\t", index_col=0, keep_default_na=False
        )
        _, report = harmonize(data_path, desc_path, return_report=True)
        written["batches_used"] = written["batches_used"].astype(str)
        written.index.name = report.index.name
        pd.testing.assert_frame_equal(written, report, check_dtype=False)

    def test_report_counts_repeats_in_the_data_file(self, tmp_path: Path) -> None:
        """Repeated rows in the data file are counted in the written report.

        Failure condition: the command line removes the repeats before the report
        sees them and reports 0.
        """
        data, _ = _known_outcomes()
        data = pd.concat([data, data.iloc[[0, 0]]])
        data_path = tmp_path / "data.tsv"
        data.to_csv(data_path, sep="\t")
        desc_path = tmp_path / "desc.csv"
        _DESC.to_csv(desc_path, index=False)
        report_path = tmp_path / "report.tsv"
        main([str(data_path), str(desc_path), "--no-log", "-q", "--report", str(report_path)])
        written = pd.read_csv(report_path, sep="\t", index_col=0, keep_default_na=False)
        assert written.loc["full_0", "repeats_removed"] == 2
        assert written["repeats_removed"].sum() == 2

    def test_parquet_report_keeps_column_types(self, tmp_path: Path) -> None:
        """A Parquet report round-trips with its index and column types.

        Failure condition: the boolean or text columns are written as another type.
        """
        pytest.importorskip("pyarrow")
        data_path, desc_path = self._inputs(tmp_path)
        report_path = tmp_path / "report.parquet"
        main([data_path, desc_path, "--no-log", "-q", "--report", str(report_path)])
        _, report = harmonize(data_path, desc_path, return_report=True)
        pd.testing.assert_frame_equal(pd.read_parquet(report_path), report, check_dtype=False)
        assert pd.read_parquet(report_path)["cropped"].dtype == bool

    @pytest.mark.parametrize("target", ["output", "data", "description"])
    def test_report_path_that_would_overwrite_a_file_is_refused(
        self, tmp_path: Path, target: str
    ) -> None:
        """``--report`` naming the output or an input file is a usage error.

        Failure condition: the report overwrites the corrected matrix or an input.
        """
        data_path, desc_path = self._inputs(tmp_path)
        output = str(tmp_path / "out.tsv")
        report = {"output": output, "data": data_path, "description": desc_path}[target]
        before = Path(data_path).read_bytes()
        with pytest.raises(SystemExit) as excinfo:
            main([data_path, desc_path, "--no-log", "-q", "-o", output, "--report", report])
        assert excinfo.value.code == 2
        assert Path(data_path).read_bytes() == before
        assert not Path(output).exists()

    def test_empty_report_path_is_refused(self, tmp_path: Path) -> None:
        """An empty ``--report`` path is a usage error, not a silent no-op.

        Failure condition: the run succeeds and writes no report.
        """
        data_path, desc_path = self._inputs(tmp_path)
        with pytest.raises(SystemExit) as excinfo:
            main([data_path, desc_path, "--no-log", "-q", "--report", ""])
        assert excinfo.value.code == 2

    def test_report_from_config_file(self, tmp_path: Path) -> None:
        """A ``report`` key in the config file writes the report, and a dry run ignores it.

        Failure condition: the key is rejected, or makes the dry run a usage error.
        """
        data_path, desc_path = self._inputs(tmp_path)
        report_path = tmp_path / "report.tsv"
        config = tmp_path / "run.toml"
        config.write_text(f'report = "{report_path.as_posix()}"\n')
        main([data_path, desc_path, "--dry-run", "--config", str(config)])
        assert not report_path.exists()
        main([data_path, desc_path, "--no-log", "-q", "--config", str(config)])
        assert report_path.is_file()

    def test_dry_run_refuses_report(self, tmp_path: Path) -> None:
        """A dry run writes no files, so ``--report`` is a usage error.

        Failure condition: the report option is silently ignored.
        """
        data_path, desc_path = self._inputs(tmp_path)
        with pytest.raises(SystemExit) as excinfo:
            main([data_path, desc_path, "--dry-run", "--report", str(tmp_path / "r.tsv")])
        assert excinfo.value.code == 2
        assert not (tmp_path / "r.tsv").exists()
