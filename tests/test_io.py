"""Direct unit tests for harmonizepy.io.

Covers read_main_data (TSV, CSV, edge cases), read_description, write_output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harmonizepy.io import read_description, read_main_data, write_output

# ---------------------------------------------------------------------------
# read_main_data
# ---------------------------------------------------------------------------


def _write_tsv(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path, sep="\t")
    return path


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path, sep=",")
    return path


class TestReadMainData:
    def test_tsv_file(self, tmp_path: Path) -> None:
        """TSV file is read correctly with feature names as index."""
        expected = pd.DataFrame(
            {"s1": [1.0, 2.0], "s2": [3.0, 4.0]},
            index=["f1", "f2"],
        )
        path = _write_tsv(tmp_path / "data.tsv", expected)
        result = read_main_data(str(path))
        pd.testing.assert_frame_equal(result, expected)

    def test_csv_file(self, tmp_path: Path) -> None:
        """CSV file uses comma separator."""
        expected = pd.DataFrame(
            {"s1": [1.0, 2.0], "s2": [3.0, 4.0]},
            index=["f1", "f2"],
        )
        path = _write_csv(tmp_path / "data.csv", expected)
        result = read_main_data(str(path))
        pd.testing.assert_frame_equal(result, expected)

    def test_fallback_extension_tsv(self, tmp_path: Path) -> None:
        """Unrecognised extension falls back to tab-separated."""
        expected = pd.DataFrame(
            {"s1": [1.0]}, index=["f1"],
        )
        path = _write_tsv(tmp_path / "data.txt", expected)
        result = read_main_data(str(path))
        pd.testing.assert_frame_equal(result, expected)

    def test_drops_all_nan_rows(self, tmp_path: Path) -> None:
        """Rows that are entirely NaN are removed."""
        df = pd.DataFrame(
            {"s1": [1.0, np.nan], "s2": [2.0, np.nan]},
            index=["keep", "drop"],
        )
        path = _write_tsv(tmp_path / "data.tsv", df)
        result = read_main_data(str(path))
        assert "drop" not in result.index
        assert "keep" in result.index

    def test_preserves_all_nan_columns(self, tmp_path: Path) -> None:
        """All-NaN sample columns are kept (structural missingness)."""
        df = pd.DataFrame(
            {"s1": [1.0, 2.0], "s2": [np.nan, np.nan]},
            index=["f1", "f2"],
        )
        path = _write_tsv(tmp_path / "data.tsv", df)
        result = read_main_data(str(path))
        assert "s2" in result.columns
        assert result["s2"].isna().all()

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_main_data("/nonexistent/file.tsv")

    def test_empty_file_after_dropna(self, tmp_path: Path) -> None:
        """File with only all-NaN rows returns empty DataFrame."""
        df = pd.DataFrame({"s1": [np.nan], "s2": [np.nan]}, index=["f1"])
        path = _write_tsv(tmp_path / "empty.tsv", df)
        result = read_main_data(str(path))
        assert len(result) == 0


# ---------------------------------------------------------------------------
# read_description
# ---------------------------------------------------------------------------


class TestReadDescription:
    def test_basic_read(self, tmp_path: Path) -> None:
        desc = pd.DataFrame(
            {"ID": ["s1", "s2"], "sample": [1, 2], "batch": [1, 1]},
        )
        path = tmp_path / "desc.csv"
        desc.to_csv(path, index=False)
        result = read_description(str(path))
        pd.testing.assert_frame_equal(result, desc)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_description("/nonexistent/desc.csv")


# ---------------------------------------------------------------------------
# write_output
# ---------------------------------------------------------------------------


class TestWriteOutput:
    def test_basic_write(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"s1": [1.0, 2.0]}, index=["f1", "f2"])
        path = tmp_path / "out.tsv"
        write_output(df, str(path))
        assert path.exists()
        restored = pd.read_csv(path, sep="\t", index_col=0)
        pd.testing.assert_frame_equal(restored, df)

    def test_write_empty_df(self, tmp_path: Path) -> None:
        df = pd.DataFrame()
        path = tmp_path / "empty.tsv"
        write_output(df, str(path))
        assert path.exists()
        # to_csv on an empty DataFrame produces a header-only file (>0 bytes)
