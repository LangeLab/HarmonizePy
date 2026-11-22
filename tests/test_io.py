"""Direct unit tests for harmonizepy.io.

Covers read_main_data (TSV, CSV, edge cases), read_description, write_output.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harmonizepy import harmonize
from harmonizepy.io import clean_features, read_description, read_main_data, write_output

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
        """TSV file is read correctly with feature names as index.

        Failure condition: the TSV parser fails, the index is not set
        from column 0, or the wrong separator is used.
        """
        expected = pd.DataFrame(
            {"s1": [1.0, 2.0], "s2": [3.0, 4.0]},
            index=["f1", "f2"],
        )
        path = _write_tsv(tmp_path / "data.tsv", expected)
        result = read_main_data(str(path))
        pd.testing.assert_frame_equal(result, expected)

    def test_csv_file(self, tmp_path: Path) -> None:
        """CSV file uses comma separator.

        Failure condition: the wrong delimiter is used or the index
        is not set from the first column.
        """
        expected = pd.DataFrame(
            {"s1": [1.0, 2.0], "s2": [3.0, 4.0]},
            index=["f1", "f2"],
        )
        path = _write_csv(tmp_path / "data.csv", expected)
        result = read_main_data(str(path))
        pd.testing.assert_frame_equal(result, expected)

    def test_fallback_extension_tsv(self, tmp_path: Path) -> None:
        """Unrecognised extension falls back to tab-separated.

        Failure condition: a non-standard extension like ``.txt`` is
        rejected or not treated as tab-separated.
        """
        expected = pd.DataFrame(
            {"s1": [1.0]},
            index=["f1"],
        )
        path = _write_tsv(tmp_path / "data.txt", expected)
        result = read_main_data(str(path))
        pd.testing.assert_frame_equal(result, expected)

    def test_keeps_all_nan_rows(self, tmp_path: Path) -> None:
        """Rows that are entirely missing are kept, as for a DataFrame passed directly.

        Failure condition: a file and the same frame give matrices with different rows.
        """
        df = pd.DataFrame(
            {"s1": [1.0, np.nan], "s2": [2.0, np.nan]},
            index=["kept", "blank"],
        )
        path = _write_tsv(tmp_path / "data.tsv", df)
        result = read_main_data(str(path))
        assert list(result.index) == ["kept", "blank"]
        assert result.loc["blank"].isna().all()

    def test_preserves_all_nan_columns(self, tmp_path: Path) -> None:
        """All-NaN sample columns are kept as structural missingness.

        Failure condition: an all-NaN column is silently dropped,
        which would remove genuine missing data from the output.
        """
        df = pd.DataFrame(
            {"s1": [1.0, 2.0], "s2": [np.nan, np.nan]},
            index=["f1", "f2"],
        )
        path = _write_tsv(tmp_path / "data.tsv", df)
        result = read_main_data(str(path))
        assert "s2" in result.columns
        assert result["s2"].isna().all()

    def test_file_not_found(self) -> None:
        """Reading a non-existent TSV raises FileNotFoundError.

        Failure condition: a missing file returns a partial result
        or raises a different exception type.
        """
        with pytest.raises(FileNotFoundError):
            read_main_data("/nonexistent/file.tsv")

    def test_file_of_blank_rows_keeps_them(self, tmp_path: Path) -> None:
        """A file whose only row is blank is read as one missing row.

        Failure condition: the reader drops the row or fails on a matrix without values.
        """
        df = pd.DataFrame({"s1": [np.nan], "s2": [np.nan]}, index=["f1"])
        path = _write_tsv(tmp_path / "empty.tsv", df)
        result = read_main_data(str(path))
        assert list(result.index) == ["f1"]
        assert result.isna().all().all()


# ---------------------------------------------------------------------------
# read_description
# ---------------------------------------------------------------------------


class TestReadDescription:
    def test_basic_read(self, tmp_path: Path) -> None:
        """Read a CSV description file correctly.

        Failure condition: the CSV is not parsed, columns are misnamed,
        or the row count does not match.
        """
        desc = pd.DataFrame(
            {"ID": ["s1", "s2"], "sample": [1, 2], "batch": [1, 1]},
        )
        path = tmp_path / "desc.csv"
        desc.to_csv(path, index=False)
        result = read_description(str(path))
        pd.testing.assert_frame_equal(result, desc)

    def test_file_not_found(self) -> None:
        """Reading a non-existent CSV raises FileNotFoundError.

        Failure condition: a missing file does not raise or raises
        a different exception type.
        """
        with pytest.raises(FileNotFoundError):
            read_description("/nonexistent/desc.csv")


# ---------------------------------------------------------------------------
# write_output
# ---------------------------------------------------------------------------


class TestWriteOutput:
    def test_basic_write(self, tmp_path: Path) -> None:
        """Write a TSV and verify it can be read back.

        Failure condition: the file is not created, or the round-tripped
        data does not match the original.
        """
        df = pd.DataFrame({"s1": [1.0, 2.0]}, index=["f1", "f2"])
        path = tmp_path / "out.tsv"
        write_output(df, str(path))
        assert path.exists()
        restored = pd.read_csv(path, sep="\t", index_col=0)
        pd.testing.assert_frame_equal(restored, df)

    def test_write_empty_df(self, tmp_path: Path) -> None:
        """Write an empty DataFrame results in a header-only file.

        Failure condition: the function crashes on an empty DataFrame
        or no file is produced.
        """
        df = pd.DataFrame()
        path = tmp_path / "empty.tsv"
        write_output(df, str(path))
        assert path.exists()
        # to_csv on an empty DataFrame produces a header-only file (>0 bytes)


class TestRepeatedFeatures:
    """Repeated feature names, removed only when their values also match."""

    def test_identical_values_different_names_both_kept(self, tmp_path: Path) -> None:
        """Two features with identical values and different names are both kept.

        Failure condition: a feature is removed because its values match another's.
        """
        path = tmp_path / "m.tsv"
        path.write_text("id\ts1\ts2\ts3\np1\t1\t2\t3\np2\t1\t2\t3\np3\t4\t5\t6\n")
        assert list(read_main_data(str(path)).index) == ["p1", "p2", "p3"]

    def test_same_name_and_values_removed_and_logged(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A repeated feature with identical values is removed, and the removal is logged.

        Failure condition: the repeat stays in the matrix, or the removal is silent.
        """
        # The command line turns off propagation on the package logger, and caplog
        # only sees records that propagate to the root logger.
        monkeypatch.setattr(logging.getLogger("harmonizepy"), "propagate", True)
        path = tmp_path / "m.tsv"
        path.write_text("id\ts1\ts2\ts3\np1\t1\t2\t3\np1\t1\t2\t3\np3\t4\t5\t6\n")
        with caplog.at_level(logging.INFO, logger="harmonizepy.io"):
            result = read_main_data(str(path))
        assert list(result.index) == ["p1", "p3"]
        assert "Removed 1 repeated feature(s) with identical values: p1" in caplog.text

    def test_same_name_different_values_raises(self, tmp_path: Path) -> None:
        """A name repeated with different values is an error.

        Failure condition: the matrix is read and one of the two rows is silently kept.
        """
        path = tmp_path / "m.tsv"
        path.write_text("id\ts1\ts2\ts3\np1\t1\t2\t3\np1\t1\t2\t9\n")
        with pytest.raises(ValueError, match="duplicate feature names with different values"):
            read_main_data(str(path))

    def test_missing_cells_in_the_same_places_count_as_identical(self) -> None:
        """Repeats with missing cells in the same places are identical; elsewhere they differ.

        Failure condition: missing cells make identical repeats look different, or
        a missing cell matches a value.
        """
        same = pd.DataFrame([[1.0, np.nan], [1.0, np.nan]], index=["a", "a"])
        assert len(clean_features(same)) == 1
        differs = pd.DataFrame([[1.0, np.nan], [1.0, 2.0]], index=["a", "a"])
        with pytest.raises(ValueError, match="'a'"):
            clean_features(differs)

    def test_harmonize_removes_identical_repeat_from_a_dataframe(self) -> None:
        """``harmonize()`` applies the same rule to a DataFrame passed directly.

        Failure condition: the repeat raises, or reaches the engines as two features.
        """
        rng = np.random.default_rng(4)
        samples = [f"s{i}" for i in range(1, 7)]
        data = pd.DataFrame(
            rng.normal(10, 1, (5, 6)), index=[f"p{i}" for i in range(5)], columns=samples
        )
        repeated = pd.concat([data, data.iloc[[0]]])
        desc = pd.DataFrame({"ID": samples, "sample": range(1, 7), "batch": [1, 1, 1, 2, 2, 2]})
        with_repeat = harmonize(repeated, desc, algorithm="limma")
        pd.testing.assert_frame_equal(with_repeat, harmonize(data, desc, algorithm="limma"))

    def test_non_numeric_columns_use_the_same_rule(self) -> None:
        """With a text column, identical repeats are removed and differing ones raise.

        Failure condition: the check for frames that are not all numeric drops a
        differing row or keeps an identical repeat.
        """
        same = pd.DataFrame({"s1": [1.0, 1.0, 2.0], "note": ["x", "x", "y"]}, index=["a", "a", "b"])
        assert list(clean_features(same).index) == ["a", "b"]
        differs = pd.DataFrame({"s1": [1.0, 1.0], "note": ["x", "z"]}, index=["a", "a"])
        with pytest.raises(ValueError, match="'a'"):
            clean_features(differs)

    def test_name_repeated_once_blank_raises(self, tmp_path: Path) -> None:
        """A name repeated with values and then with only blanks is a conflict.

        Failure condition: the blank row is dropped first and the conflict is never
        reported.
        """
        path = tmp_path / "m.tsv"
        path.write_text("id\ts1\ts2\ts3\np1\t1\t2\t3\np1\t\t\t\np2\t4\t5\t6\n")
        with pytest.raises(ValueError, match="'p1'"):
            read_main_data(str(path))

    def test_large_number_of_repeated_names_is_fast(self) -> None:
        """Checking 20,000 repeated names takes well under a second.

        Failure condition: the check scans the index once per repeated name, which
        takes tens of seconds at this size.
        """
        values = np.random.default_rng(5).normal(size=(20_000, 4))
        frame = pd.DataFrame(
            np.vstack([values, values]), index=[f"p{i}" for i in range(20_000)] * 2
        )
        start = time.perf_counter()
        result = clean_features(frame)
        # A generous bound, so that a loaded machine does not fail the test.
        assert time.perf_counter() - start < 5.0
        assert len(result) == 20_000


class TestIntegerColumns:
    """Columns of whole numbers in the data matrix and the description."""

    @pytest.mark.parametrize("ext", ["tsv", "csv", "parquet"])
    def test_integer_column_with_blank_loads_as_missing(self, tmp_path: Path, ext: str) -> None:
        """A whole-number column with a blank cell loads, and the blank is missing.

        Failure condition: the reader raises on the blank cell.
        """
        frame = pd.DataFrame(
            {"s1": pd.array([1, None, 6], dtype="Int64"), "s2": [2, 4, 7], "s3": [3, 5, 8]},
            index=pd.Index(["p1", "p2", "p3"], name="id"),
        )
        path = tmp_path / f"m.{ext}"
        if ext == "parquet":
            pytest.importorskip("pyarrow")
            frame.to_parquet(path)
        else:
            frame.to_csv(path, sep="\t" if ext == "tsv" else ",")
        result = read_main_data(str(path))
        assert result["s1"].dtype == np.float64
        assert np.isnan(result.loc["p2", "s1"])
        assert result["s1"].tolist()[::2] == [1.0, 6.0]

    def test_wide_and_narrow_readers_agree(self, tmp_path: Path) -> None:
        """Both readers give the same frame for whole-number and blank columns.

        Failure condition: the dtype of a column depends on how many columns the
        file has, or a column of blanks is read as text by one reader.
        """

        def write(n_cols: int, path: Path) -> pd.DataFrame:
            cols = [f"s{i}" for i in range(n_cols)]
            rows = {
                "p1": ["", ""] + ["1"] * (n_cols - 2),
                "p2": ["2", ""] + ["2"] * (n_cols - 2),
            }
            lines = ["id\t" + "\t".join(cols)]
            lines += [name + "\t" + "\t".join(vals) for name, vals in rows.items()]
            path.write_text("\n".join(lines) + "\n")
            return read_main_data(str(path))

        wide = write(501, tmp_path / "wide.tsv")
        narrow = write(4, tmp_path / "narrow.tsv")
        for col in ["s0", "s1", "s2"]:
            assert wide[col].dtype == narrow[col].dtype, col
        assert narrow["s0"].dtype == np.float64
        assert narrow["s1"].dtype == np.float64
        assert narrow["s2"].dtype == np.int64

    def test_integer_column_without_blank_stays_integer(self, tmp_path: Path) -> None:
        """A whole-number column without a blank keeps an integer dtype.

        Failure condition: integer columns become floating point without a blank,
        so the dtype depends on the reader.
        """
        path = tmp_path / "m.tsv"
        path.write_text("id\ts1\ts2\np1\t1\t2\np2\t3\t4\n")
        assert read_main_data(str(path))["s1"].dtype == np.int64

    def test_description_keeps_integer_columns(self, tmp_path: Path) -> None:
        """The batch description keeps its integer columns.

        Failure condition: batch labels are read as floating point.
        """
        path = tmp_path / "d.csv"
        path.write_text("ID,sample,batch\ns1,1,1\ns2,2,2\n")
        desc = read_description(str(path))
        assert desc["batch"].dtype == np.int64
        assert desc["sample"].dtype == np.int64


@pytest.mark.parametrize("ext, sep", [("tsv", "\t"), ("csv", ",")])
def test_repeated_sample_name_in_file_raises(tmp_path: Path, ext: str, sep: str) -> None:
    """A sample name repeated in a file header is rejected by name.

    Failure condition: the parser renames the second column to ``s1.1`` and the
    run treats the two columns as different samples.
    """
    path = tmp_path / f"m.{ext}"
    header = sep.join(["id", "s1", "s1", "s2"])
    path.write_text(f"{header}\n" + sep.join(["p1", "1", "2", "3"]) + "\n")
    with pytest.raises(ValueError, match=r"duplicate sample names: \['s1'\]"):
        read_main_data(str(path))


@pytest.mark.parametrize(
    "name, fmt",
    [
        ("a.tsv", "tsv"),
        ("a.TXT", "tsv"),
        ("a.csv", "csv"),
        ("a.CSV", "csv"),
        ("a.pq", "parquet"),
        ("a.parquet", "parquet"),
        ("a.dat", "tsv"),
        ("a", "tsv"),
    ],
)
def test_format_from_suffix(name: str, fmt: str) -> None:
    """The output format follows the suffix, case-insensitively, and is TSV otherwise.

    Failure condition: the CLI and ``write_output`` disagree on a suffix's format.
    """
    from harmonizepy.io import format_from_suffix

    assert format_from_suffix(name) == fmt


@pytest.mark.parametrize("fmt, sep", [("tsv", "\t"), ("csv", ",")])
def test_write_frame_writes_the_named_format(tmp_path: Path, fmt: str, sep: str) -> None:
    """``write_frame`` writes the requested format whatever the suffix.

    Failure condition: the suffix overrides the requested format.
    """
    from harmonizepy.io import write_frame

    path = tmp_path / "out.dat"
    write_frame(pd.DataFrame({"s1": [1.0], "s2": [2.0]}, index=["p1"]), str(path), fmt)
    assert path.read_text().splitlines()[0].count(sep) == 2


def test_reading_a_delimited_file_starts_no_threads(tmp_path: Path) -> None:
    """Reading a TSV and a CSV runs on the calling thread and leaves no new thread behind.

    Failure condition: the reader starts a thread pool, as pyarrow's CSV reader does.
    """
    tasks = Path("/proc/self/task")
    if not tasks.is_dir():
        pytest.skip("thread listing needs /proc")
    data_path = tmp_path / "m.tsv"
    data_path.write_text("id\ts1\ts2\ts3\np1\t1\t2\t3\np2\t4\t5\t6\n")
    desc_path = tmp_path / "d.csv"
    desc_path.write_text("ID,sample,batch\ns1,1,1\ns2,2,1\ns3,3,2\n")
    before = set(tasks.iterdir())
    read_main_data(str(data_path))
    read_description(str(desc_path))
    assert set(tasks.iterdir()) <= before


class TestNamesReadAsWritten:
    """Feature names and description IDs are text, read exactly as in the file."""

    def test_feature_names_that_look_missing_are_kept(self, tmp_path: Path) -> None:
        """Rows named NA and null keep their names; NA in a value cell is still missing.

        Failure condition: the reader applies its missing-value markers to the
        name column, so both rows are named NaN and one is removed as a repeat.
        """
        path = tmp_path / "data.tsv"
        path.write_text("feature\ts1\ts2\nNA\t1\t2\nnull\t1\t2\nf3\t3\tNA\n")

        result = read_main_data(str(path))

        assert result.index.tolist() == ["NA", "null", "f3"]
        assert np.isnan(result.loc["f3", "s2"])

    def test_numeric_looking_feature_names_stay_text(self, tmp_path: Path) -> None:
        """Feature names such as 001 and 002 are not turned into numbers.

        Failure condition: the name column is type-inferred, so 001 becomes 1.
        """
        path = tmp_path / "data.tsv"
        path.write_text("feature\ts1\ts2\n001\t1\t2\n002\t3\t4\n")

        result = read_main_data(str(path))

        assert result.index.tolist() == ["001", "002"]

    def test_numeric_looking_sample_ids_match_the_header(self, tmp_path: Path) -> None:
        """Description IDs 01 to 04 match data columns 01 to 04 and the run succeeds.

        Failure condition: the ID column is type-inferred, so 01 becomes 1 and
        the run is refused with a mismatch the user did not make.
        """
        ids = ["01", "02", "03", "04"]
        data = pd.DataFrame(
            np.random.default_rng(3).normal(10, 1, (6, 4)),
            index=[f"f{i}" for i in range(6)],
            columns=ids,
        )
        data_path = _write_tsv(tmp_path / "data.tsv", data)
        desc_path = tmp_path / "desc.csv"
        desc_path.write_text("ID,sample,batch\n01,1,1\n02,2,1\n03,3,2\n04,4,2\n")

        assert read_description(str(desc_path))["ID"].tolist() == ids
        result = harmonize(str(data_path), str(desc_path), algorithm="limma")
        assert result.columns.tolist() == ids


