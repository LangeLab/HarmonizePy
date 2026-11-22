"""Tests for the harmonizepy CLI (__main__.py).

Calls main(argv=[...]) directly so no installation or subprocess overhead.
All tests that write files use tmp_path for isolation.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harmonizepy.__main__ import (
    _infer_format,
    _resolve_output_path,
    main,
)
from harmonizepy.core import harmonize

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent / "fixtures"
DATA = str(_FIXTURES / "small_input.tsv")
BATCH = str(_FIXTURES / "small_batch.csv")

# Medium dataset for block/sort tests (has enough batches)
MED_DATA = str(_FIXTURES / "medium_input.tsv")
MED_BATCH = str(_FIXTURES / "medium_batch.csv")

_has_small = (Path(_FIXTURES) / "small_input.tsv").exists()
_has_medium = (Path(_FIXTURES) / "medium_input.tsv").exists()


# ===========================================================================
# Helpers
# ===========================================================================


class TestResolveOutputPath:
    def test_explicit_output_returned_unchanged(self) -> None:
        """Explicit output path is returned as-is.

        Failure condition: the explicit path is modified or ignored.
        """
        assert _resolve_output_path("data.tsv", "/out/result.tsv") == "/out/result.tsv"

    def test_default_placed_next_to_input(self) -> None:
        """Default output path lives next to the input file with ``_corrected`` suffix.

        Failure condition: the default path uses a different directory or suffix.
        """
        result = _resolve_output_path("/data/my_file.tsv", None)
        assert result == "/data/my_file_corrected.tsv"

    def test_default_stem_preservation(self) -> None:
        """Input filename stem is preserved in the default output path.

        Failure condition: the stem is modified or dropped.
        """
        result = _resolve_output_path("proteins.tsv", None)
        assert result == "proteins_corrected.tsv"

    def test_default_parquet_when_output_format_requests_it(self) -> None:
        """Default path switches to .parquet when parquet output is requested.

        Failure condition: the default path ignores explicit parquet output.
        """
        result = _resolve_output_path("/data/my_file.tsv", None, "parquet")
        assert result == "/data/my_file_corrected.parquet"

    def test_windows_path_style_is_preserved(self) -> None:
        """Windows drive and separator syntax remains native in the result.

        Failure condition: a Windows input path is converted to POSIX syntax.
        """
        result = _resolve_output_path(r"C:\data\my_file.tsv", None)
        assert result == r"C:\data\my_file_corrected.tsv"


class TestInferFormat:
    def test_explicit_flag_wins(self) -> None:
        """Explicit --output-format flag overrides the file extension.

        Failure condition: extension inference takes priority over
        the explicit flag.
        """
        assert _infer_format("result.tsv", "csv") == "csv"

    def test_tsv_extension(self) -> None:
        """.tsv extension maps to ``"tsv"`` format.

        Failure condition: .tsv maps to a different format.
        """
        assert _infer_format("result.tsv", None) == "tsv"

    def test_csv_extension(self) -> None:
        """.csv extension maps to ``"csv"`` format.

        Failure condition: .csv maps to a different format.
        """
        assert _infer_format("result.csv", None) == "csv"

    def test_parquet_extension(self) -> None:
        """.parquet extension maps to ``"parquet"`` format.

        Failure condition: .parquet maps to a different format.
        """
        assert _infer_format("result.parquet", None) == "parquet"

    def test_pq_extension(self) -> None:
        """.pq extension maps to ``"parquet"`` format.

        Failure condition: .pq maps to a different format.
        """
        assert _infer_format("result.pq", None) == "parquet"

    def test_unknown_extension_falls_back_to_tsv(self) -> None:
        """Unrecognised extension falls back to ``"tsv"`` format.

        Failure condition: an unknown extension raises or maps to
        a different format.
        """
        assert _infer_format("result.dat", None) == "tsv"

    def test_txt_extension_is_tsv(self) -> None:
        """.txt extension maps to ``"tsv"`` format (tab-separated default).

        Failure condition: .txt maps to a different format.
        """
        assert _infer_format("result.txt", None) == "tsv"


# ===========================================================================
# Minimal happy-path
# ===========================================================================


@pytest.mark.skipif(not _has_small, reason="R fixtures not generated")
class TestCLIMinimal:
    def test_creates_output_file(self, tmp_path: Path) -> None:
        """CLI must create a valid output file.

        Failure condition: the output file is not created.
        """
        out = str(tmp_path / "result.tsv")
        main([DATA, BATCH, "-o", out])
        assert Path(out).exists()

    def test_output_is_nonempty_dataframe(self, tmp_path: Path) -> None:
        """CLI output must be a non-empty DataFrame.

        Failure condition: the output file is empty or unreadable.
        """
        out = str(tmp_path / "result.tsv")
        main([DATA, BATCH, "-o", out])
        df = pd.read_csv(out, sep="\t", index_col=0)
        assert not df.empty

    def test_output_matches_harmonize_api(self, tmp_path: Path) -> None:
        """CLI result must be numerically identical to calling harmonize().

        Failure condition: the CLI code path diverges from the API
        code path.

        Tolerances: atol=1e-10 for identical float64 pipeline output.
        """
        out = str(tmp_path / "result.tsv")
        main([DATA, BATCH, "-o", out])
        cli_result = pd.read_csv(out, sep="\t", index_col=0)
        api_result = harmonize(DATA, BATCH)
        pd.testing.assert_frame_equal(cli_result, api_result, atol=1e-10)

    def test_default_output_path_next_to_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With ``-o`` omitted, output lands next to the input file.

        Failure condition: the output is written to the current directory
        instead of next to the input.
        """
        import shutil

        local_data = str(tmp_path / "small_input.tsv")
        local_batch = str(tmp_path / "small_batch.csv")
        shutil.copy(DATA, local_data)
        shutil.copy(BATCH, local_batch)

        main([local_data, local_batch])
        expected = tmp_path / "small_input_corrected.tsv"
        assert expected.exists()

    def test_python_m_invocation(self, tmp_path: Path) -> None:
        """__main__.py must work when invoked as ``python -m harmonizepy``.

        Failure condition: the module entry point fails when called
        via ``main()`` (which mirrors ``-m`` invocation).
        """
        out = str(tmp_path / "result.tsv")
        main([DATA, BATCH, "-o", out])
        assert Path(out).is_file()


# ===========================================================================
# Algorithm flags
# ===========================================================================


@pytest.mark.skipif(not _has_small, reason="R fixtures not generated")
class TestCLIAlgorithmFlags:
    def test_algorithm_combat_default(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out])
        assert Path(out).exists()

    def test_algorithm_limma(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--algorithm", "limma"])
        assert Path(out).exists()

    def test_combat_mode_1(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--combat-mode", "1"])
        assert Path(out).exists()

    def test_combat_mode_2(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--combat-mode", "2"])
        assert Path(out).exists()

    def test_combat_mode_3(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--combat-mode", "3"])
        assert Path(out).exists()

    def test_combat_mode_4(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--combat-mode", "4"])
        assert Path(out).exists()

    def test_needed_values_explicit(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--needed-values", "2"])
        assert Path(out).exists()

    def test_limma_result_matches_api(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--algorithm", "limma"])
        cli = pd.read_csv(out, sep="\t", index_col=0)
        api = harmonize(DATA, BATCH, algorithm="limma")
        pd.testing.assert_frame_equal(cli, api, atol=1e-10)


# ===========================================================================
# Sorting and blocking
# ===========================================================================


@pytest.mark.skipif(not _has_medium, reason="R medium fixtures not generated")
class TestCLISortBlock:
    def test_sort_sparsity(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([MED_DATA, MED_BATCH, "-o", out, "--sort", "sparsity"])
        assert Path(out).exists()

    def test_sort_jaccard(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([MED_DATA, MED_BATCH, "-o", out, "--sort", "jaccard"])
        assert Path(out).exists()

    def test_sort_seriation(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([MED_DATA, MED_BATCH, "-o", out, "--sort", "seriation"])
        assert Path(out).exists()

    def test_sort_and_block(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([MED_DATA, MED_BATCH, "-o", out, "--sort", "sparsity", "--block", "2"])
        assert Path(out).exists()

    def test_sort_block_matches_api(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([MED_DATA, MED_BATCH, "-o", out, "--sort", "sparsity", "--block", "2"])
        cli = pd.read_csv(out, sep="\t", index_col=0)
        api = harmonize(MED_DATA, MED_BATCH, sort="sparsity", block=2)
        pd.testing.assert_frame_equal(cli, api, atol=1e-10)

    def test_block_without_sort(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([MED_DATA, MED_BATCH, "-o", out, "--block", "2"])
        assert Path(out).exists()


# ===========================================================================
# Unique removal
# ===========================================================================


@pytest.mark.skipif(not _has_small, reason="R fixtures not generated")
class TestCLIUniqueRemoval:
    def test_unique_removal_enabled_by_default(self, tmp_path: Path) -> None:
        out_default = str(tmp_path / "default.tsv")
        out_explicit = str(tmp_path / "explicit.tsv")
        main([DATA, BATCH, "-o", out_default])
        main([DATA, BATCH, "-o", out_explicit, "--unique-removal"])
        df_default = pd.read_csv(out_default, sep="\t", index_col=0)
        df_explicit = pd.read_csv(out_explicit, sep="\t", index_col=0)
        pd.testing.assert_frame_equal(df_default, df_explicit, atol=1e-10)

    def test_no_unique_removal(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--no-unique-removal"])
        assert Path(out).exists()

    def test_no_unique_removal_matches_api(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--no-unique-removal"])
        cli = pd.read_csv(out, sep="\t", index_col=0)
        api = harmonize(DATA, BATCH, unique_removal=False)
        pd.testing.assert_frame_equal(cli, api, atol=1e-10)


# ===========================================================================
# Output formats
# ===========================================================================


@pytest.mark.skipif(not _has_small, reason="R fixtures not generated")
class TestCLIOutputFormats:
    def test_tsv_explicit_flag(self, tmp_path: Path) -> None:
        out = str(tmp_path / "result.tsv")
        main([DATA, BATCH, "-o", out, "--output-format", "tsv"])
        df = pd.read_csv(out, sep="\t", index_col=0)
        assert not df.empty

    def test_tsv_inferred_from_extension(self, tmp_path: Path) -> None:
        out = str(tmp_path / "result.tsv")
        main([DATA, BATCH, "-o", out])
        df = pd.read_csv(out, sep="\t", index_col=0)
        assert not df.empty

    def test_csv_inferred_from_extension(self, tmp_path: Path) -> None:
        out = str(tmp_path / "result.csv")
        main([DATA, BATCH, "-o", out])
        df = pd.read_csv(out, index_col=0)
        assert not df.empty

    def test_csv_explicit_flag(self, tmp_path: Path) -> None:
        out = str(tmp_path / "result.csv")
        main([DATA, BATCH, "-o", out, "--output-format", "csv"])
        df = pd.read_csv(out, index_col=0)
        assert not df.empty

    def test_format_flag_overrides_extension(self, tmp_path: Path) -> None:
        """--output-format csv on a .tsv path: file is valid CSV, not TSV."""
        out = str(tmp_path / "result.tsv")
        main([DATA, BATCH, "-o", out, "--output-format", "csv"])
        # A CSV file written to .tsv path: readable as CSV (comma-separated)
        df = pd.read_csv(out, index_col=0)
        assert not df.empty

    def test_parquet_inferred_from_extension(self, tmp_path: Path) -> None:
        pytest.importorskip("pyarrow")
        out = str(tmp_path / "result.parquet")
        main([DATA, BATCH, "-o", out])
        df = pd.read_parquet(out)
        assert not df.empty

    def test_parquet_explicit_flag(self, tmp_path: Path) -> None:
        pytest.importorskip("pyarrow")
        out = str(tmp_path / "result.out")
        main([DATA, BATCH, "-o", out, "--output-format", "parquet"])
        df = pd.read_parquet(out)
        assert not df.empty

    def test_parquet_default_path_when_format_explicit(self, tmp_path: Path) -> None:
        pytest.importorskip("pyarrow")

        import shutil

        local_data = str(tmp_path / "small_input.tsv")
        local_batch = str(tmp_path / "small_batch.csv")
        shutil.copy(DATA, local_data)
        shutil.copy(BATCH, local_batch)

        main([local_data, local_batch, "--output-format", "parquet"])
        df = pd.read_parquet(tmp_path / "small_input_corrected.parquet")
        assert not df.empty

    def test_csv_content_matches_api(self, tmp_path: Path) -> None:
        out = str(tmp_path / "result.csv")
        main([DATA, BATCH, "-o", out])
        cli = pd.read_csv(out, index_col=0)
        api = harmonize(DATA, BATCH)
        pd.testing.assert_frame_equal(cli, api, atol=1e-10)


# ===========================================================================
# Dry-run
# ===========================================================================


@pytest.mark.skipif(not _has_small, reason="R fixtures not generated")
class TestCLIDryRun:
    def test_dry_run_no_output_file_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        main([DATA, BATCH, "--dry-run"])
        # Only the dry-run; no corrected file should be written
        assert not (tmp_path / "small_input_corrected.tsv").exists()

    def test_dry_run_prints_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        main([DATA, BATCH, "--dry-run"])
        out = capsys.readouterr().out
        assert "dry run" in out.lower()

    def test_dry_run_shows_dimensions(self, capsys: pytest.CaptureFixture[str]) -> None:
        main([DATA, BATCH, "--dry-run"])
        out = capsys.readouterr().out
        assert "Features" in out
        assert "Samples" in out
        assert "Batches" in out
        assert "Sub-matrices" in out

    def test_dry_run_shows_algorithm(self, capsys: pytest.CaptureFixture[str]) -> None:
        main([DATA, BATCH, "--dry-run", "--algorithm", "limma"])
        out = capsys.readouterr().out
        assert "limma" in out

    def test_dry_run_shows_sort_strategy(self, capsys: pytest.CaptureFixture[str]) -> None:
        main([MED_DATA, MED_BATCH, "--dry-run", "--sort", "sparsity"])
        out = capsys.readouterr().out
        assert "sparsity" in out

    def test_dry_run_exits_zero(self) -> None:
        """dry-run returns normally (exit code 0 implicitly)."""
        # main() returns None on success; no SystemExit is raised
        main([DATA, BATCH, "--dry-run"])

    def test_dry_run_invalid_block_exits_nonzero(self) -> None:
        """--dry-run with --block 1 (< 2) should exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            main([DATA, BATCH, "--dry-run", "--block", "1"])
        assert exc_info.value.code == 1

    def test_dry_run_with_explicit_output_flag(self, tmp_path: Path) -> None:
        out = str(tmp_path / "will_not_be_created.tsv")
        main([DATA, BATCH, "--dry-run", "-o", out])
        assert not Path(out).exists()


# ===========================================================================
# Run summary JSON
# ===========================================================================


@pytest.mark.skipif(not _has_small, reason="R fixtures not generated")
class TestCLISummary:
    def test_summary_file_created(self, tmp_path: Path) -> None:
        out = str(tmp_path / "result.tsv")
        summary = str(tmp_path / "summary.json")
        main([DATA, BATCH, "-o", out, "--summary", summary])
        assert Path(summary).exists()

    def test_summary_is_valid_json(self, tmp_path: Path) -> None:
        out = str(tmp_path / "result.tsv")
        summary = str(tmp_path / "summary.json")
        main([DATA, BATCH, "-o", out, "--summary", summary])
        with Path(summary).open() as fh:
            data = json.load(fh)
        assert isinstance(data, dict)

    def test_summary_contains_required_keys(self, tmp_path: Path) -> None:
        out = str(tmp_path / "result.tsv")
        summary = str(tmp_path / "summary.json")
        main([DATA, BATCH, "-o", out, "--summary", summary])
        with Path(summary).open() as fh:
            s = json.load(fh)
        required = {
            "harmonizepy_version",
            "algorithm",
            "combat_mode",
            "needed_values",
            "sort_strategy",
            "block_size",
            "unique_removal",
            "n_features_input",
            "n_features_output",
            "n_samples",
            "n_batches",
            "data_file",
            "description_file",
            "output_file",
            "output_format",
        }
        assert required <= s.keys()

    def test_summary_algorithm_recorded(self, tmp_path: Path) -> None:
        out = str(tmp_path / "result.tsv")
        summary = str(tmp_path / "summary.json")
        main([DATA, BATCH, "-o", out, "--summary", summary, "--algorithm", "limma"])
        with Path(summary).open() as fh:
            s = json.load(fh)
        assert s["algorithm"] == "limma"

    def test_summary_combat_mode_recorded(self, tmp_path: Path) -> None:
        out = str(tmp_path / "result.tsv")
        summary = str(tmp_path / "summary.json")
        main([DATA, BATCH, "-o", out, "--summary", summary, "--combat-mode", "3"])
        with Path(summary).open() as fh:
            s = json.load(fh)
        assert s["combat_mode"] == 3

    def test_summary_dimensions_correct(self, tmp_path: Path) -> None:
        out = str(tmp_path / "result.tsv")
        summary = str(tmp_path / "summary.json")
        main([DATA, BATCH, "-o", out, "--summary", summary])
        with Path(summary).open() as fh:
            s = json.load(fh)
        # n_features_output <= n_features_input (correction may drop rows)
        assert s["n_features_output"] <= s["n_features_input"]
        assert s["n_samples"] > 0
        assert s["n_batches"] >= 2

    def test_summary_version_is_string(self, tmp_path: Path) -> None:
        out = str(tmp_path / "result.tsv")
        summary = str(tmp_path / "summary.json")
        main([DATA, BATCH, "-o", out, "--summary", summary])
        with Path(summary).open() as fh:
            s = json.load(fh)
        assert isinstance(s["harmonizepy_version"], str)
        assert re.search(r"\d+\.\d+\.\d+", s["harmonizepy_version"])


# ===========================================================================
# Verbosity flags
# ===========================================================================


@pytest.mark.skipif(not _has_small, reason="R fixtures not generated")
class TestCLIVerbosity:
    def test_verbose_flag_does_not_crash(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--verbose"])
        assert Path(out).exists()

    def test_short_v_flag(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "-v"])
        assert Path(out).exists()

    def test_quiet_flag_does_not_crash(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--quiet"])
        assert Path(out).exists()

    def test_short_q_flag(self, tmp_path: Path) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "-q"])
        assert Path(out).exists()

    def test_verbose_and_quiet_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([DATA, BATCH, "--verbose", "--quiet"])
        assert exc_info.value.code != 0


# ===========================================================================
# Error handling
# ===========================================================================


@pytest.mark.skipif(not _has_small, reason="R fixtures not generated")
class TestCLIErrors:
    def test_missing_all_positional_args(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_missing_description_arg(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([DATA])
        assert exc_info.value.code != 0

    def test_data_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["nonexistent_data.tsv", BATCH, "-o", str(tmp_path / "r.tsv")])
        assert exc_info.value.code != 0

    def test_description_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([DATA, "nonexistent_batch.csv", "-o", str(tmp_path / "r.tsv")])
        assert exc_info.value.code != 0

    def test_invalid_algorithm(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([DATA, BATCH, "--algorithm", "InvalidAlgo"])
        assert exc_info.value.code != 0

    def test_invalid_combat_mode(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([DATA, BATCH, "--combat-mode", "5"])
        assert exc_info.value.code != 0

    def test_invalid_sort_strategy(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([DATA, BATCH, "--sort", "random"])
        assert exc_info.value.code != 0

    def test_block_lt_2_exits_nonzero(self, tmp_path: Path) -> None:
        """block=1 fails validation inside harmonize gives exit 1."""
        with pytest.raises(SystemExit) as exc_info:
            main([DATA, BATCH, "-o", str(tmp_path / "r.tsv"), "--block", "1"])
        assert exc_info.value.code == 1

    def test_invalid_output_format(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([DATA, BATCH, "--output-format", "xlsx"])
        assert exc_info.value.code != 0


# ===========================================================================
# --help and --version
# ===========================================================================


class TestCLIHelpVersion:
    def test_help_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_help_lists_core_flags(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main(["--help"])
        out = capsys.readouterr().out
        for flag in (
            "--algorithm",
            "--combat-mode",
            "--sort",
            "--block",
            "--unique-removal",
            "--dry-run",
            "--summary",
            "--config",
            "--json",
            "--verbose",
            "--quiet",
        ):
            assert flag in out, f"--help output missing flag: {flag}"

    def test_version_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_version_contains_version_number(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main(["--version"])
        combined = capsys.readouterr()
        output = combined.out + combined.err
        assert re.search(r"\d+\.\d+\.\d+", output), f"No version number in: {output!r}"

    def test_version_contains_package_name(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main(["--version"])
        combined = capsys.readouterr()
        output = combined.out + combined.err
        assert "harmonizepy" in output.lower()


# ===========================================================================
# --config flag
# ===========================================================================


@pytest.mark.skipif(not _has_small, reason="R fixtures not generated")
class TestCLIConfig:
    """Tests for config-file loading via --config."""

    # -- JSON config --------------------------------------------------------

    def test_json_config_sets_algorithm(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.json"
        cfg.write_text('{"algorithm": "limma"}', encoding="utf-8")
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--config", str(cfg)])
        api = harmonize(DATA, BATCH, algorithm="limma")
        cli = pd.read_csv(out, sep="\t", index_col=0)
        pd.testing.assert_frame_equal(cli, api, atol=1e-10)

    def test_json_config_sets_combat_mode(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.json"
        cfg.write_text('{"combat_mode": 2}', encoding="utf-8")
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--config", str(cfg)])
        assert Path(out).exists()

    def test_json_config_sets_unique_removal_false(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.json"
        cfg.write_text('{"unique_removal": false}', encoding="utf-8")
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--config", str(cfg)])
        api = harmonize(DATA, BATCH, unique_removal=False)
        cli = pd.read_csv(out, sep="\t", index_col=0)
        pd.testing.assert_frame_equal(cli, api, atol=1e-10)

    def test_cli_flag_overrides_config(self, tmp_path: Path) -> None:
        """An explicit CLI flag must win over the config file value."""
        cfg = tmp_path / "cfg.json"
        cfg.write_text('{"algorithm": "limma"}', encoding="utf-8")
        out = str(tmp_path / "r.tsv")
        # CLI supplies --algorithm ComBat which should override config's limma
        main([DATA, BATCH, "-o", out, "--config", str(cfg), "--algorithm", "ComBat"])
        api_combat = harmonize(DATA, BATCH, algorithm="ComBat")
        cli = pd.read_csv(out, sep="\t", index_col=0)
        pd.testing.assert_frame_equal(cli, api_combat, atol=1e-10)

    def test_json_config_unknown_key_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = tmp_path / "cfg.json"
        cfg.write_text('{"unknown_param": 99}', encoding="utf-8")
        out = str(tmp_path / "r.tsv")
        with pytest.raises(SystemExit) as exc_info:
            main([DATA, BATCH, "-o", out, "--config", str(cfg)])
        assert exc_info.value.code != 0
        err = capsys.readouterr().err
        assert "unknown_param" in err

    def test_config_file_not_found_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([DATA, BATCH, "--config", str(tmp_path / "nonexistent.json")])
        assert exc_info.value.code != 0

    # -- TOML config --------------------------------------------------------

    def test_toml_config_sets_algorithm(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text('algorithm = "limma"\n', encoding="utf-8")
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--config", str(cfg)])
        api = harmonize(DATA, BATCH, algorithm="limma")
        cli = pd.read_csv(out, sep="\t", index_col=0)
        pd.testing.assert_frame_equal(cli, api, atol=1e-10)

    def test_toml_config_multiple_keys(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text(
            'algorithm = "ComBat"\ncombat_mode = 2\nunique_removal = false\n',
            encoding="utf-8",
        )
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--config", str(cfg)])
        api = harmonize(DATA, BATCH, combat_mode=2, unique_removal=False)
        cli = pd.read_csv(out, sep="\t", index_col=0)
        pd.testing.assert_frame_equal(cli, api, atol=1e-10)

    # -- Help text ----------------------------------------------------------

    def test_help_mentions_config(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main(["--help"])
        out = capsys.readouterr().out
        assert "--config" in out

    def test_help_mentions_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            main(["--help"])
        out = capsys.readouterr().out
        assert "--json" in out


# ===========================================================================
# --json flag
# ===========================================================================


@pytest.mark.skipif(not _has_small, reason="R fixtures not generated")
class TestCLIJson:
    """Tests for the --json stdout run-summary flag."""

    def test_json_flag_prints_valid_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--json"])
        captured = capsys.readouterr().out
        parsed = json.loads(captured)
        assert isinstance(parsed, dict)

    def test_json_output_contains_expected_keys(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--json"])
        summary = json.loads(capsys.readouterr().out)
        for key in (
            "harmonizepy_version",
            "algorithm",
            "combat_mode",
            "n_features_input",
            "n_features_output",
            "n_samples",
            "n_batches",
        ):
            assert key in summary, f"Missing key: {key}"

    def test_json_n_features_output_leq_input(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--json"])
        summary = json.loads(capsys.readouterr().out)
        assert summary["n_features_output"] <= summary["n_features_input"]

    def test_json_suppresses_info_logging(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """stdout must contain only JSON, no log lines mixed in."""
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--json"])
        stdout = capsys.readouterr().out.strip()
        # The entire stdout must be parseable as a single JSON object
        parsed = json.loads(stdout)
        assert isinstance(parsed, dict)

    def test_json_and_summary_both_work(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = str(tmp_path / "r.tsv")
        summary_file = str(tmp_path / "run.json")
        main([DATA, BATCH, "-o", out, "--json", "--summary", summary_file])
        # stdout JSON
        stdout_summary = json.loads(capsys.readouterr().out)
        assert isinstance(stdout_summary, dict)
        # file JSON
        with Path(summary_file).open() as fh:
            file_summary = json.load(fh)
        assert file_summary == stdout_summary

    def test_json_config_combo(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """--config + --json: config sets algorithm, json reports it."""
        cfg = tmp_path / "cfg.json"
        cfg.write_text('{"algorithm": "limma"}', encoding="utf-8")
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--config", str(cfg), "--json"])
        summary = json.loads(capsys.readouterr().out)
        assert summary["algorithm"] == "limma"

    # -- YAML config --------------------------------------------------------

    def test_yaml_config_sets_algorithm(self, tmp_path: Path) -> None:
        yaml = pytest.importorskip("yaml")  # skip when pyyaml is not installed
        del yaml  # only needed for the skip check; _load_config imports it internally
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("algorithm: limma\n", encoding="utf-8")
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--config", str(cfg)])
        api = harmonize(DATA, BATCH, algorithm="limma")
        cli = pd.read_csv(out, sep="\t", index_col=0)
        pd.testing.assert_frame_equal(cli, api, atol=1e-10)

    def test_yaml_config_yml_extension(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        cfg = tmp_path / "cfg.yml"
        cfg.write_text("unique_removal: false\n", encoding="utf-8")
        out = str(tmp_path / "r.tsv")
        main([DATA, BATCH, "-o", out, "--config", str(cfg)])
        assert Path(out).exists()

    # -- Unsupported extension ----------------------------------------------

    def test_unsupported_config_extension_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = tmp_path / "cfg.xml"
        cfg.write_text("<config/>", encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            main([DATA, BATCH, "--config", str(cfg)])
        assert exc_info.value.code != 0

    # -- --config + --dry-run combo -----------------------------------------

    def test_config_and_dry_run_combo(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Config-file settings must be visible in --dry-run output."""
        cfg = tmp_path / "cfg.json"
        cfg.write_text('{"algorithm": "limma"}', encoding="utf-8")
        main([DATA, BATCH, "--config", str(cfg), "--dry-run"])
        out = capsys.readouterr().out
        assert "limma" in out

    def test_config_dry_run_exits_zero(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.json"
        cfg.write_text('{"combat_mode": 2}', encoding="utf-8")
        # SystemExit is not raised on success. Function returns normally.
        main([DATA, BATCH, "--config", str(cfg), "--dry-run"])


# ===========================================================================
# Every option at once
# ===========================================================================


@pytest.mark.skipif(not _has_medium, reason="R medium fixtures not generated")
class TestCLIAllFlags:
    """Single test that fires every non-mutually-exclusive flag at once.

    Guards against unexpected flag interactions or import-time failures
    when the full combination is used together.
    """

    def test_all_flags_combined(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        out = str(tmp_path / "result.tsv")
        summary = str(tmp_path / "run.json")
        main(
            [
                MED_DATA,
                MED_BATCH,
                "-o",
                out,
                "--algorithm",
                "ComBat",
                "--combat-mode",
                "1",
                "--needed-values",
                "2",
                "--sort",
                "sparsity",
                "--block",
                "2",
                "--unique-removal",
                "--output-format",
                "tsv",
                "--summary",
                summary,
                "--json",
                "--verbose",
            ]
        )
        # Output file written
        assert Path(out).exists()
        # Summary file written and valid JSON
        file_summary = json.loads(Path(summary).read_text())
        assert file_summary["algorithm"] == "ComBat"
        # stdout JSON also valid (--json + --verbose: verbose wins for level but json still printed)
        stdout_json = json.loads(capsys.readouterr().out)
        assert stdout_json == file_summary

    def test_all_flags_matches_api(self, tmp_path: Path) -> None:
        """All-flags CLI result must equal harmonize() called with the same parameters."""
        out = str(tmp_path / "result.tsv")
        main(
            [
                MED_DATA,
                MED_BATCH,
                "-o",
                out,
                "--algorithm",
                "ComBat",
                "--combat-mode",
                "1",
                "--needed-values",
                "2",
                "--sort",
                "sparsity",
                "--block",
                "2",
                "--unique-removal",
            ]
        )
        cli = pd.read_csv(out, sep="\t", index_col=0)
        api = harmonize(
            MED_DATA,
            MED_BATCH,
            algorithm="ComBat",
            combat_mode=1,
            needed_values=2,
            sort="sparsity",
            block=2,
            unique_removal=True,
        )
        pd.testing.assert_frame_equal(cli, api, atol=1e-10)


_SMALL_DATA: str = (
    "id\ts1\ts2\ts3\ts4\ts5\ts6\n"
    "p1\t1.0\t2.0\t3.0\t4.0\t5.0\t6.5\n"
    "p2\t2.0\t2.5\t3.1\t4.2\t5.1\t6.0\n"
    "p3\t1.5\t2.7\t3.3\t3.9\t5.5\t6.9\n"
    "p4\t0.9\t2.2\t3.6\t4.1\t4.8\t6.2\n"
)
_DESC_ORDERS: dict[str, str] = {
    "ID, sample, batch": "ID,sample,batch\n"
    + "".join(f"s{i},{i},{1 if i <= 3 else 2}\n" for i in range(1, 7)),
    "sample, ID, batch": "sample,ID,batch\n"
    + "".join(f"{i},s{i},{1 if i <= 3 else 2}\n" for i in range(1, 7)),
    "ID, batch, sample": "ID,batch,sample\n"
    + "".join(f"s{i},{1 if i <= 3 else 2},{i}\n" for i in range(1, 7)),
}


def _small_inputs(
    tmp_path: Path, desc_text: str = _DESC_ORDERS["ID, sample, batch"]
) -> tuple[str, str]:
    data = tmp_path / "data.tsv"
    data.write_text(_SMALL_DATA, encoding="utf-8")
    desc = tmp_path / "desc.csv"
    desc.write_text(desc_text, encoding="utf-8")
    return str(data), str(desc)


class TestCLICsvDefaultName:
    """The default output name follows --output-format."""

    def test_csv_without_output_writes_csv_file(self, tmp_path: Path) -> None:
        """With --output-format csv and no -o, a comma-separated .csv file is written.

        Failure condition: CSV text is written into a file named .tsv.
        """
        from harmonizepy.__main__ import main

        data, desc = _small_inputs(tmp_path)
        main([data, desc, "--output-format", "csv", "-q"])
        written = tmp_path / "data_corrected.csv"
        assert written.exists()
        assert not (tmp_path / "data_corrected.tsv").exists()
        assert written.read_text(encoding="utf-8").splitlines()[0].count(",") == 6

    def test_posix_path_with_colon_stays_posix(self) -> None:
        """A POSIX path whose second character is a colon keeps its directory.

        Failure condition: the path is read as a Windows drive path and the output
        lands in the working directory under a name with a backslash.
        """
        from harmonizepy.__main__ import _resolve_output_path

        assert _resolve_output_path("a:b/data.tsv", None) == "a:b/data_corrected.tsv"
        assert _resolve_output_path("C:\\x\\data.tsv", None) == "C:\\x\\data_corrected.tsv"


class TestCLIDryRunWritesNoFiles:
    """A dry run writes no files."""

    def test_dry_run_writes_no_log_file(self, tmp_path: Path) -> None:
        """A dry run creates neither the output nor a log file.

        Failure condition: a log is written next to an output that is never created.
        """
        from harmonizepy.__main__ import main

        data, desc = _small_inputs(tmp_path)
        main([data, desc, "--dry-run"])
        assert not list(tmp_path.glob("*.log"))
        assert not (tmp_path / "data_corrected.tsv").exists()

    @pytest.mark.parametrize("flag", ["--log-file", "--summary", "--json"])
    def test_dry_run_with_file_or_summary_option_is_an_error(
        self, tmp_path: Path, flag: str
    ) -> None:
        """--dry-run with an option it would ignore exits with a usage error and writes nothing.

        Failure condition: the option is silently ignored and the run exits 0.
        """
        from harmonizepy.__main__ import main

        data, desc = _small_inputs(tmp_path)
        extra = [] if flag == "--json" else [str(tmp_path / "x.out")]
        with pytest.raises(SystemExit) as excinfo:
            main([data, desc, "--dry-run", flag, *extra])
        assert excinfo.value.code == 2
        assert not (tmp_path / "x.out").exists()

    def test_two_column_description_is_a_validation_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A description with two columns gives an error message, not a traceback.

        Failure condition: an IndexError escapes from the batch lookup.
        """
        from harmonizepy.__main__ import main

        data, desc = _small_inputs(tmp_path, "ID,grp\n" + "".join(f"s{i},1\n" for i in range(1, 7)))
        with pytest.raises(SystemExit) as excinfo:
            main([data, desc, "-q"])
        assert excinfo.value.code == 1
        assert "at least 3 columns" in capsys.readouterr().err


class TestCLIBatchCount:
    """The batch count does not depend on the column order of the description."""

    @pytest.mark.parametrize("order", list(_DESC_ORDERS))
    def test_summary_batch_count(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], order: str
    ) -> None:
        """Every description column order gives two batches in the JSON summary.

        Failure condition: the summary counts the values of another column as batches.
        """
        from harmonizepy.__main__ import main

        data, desc = _small_inputs(tmp_path, _DESC_ORDERS[order])
        main([data, desc, "--json", "-q"])
        assert json.loads(capsys.readouterr().out)["n_batches"] == 2

    @pytest.mark.parametrize("order", list(_DESC_ORDERS))
    def test_dry_run_batch_count(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], order: str
    ) -> None:
        """Every description column order gives two batches in the dry-run plan.

        Failure condition: the plan counts the values of another column as batches.
        """
        from harmonizepy.__main__ import main

        data, desc = _small_inputs(tmp_path, _DESC_ORDERS[order])
        main([data, desc, "--dry-run"])
        assert re.search(r"Batches:\s+2\b", capsys.readouterr().out)


_PATTERN_DATA: str = (
    "id\ts1\ts2\ts3\ts4\ts5\ts6\n"
    + "".join(f"p{i}\t{1.0 + i}\t{1.5 + i}\t{2.1 + i}\t{2.4 + i}\t\t\n" for i in range(1, 5))
    + "p5\t1.0\t1.4\t2.0\t2.6\t3.1\t3.5\n"
)
_PATTERN_DESC: str = "ID,sample,batch\n" + "".join(
    f"s{i},{i},{(i + 1) // 2}\n" for i in range(1, 7)
)


class TestCLIDryRunMatchesRun:
    """The dry run reports the groups the correction run uses."""

    @pytest.mark.parametrize("extra", [[], ["--sort", "sparsity", "--block", "2"]])
    def test_submatrix_count_matches_correction_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
    ) -> None:
        """The dry-run sub-matrix count equals the number of groups the run corrects.

        p5 is the only feature seen in all three batches, so unique removal moves it
        into the pattern of p1 to p4 and the run corrects one group.

        Failure condition: the dry run skips unique removal or sorting and reports a
        different count.
        """
        data = tmp_path / "data.tsv"
        data.write_text(_PATTERN_DATA, encoding="utf-8")
        desc = tmp_path / "desc.csv"
        desc.write_text(_PATTERN_DESC, encoding="utf-8")

        main([str(data), str(desc), "--dry-run", *extra])
        planned = re.search(r"Sub-matrices:\s+(\d+)", capsys.readouterr().out)
        main([str(data), str(desc), "--no-log", *extra])
        corrected = re.search(r"Adjusting (\d+) unique affiliation group", capsys.readouterr().err)
        assert planned is not None and corrected is not None
        assert planned.group(1) == corrected.group(1) == "1"


class TestCLIRepeatedCalls:
    """Several runs in one process."""

    def test_second_run_logs_each_line_once(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A second ``main()`` call prints each log line once.

        Failure condition: the first call's handlers stay attached, and every line of
        the second call appears twice.
        """
        data, desc = _small_inputs(tmp_path)
        main([data, desc, "--no-log"])
        capsys.readouterr()
        main([data, desc, "--no-log"])
        lines = capsys.readouterr().err.splitlines()
        done = [line for line in lines if "Done (" in line]
        assert len(done) == 1


class TestCLISummaryValues:
    """Values recorded in the run summary."""

    def test_summary_records_resolved_needed_values(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Without --needed-values, the summary records the value the run used.

        Failure condition: the summary records null for an automatic value.
        """
        data, desc = _small_inputs(tmp_path)
        main([data, desc, "--json", "-q", "--combat-mode", "2"])
        assert json.loads(capsys.readouterr().out)["needed_values"] == 1


class TestCLIConfigErrors:
    """Unreadable config files end with a usage error, not a traceback."""

    def test_malformed_yaml_is_a_usage_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A YAML config that does not parse exits with code 2 and a message.

        Failure condition: the YAML parser's exception escapes as a traceback.
        """
        pytest.importorskip("yaml")
        data, desc = _small_inputs(tmp_path)
        config = tmp_path / "run.yaml"
        config.write_text("algorithm: [unclosed\n", encoding="utf-8")
        with pytest.raises(SystemExit) as excinfo:
            main([data, desc, "--config", str(config)])
        assert excinfo.value.code == 2
        assert "invalid YAML" in capsys.readouterr().err


def test_main_restores_package_logger_settings(tmp_path: Path) -> None:
    """After ``main()`` returns, the package logger has its earlier level and propagation.

    Failure condition: the logger stays at DEBUG without propagation, so later
    library calls in the same process lose records from root handlers.
    """
    import logging

    package_logger = logging.getLogger("harmonizepy")
    before = (package_logger.level, package_logger.propagate)
    data, desc = _small_inputs(tmp_path)
    main([data, desc, "--no-log", "-q"])
    assert (package_logger.level, package_logger.propagate) == before


def test_dry_run_reports_untransformed_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dry run shows the input-scale warning on stderr for raw intensities.

    Failure condition: the dry run validates raw intensities without a word.
    """
    rng = np.random.default_rng(31)
    samples = [f"s{i}" for i in range(1, 7)]
    data = pd.DataFrame(
        rng.lognormal(12, 1, (20, 6)), index=[f"p{i}" for i in range(20)], columns=samples
    )
    data_path = tmp_path / "raw.tsv"
    data.to_csv(data_path, sep="\t")
    desc_path = tmp_path / "desc.csv"
    pd.DataFrame({"ID": samples, "sample": range(1, 7), "batch": [1, 1, 1, 2, 2, 2]}).to_csv(
        desc_path, index=False
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        main([str(data_path), str(desc_path), "--dry-run"])
    assert any("look untransformed" in str(w.message) for w in caught)


class TestCLIOutputSafety:
    """No output file may overwrite an input or another output, and config values are checked."""

    @pytest.mark.parametrize("flag", ["-o", "--summary", "--log-file", "--report"])
    def test_output_naming_the_data_file_is_refused(self, tmp_path: Path, flag: str) -> None:
        """Any output named like the data file is a usage error, and the file is untouched.

        Failure condition: ``-o`` or ``--summary`` overwrites the data file, or
        ``--log-file`` truncates it before it is read.
        """
        data, desc = _small_inputs(tmp_path)
        before = Path(data).read_bytes()
        argv = [data, desc, "-q", flag, data]
        if flag != "-o":
            argv += ["-o", str(tmp_path / "out.tsv")]
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2
        assert Path(data).read_bytes() == before

    def test_two_outputs_naming_one_file_are_refused(self, tmp_path: Path) -> None:
        """The summary and the report cannot be the same file.

        Failure condition: the report silently replaces the summary.
        """
        data, desc = _small_inputs(tmp_path)
        same = str(tmp_path / "same.json")
        with pytest.raises(SystemExit) as excinfo:
            main([data, desc, "--no-log", "-q", "--summary", same, "--report", same])
        assert excinfo.value.code == 2

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("output_format", "xlsx"),
            ("combat_mode", 9),
            ("needed_values", 2.5),
            ("unique_removal", "no"),
            ("block", True),
        ],
    )
    def test_invalid_config_value_is_a_usage_error(
        self, tmp_path: Path, key: str, value: object
    ) -> None:
        """A config value the matching flag would reject exits with code 2.

        Failure condition: ``output_format: xlsx`` writes TSV and exits 0, or a bad
        mode exits 1 after reading the inputs.
        """
        data, desc = _small_inputs(tmp_path)
        config = tmp_path / "run.json"
        config.write_text(json.dumps({key: value}), encoding="utf-8")
        with pytest.raises(SystemExit) as excinfo:
            main([data, desc, "--no-log", "-q", "--config", str(config)])
        assert excinfo.value.code == 2

    def test_dry_run_refuses_a_flag_that_repeats_the_config(self, tmp_path: Path) -> None:
        """``--summary`` typed on the command line conflicts with a dry run even when
        the config file holds the same value.

        Failure condition: the flag is accepted because its value equals the config's.
        """
        data, desc = _small_inputs(tmp_path)
        summary = str(tmp_path / "s.json")
        config = tmp_path / "run.json"
        config.write_text(json.dumps({"summary": summary}), encoding="utf-8")
        with pytest.raises(SystemExit) as excinfo:
            main([data, desc, "--dry-run", "--config", str(config), "--summary", summary])
        assert excinfo.value.code == 2
