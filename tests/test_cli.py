"""Tests for the harmonizepy CLI (__main__.py).

Calls main(argv=[...]) directly so no installation or subprocess overhead.
All tests that write files use tmp_path for isolation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

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
        assert result == "/data/my_file_corrected.parquet"

    def test_default_stem_preservation(self) -> None:
        """Input filename stem is preserved in the default output path.

        Failure condition: the stem is modified or dropped.
        """
        result = _resolve_output_path("proteins.tsv", None)
        assert result == "proteins_corrected.parquet"

    def test_default_tsv_when_no_pyarrow(self) -> None:
        """Default path falls back to .tsv when pyarrow is not installed.

        Failure condition: the default path uses .parquet despite missing pyarrow.
        """
        from harmonizepy.io import _HAVE_PYARROW

        if not _HAVE_PYARROW:
            result = _resolve_output_path("/data/my_file.tsv", None)
            assert result == "/data/my_file_corrected.tsv"


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
        expected = tmp_path / "small_input_corrected.parquet"
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
        """block=1 fails validation inside harmonize → exit 1."""
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
# All-flags smoke test (plan §2.5 test_cli_all_flags)
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
