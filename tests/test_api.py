"""API surface tests for HarmonizePy.

Exercises the three levels of public API:
1. Pipeline-level: ``harmonize()``
2. Engine-level: ``combat()``, ``remove_batch_effect()``
3. Wrapper-level: ``adjust_combat()``, ``adjust_limma()``

Also verifies that all public symbols are importable from the top-level
package and that Path objects work as inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
_has_fixtures = (FIXTURE_DIR / "small_input.tsv").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n_features=20, n_per_batch=5, n_batches=3, seed=0):
    """Synthetic data with known batch shifts."""
    rng = np.random.default_rng(seed)
    n_samples = n_per_batch * n_batches
    data = rng.normal(10, 2, size=(n_features, n_samples))
    batches = np.repeat(range(n_batches), n_per_batch)
    for b in range(n_batches):
        data[:, batches == b] += rng.normal(0, 2)
    return data, batches


def _make_df_and_desc(n_features=20, n_per_batch=5, n_batches=3, seed=0):
    """Return a DataFrame + description DataFrame pair."""
    data, batches = _make_data(n_features, n_per_batch, n_batches, seed)
    n_samples = data.shape[1]
    df = pd.DataFrame(
        data,
        index=[f"feat_{i}" for i in range(n_features)],
        columns=[f"s{j}" for j in range(n_samples)],
    )
    desc = pd.DataFrame(
        {
            "ID": df.columns.tolist(),
            "sample": list(range(1, n_samples + 1)),
            "batch": batches + 1,
        }
    )
    return df, desc


# ---------------------------------------------------------------------------
# 1. Import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_top_level_imports(self):
        from harmonizepy import (
            HarmonizeConfig,
            __version__,
            adjust_combat,
            adjust_limma,
            combat,
            harmonize,
            remove_batch_effect,
        )

        assert callable(harmonize)
        assert callable(combat)
        assert callable(adjust_combat)
        assert callable(remove_batch_effect)
        assert callable(adjust_limma)
        assert isinstance(__version__, str)
        assert HarmonizeConfig is not None

    def test_version_format(self):
        from harmonizepy import __version__

        parts = __version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# 2. Pipeline-level: harmonize()
# ---------------------------------------------------------------------------


class TestHarmonizePipeline:
    def test_dataframe_input(self):
        """harmonize() accepts DataFrames and returns a DataFrame."""
        from harmonizepy import harmonize

        df, desc = _make_df_and_desc()
        result = harmonize(df, desc)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape
        assert not np.isnan(result.values).any()

    @pytest.mark.skipif(not _has_fixtures, reason="R fixtures not generated")
    def test_path_input(self, tmp_path):
        """harmonize() accepts string paths."""
        from harmonizepy import harmonize

        result = harmonize(
            str(FIXTURE_DIR / "small_input.tsv"),
            str(FIXTURE_DIR / "small_batch.csv"),
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    @pytest.mark.skipif(not _has_fixtures, reason="R fixtures not generated")
    def test_pathlib_input(self):
        """harmonize() accepts pathlib.Path objects."""
        from harmonizepy import harmonize

        result = harmonize(
            FIXTURE_DIR / "small_input.tsv",
            FIXTURE_DIR / "small_batch.csv",
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] > 0

    @pytest.mark.skipif(not _has_fixtures, reason="R fixtures not generated")
    def test_path_and_df_produce_same_result(self):
        """File path and DataFrame inputs yield identical output."""
        from harmonizepy import harmonize

        result_paths = harmonize(
            str(FIXTURE_DIR / "small_input.tsv"),
            str(FIXTURE_DIR / "small_batch.csv"),
        )
        data_df = pd.read_csv(FIXTURE_DIR / "small_input.tsv", sep="\t", index_col=0)
        desc_df = pd.read_csv(FIXTURE_DIR / "small_batch.csv")
        result_dfs = harmonize(data_df, desc_df)

        pd.testing.assert_frame_equal(result_paths, result_dfs)

    def test_output_file(self, tmp_path):
        """output_file writes a TSV."""
        from harmonizepy import harmonize

        df, desc = _make_df_and_desc()
        out = tmp_path / "out.tsv"
        result = harmonize(df, desc, output_file=out)
        assert out.exists()
        reloaded = pd.read_csv(out, sep="\t", index_col=0)
        # Same float64 data through TSV serialization; 1e-10 covers
        # text-format rounding differences (17 significant digits).
        np.testing.assert_allclose(reloaded.values, result.values, rtol=1e-10)

    def test_output_file_path_object(self, tmp_path):
        """output_file accepts a Path object."""
        from harmonizepy import harmonize

        df, desc = _make_df_and_desc()
        out = Path(tmp_path) / "result.tsv"
        harmonize(df, desc, output_file=out)
        assert out.exists()

    def test_algorithm_limma(self):
        from harmonizepy import harmonize

        df, desc = _make_df_and_desc()
        result = harmonize(df, desc, algorithm="limma")
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape

    @pytest.mark.parametrize("mode", [1, 2, 3, 4])
    def test_combat_modes(self, mode):
        from harmonizepy import harmonize

        df, desc = _make_df_and_desc()
        result = harmonize(df, desc, combat_mode=mode)
        assert result.shape == df.shape
        assert not np.isnan(result.values).any()

    def test_needed_values_explicit(self):
        """Explicit needed_values is respected."""
        from harmonizepy import harmonize

        df, desc = _make_df_and_desc()
        r1 = harmonize(df, desc, needed_values=1)
        r2 = harmonize(df, desc, needed_values=2)
        # Both paths use the same float64 pipeline; no NaN in this data
        # so affiliation is identical. 1e-10 covers float64 accumulation
        # differences from different needed_values code branches.
        np.testing.assert_allclose(r1.values, r2.values, rtol=1e-10)

    def test_invalid_algorithm_raises(self):
        from harmonizepy import harmonize

        df, desc = _make_df_and_desc()
        with pytest.raises(ValueError, match="algorithm"):
            harmonize(df, desc, algorithm="wrong")  # type: ignore[arg-type]

    def test_invalid_mode_raises(self):
        from harmonizepy import harmonize

        df, desc = _make_df_and_desc()
        with pytest.raises(ValueError, match="combat_mode"):
            harmonize(df, desc, combat_mode=99)

    def test_does_not_mutate_input(self):
        """harmonize() must not modify the caller's DataFrame.

        Failure condition: in-place mutation inside the pipeline
        (e.g. ``data.iloc[:, :] = ...`` as a debugging artifact).
        """
        from harmonizepy import harmonize

        df, desc = _make_df_and_desc()
        original = df.values.copy()
        harmonize(df, desc)
        np.testing.assert_array_equal(df.values, original)


# ---------------------------------------------------------------------------
# 3. Engine-level: combat()
# ---------------------------------------------------------------------------


class TestCombatEngine:
    def test_shape_and_no_nan(self):
        from harmonizepy import combat

        data, batches = _make_data()
        result = combat(data, batches)
        assert result.shape == data.shape
        assert not np.isnan(result).any()

    def test_batch_means_converge(self):
        """ComBat must reduce batch mean spread.

        Failure condition: if correction does not shrink spread, the test
        fails. Threshold of 1.0 reflects residual variance from EB
        shrinkage; closed-form modes would converge tighter.
        """
        from harmonizepy import combat

        data, batches = _make_data()
        means_before = [data[:, batches == b].mean() for b in range(3)]
        spread_before = max(means_before) - min(means_before)
        result = combat(data, batches)
        means_after = [result[:, batches == b].mean() for b in range(3)]
        spread_after = max(means_after) - min(means_after)
        assert spread_after < spread_before, (
            f"Batch spread not reduced: {spread_before:.3f} -> {spread_after:.3f}"
        )

    def test_returns_ndarray(self):
        from harmonizepy import combat

        data, batches = _make_data()
        result = combat(data, batches)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_all_modes(self):
        """All four ComBat modes must change the data and reduce batch spread.

        Failure condition: if a mode is broken and returns the input
        unchanged, or fails to reduce batch mean spread, the test fails.
        """
        from harmonizepy import combat

        data, batches = _make_data()
        means_before = [data[:, batches == b].mean() for b in range(3)]
        spread_before = max(means_before) - min(means_before)
        for par, mo in [(True, False), (True, True), (False, False), (False, True)]:
            result = combat(data, batches, par_prior=par, mean_only=mo)
            assert result.shape == data.shape
            assert not np.isnan(result).any()
            means_after = [result[:, batches == b].mean() for b in range(3)]
            spread_after = max(means_after) - min(means_after)
            assert spread_after < spread_before, (
                f"Mode par={par} mean_only={mo}: spread {spread_before:.3f} -> {spread_after:.3f}"
            )

    def test_ref_batch(self):
        from harmonizepy import combat

        data, batches = _make_data()
        result = combat(data, batches, ref_batch=0)
        np.testing.assert_array_equal(result[:, batches == 0], data[:, batches == 0])

    def test_output_isolation(self):
        """Mutating the result must not alter the input.

        Failure condition: if memory is shared via an aliased return
        (e.g. ``return data`` instead of ``return data.copy()``),
        modifying the output corrupts the input.
        """
        from harmonizepy import combat

        data, batches = _make_data()
        original = data.copy()
        result = combat(data, batches)
        result[0, 0] = -9999.0
        np.testing.assert_array_equal(data, original)

    def test_deterministic(self):
        """Same input always yields identical output.

        Failure condition: a non-deterministic code path was introduced
        (unseeded random, iteration-order-dependent dict, etc.).
        """
        from harmonizepy import combat

        data, batches = _make_data()
        result1 = combat(data, batches)
        result2 = combat(data, batches)
        np.testing.assert_array_equal(result1, result2)


# ---------------------------------------------------------------------------
# 4. Engine-level: remove_batch_effect()
# ---------------------------------------------------------------------------


class TestLimmaEngine:
    def test_shape_and_no_nan(self):
        from harmonizepy import remove_batch_effect

        data, batches = _make_data()
        result = remove_batch_effect(data, batches)
        assert result.shape == data.shape
        assert not np.isnan(result).any()

    def test_batch_means_converge(self):
        """limma must reduce batch mean spread.

        Failure condition: if the batch-effect subtraction is skipped or
        miscalculated, the spread stays wide and the test fails.
        """
        from harmonizepy import remove_batch_effect

        data, batches = _make_data()
        means_before = [data[:, batches == b].mean() for b in range(3)]
        spread_before = max(means_before) - min(means_before)
        result = remove_batch_effect(data, batches)
        means_after = [result[:, batches == b].mean() for b in range(3)]
        spread_after = max(means_after) - min(means_after)
        assert spread_after < spread_before, (
            f"Batch spread not reduced: {spread_before:.3f} -> {spread_after:.3f}"
        )

    def test_returns_ndarray(self):
        from harmonizepy import remove_batch_effect

        data, batches = _make_data()
        result = remove_batch_effect(data, batches)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_output_isolation(self):
        """Mutating the result must not alter the input.

        Failure condition: memory aliasing between input and output.
        """
        from harmonizepy import remove_batch_effect

        data, batches = _make_data()
        original = data.copy()
        result = remove_batch_effect(data, batches)
        result[0, 0] = -9999.0
        np.testing.assert_array_equal(data, original)

    def test_deterministic(self):
        """Same input always yields identical output.

        Failure condition: non-deterministic code path.
        """
        from harmonizepy import remove_batch_effect

        data, batches = _make_data()
        result1 = remove_batch_effect(data, batches)
        result2 = remove_batch_effect(data, batches)
        np.testing.assert_array_equal(result1, result2)


# ---------------------------------------------------------------------------
# 5. Wrapper-level: adjust_combat(), adjust_limma()
# ---------------------------------------------------------------------------


class TestWrapperLevel:
    def test_adjust_combat_returns_dataframe(self):
        """adjust_combat returns a DataFrame with preserved index/columns.

        Failure condition: the return type is wrong or index/columns
        are dropped.
        """
        from harmonizepy import adjust_combat

        data, batches = _make_data()
        df = pd.DataFrame(data, index=[f"f{i}" for i in range(data.shape[0])])
        result = adjust_combat(df, batches, mode=1)
        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == list(df.index)
        assert result.shape == df.shape

    @pytest.mark.parametrize("mode", [1, 2, 3, 4])
    def test_adjust_combat_modes(self, mode):
        """All 4 combat modes produce NaN-free output with correct shape.

        Failure condition: any mode produces NaN or wrong shape.
        """
        from harmonizepy import adjust_combat

        data, batches = _make_data()
        df = pd.DataFrame(data)
        result = adjust_combat(df, batches, mode=mode)
        assert result.shape == df.shape
        assert not np.isnan(result.values).any()

    def test_adjust_limma_returns_dataframe(self):
        """adjust_limma returns a DataFrame with preserved index/columns.

        Failure condition: the return type is wrong or index/columns
        are dropped.
        """
        from harmonizepy import adjust_limma

        data, batches = _make_data()
        df = pd.DataFrame(data, index=[f"f{i}" for i in range(data.shape[0])])
        result = adjust_limma(df, batches)
        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == list(df.index)
        assert result.shape == df.shape

    def test_adjust_combat_invalid_mode(self):
        """Invalid mode passed to adjust_combat raises ValueError.

        Failure condition: mode=0 is accepted instead of rejected.
        """
        from harmonizepy import adjust_combat

        data, batches = _make_data()
        df = pd.DataFrame(data)
        with pytest.raises(ValueError, match="mode"):
            adjust_combat(df, batches, mode=0)


# ---------------------------------------------------------------------------
# 6. HarmonizeConfig
# ---------------------------------------------------------------------------


class TestHarmonizeConfig:
    def test_defaults(self):
        """Default config uses ComBat mode 1 with auto needed_values.

        Failure condition: defaults change, breaking backward compatibility.
        """
        from harmonizepy import HarmonizeConfig

        cfg = HarmonizeConfig()
        assert cfg.algorithm == "ComBat"
        assert cfg.combat_mode == 1
        assert cfg.needed_values is None

    def test_custom(self):
        """Custom config values are stored correctly.

        Failure condition: a custom value is overwritten or ignored.
        """
        from harmonizepy import HarmonizeConfig

        cfg = HarmonizeConfig(algorithm="limma", combat_mode=3, needed_values=1)
        assert cfg.algorithm == "limma"
        assert cfg.combat_mode == 3
        assert cfg.needed_values == 1

    def test_frozen(self):
        """HarmonizeConfig is frozen and setting an attribute raises.

        Failure condition: a frozen dataclass allows mutation.
        """
        from harmonizepy import HarmonizeConfig

        cfg = HarmonizeConfig()
        with pytest.raises(AttributeError):
            cfg.algorithm = "limma"  # type: ignore[misc]

    def test_invalid_algorithm(self):
        """Invalid algorithm raises ValueError.

        Failure condition: a bogus algorithm name is accepted.
        """
        from harmonizepy import HarmonizeConfig

        with pytest.raises(ValueError, match="algorithm"):
            HarmonizeConfig(algorithm="invalid")

    def test_invalid_mode(self):
        """Invalid combat mode raises ValueError.

        Failure condition: an out-of-range mode is accepted.
        """
        from harmonizepy import HarmonizeConfig

        with pytest.raises(ValueError, match="combat_mode"):
            HarmonizeConfig(combat_mode=5)

    def test_invalid_needed_values(self):
        """needed_values=0 raises ValueError.

        Failure condition: zero is accepted as a valid threshold.
        """
        from harmonizepy import HarmonizeConfig

        with pytest.raises(ValueError, match="needed_values"):
            HarmonizeConfig(needed_values=0)

    def test_sort_strategy_sparsity(self):
        """sparsity sort strategy is stored correctly.

        Failure condition: a valid strategy is rejected.
        """
        from harmonizepy import HarmonizeConfig

        cfg = HarmonizeConfig(sort_strategy="sparsity")
        assert cfg.sort_strategy == "sparsity"

    def test_sort_strategy_jaccard(self):
        """jaccard sort strategy is stored correctly.

        Failure condition: a valid strategy is rejected.
        """
        from harmonizepy import HarmonizeConfig

        cfg = HarmonizeConfig(sort_strategy="jaccard")
        assert cfg.sort_strategy == "jaccard"

    def test_sort_strategy_seriation(self):
        """seriation sort strategy is stored correctly.

        Failure condition: a valid strategy is rejected.
        """
        from harmonizepy import HarmonizeConfig

        cfg = HarmonizeConfig(sort_strategy="seriation")
        assert cfg.sort_strategy == "seriation"

    def test_block_size(self):
        """Valid block sizes are stored correctly.

        Failure condition: a valid block size is rejected.
        """
        from harmonizepy import HarmonizeConfig

        cfg = HarmonizeConfig(block_size=2)
        assert cfg.block_size == 2
        cfg = HarmonizeConfig(block_size=4)
        assert cfg.block_size == 4

    def test_unique_removal_false(self):
        """unique_removal=False is stored correctly.

        Failure condition: False is coerced to True.
        """
        from harmonizepy import HarmonizeConfig

        cfg = HarmonizeConfig(unique_removal=False)
        assert cfg.unique_removal is False

    def test_sort_and_block_combination(self):
        """Combining sort strategy and block size works.

        Failure condition: setting both fields causes a conflict.
        """
        from harmonizepy import HarmonizeConfig

        cfg = HarmonizeConfig(sort_strategy="sparsity", block_size=3)
        assert cfg.sort_strategy == "sparsity"
        assert cfg.block_size == 3

    def test_invalid_sort_strategy(self):
        """Unknown sort strategy raises ValueError.

        Failure condition: a nonsense strategy is accepted.
        """
        from harmonizepy import HarmonizeConfig

        with pytest.raises(ValueError, match="sort must be one of"):
            HarmonizeConfig(sort_strategy="unknown")

    def test_invalid_block_size_zero(self):
        """block_size=0 raises ValueError.

        Failure condition: zero is accepted.
        """
        from harmonizepy import HarmonizeConfig

        with pytest.raises(ValueError, match="block must be an integer"):
            HarmonizeConfig(block_size=0)

    def test_invalid_block_size_one(self):
        """block_size=1 raises ValueError.

        Failure condition: 1 is accepted.
        """
        from harmonizepy import HarmonizeConfig

        with pytest.raises(ValueError, match="block must be an integer"):
            HarmonizeConfig(block_size=1)

    def test_unique_removal_not_bool(self):
        """Non-bool unique_removal raises TypeError.

        Failure condition: a string is accepted where bool is required.
        """
        from harmonizepy import HarmonizeConfig

        with pytest.raises(TypeError, match="unique_removal"):
            HarmonizeConfig(unique_removal="true")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 7. harmonize() config= parameter
# ---------------------------------------------------------------------------


class TestHarmonizeWithConfig:
    """harmonize() accepts a HarmonizeConfig that overrides individual kwargs."""

    @pytest.fixture()
    def small_inputs(self):
        rng = np.random.default_rng(99)
        data = pd.DataFrame(
            rng.normal(10, 2, size=(8, 6)),
            index=[f"p{i}" for i in range(8)],
            columns=[f"s{j}" for j in range(6)],
        )
        data.iloc[:, 3:] += 3.0
        desc = pd.DataFrame(
            {
                "ID": [f"s{j}" for j in range(6)],
                "sample": list(range(1, 7)),
                "batch": [1, 1, 1, 2, 2, 2],
            }
        )
        return data, desc

    def test_config_equivalent_to_kwargs(self, small_inputs):
        """Config with same values as kwargs must produce identical output.

        Failure condition: the config code path diverges from the
        direct-kwarg path.
        """
        from harmonizepy import HarmonizeConfig, harmonize

        data, desc = small_inputs
        cfg = HarmonizeConfig(algorithm="ComBat", combat_mode=2)
        result_cfg = harmonize(data, desc, config=cfg)
        result_kw = harmonize(data, desc, algorithm="ComBat", combat_mode=2)
        pd.testing.assert_frame_equal(result_cfg, result_kw)

    def test_config_limma(self, small_inputs):
        """Config selecting limma matches direct kwarg.

        Failure condition: config-based limma path produces different
        results than direct kwarg.
        """
        from harmonizepy import HarmonizeConfig, harmonize

        data, desc = small_inputs
        cfg = HarmonizeConfig(algorithm="limma")
        result_cfg = harmonize(data, desc, config=cfg)
        result_kw = harmonize(data, desc, algorithm="limma")
        pd.testing.assert_frame_equal(result_cfg, result_kw)

    def test_config_overrides_kwargs(self, small_inputs):
        """When config is provided, its algorithm must win over kwargs.

        Failure condition: a conflicting kwarg overrides the config
        value.
        """
        from harmonizepy import HarmonizeConfig, harmonize

        data, desc = small_inputs
        cfg = HarmonizeConfig(algorithm="limma")
        result_cfg = harmonize(data, desc, config=cfg, algorithm="ComBat")
        result_limma = harmonize(data, desc, algorithm="limma")
        pd.testing.assert_frame_equal(result_cfg, result_limma)

    def test_config_needed_values_none_auto_selects(self, small_inputs):
        """Config with needed_values=None must auto-select like the default.

        Failure condition: an explicit None in config bypasses the
        auto-selection logic.
        """
        from harmonizepy import HarmonizeConfig, harmonize

        data, desc = small_inputs
        cfg = HarmonizeConfig(needed_values=None)
        result_cfg = harmonize(data, desc, config=cfg)
        result_kw = harmonize(data, desc)
        pd.testing.assert_frame_equal(result_cfg, result_kw)
