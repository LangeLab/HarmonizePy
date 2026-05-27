"""Direct unit tests for harmonizepy.splitting.

Tests the splitting logic in isolation: affiliation grouping, sub-matrix
extraction, adjustment dispatch, and NaN preservation.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from harmonizepy.affiliation import build_affiliation_list
from harmonizepy.combat import combat as _combat
from harmonizepy.limma_wrapper import remove_batch_effect as _remove_batch_effect
from harmonizepy.splitting import splitting

_Affil = list[tuple[int, ...]]


def _affil(*rows: tuple[int, ...]) -> _Affil:
    return cast(_Affil, list(rows))


def _make_data(n_features: int = 6, n_samples: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = rng.normal(10, 2, size=(n_features, n_samples))
    df = pd.DataFrame(
        data,
        index=[f"p{i}" for i in range(n_features)],
        columns=[f"s{j}" for j in range(n_samples)],
    )
    return df


class TestSplittingBasic:
    def test_all_features_one_group(self) -> None:
        """All features in one affiliation group produce a single sub-DataFrame.

        Failure condition: features are split across multiple groups
        when they share the same affiliation.
        """
        data = _make_data()
        batch = np.array([1, 1, 1, 2, 2, 2])
        block = batch.copy()
        affil = _affil(*[(1, 2)] * data.shape[0])
        result = splitting(affil, data, batch, block, algorithm="ComBat", combat_mode=2)
        assert len(result) == 1
        concat = pd.concat(result, axis=0)
        assert concat.shape == data.shape
        assert not concat.isna().any().any()

    def test_two_groups(self) -> None:
        """Two affiliation groups produce two sub-DataFrames.

        Failure condition: groups are merged or dropped.
        """
        data = _make_data(4, 6)
        batch = np.array([1, 1, 1, 2, 2, 2])
        block = batch.copy()
        affil = _affil((1,), (1,), (1, 2), (1, 2))
        result = splitting(affil, data, batch, block, algorithm="limma")
        assert len(result) == 2
        concat = pd.concat(result, axis=0)
        assert concat.shape == data.shape

    def test_empty_affiliation_stays_nan(self) -> None:
        """Features with empty affiliation remain all-NaN in output.

        Failure condition: an empty-affiliation feature is dropped
        or filled with non-NaN values.
        """
        data = _make_data(4, 6)
        batch = np.array([1, 1, 1, 2, 2, 2])
        block = batch.copy()
        affil = _affil((), (1, 2), (1, 2), (1, 2))
        result = splitting(affil, data, batch, block, algorithm="limma")
        concat = pd.concat(result, axis=0)
        assert concat.index[0] == data.index[0]
        assert concat.iloc[0].isna().all()

    def test_single_batch_group_copies_raw(self) -> None:
        """Single-batch groups pass raw values through, no adjustment.

        Failure condition: a single-batch group is adjusted or zeroed out.
        """
        data = _make_data(4, 6)
        batch = np.array([1, 1, 1, 2, 2, 2])
        block = batch.copy()
        affil = _affil((1,), (1,), (1,), (1,))
        result = splitting(affil, data, batch, block, algorithm="ComBat", combat_mode=2)
        concat = pd.concat(result, axis=0)
        np.testing.assert_allclose(concat.values[:, :3], data.values[:, :3])
        assert concat.iloc[:, 3:].isna().all().all()

    def test_single_feature_group_copies_raw(self) -> None:
        """Single-feature groups pass raw values through, no adjustment.

        Failure condition: a single feature is adjusted instead of
        passed through unchanged.
        """
        data = _make_data(4, 6)
        batch = np.array([1, 1, 1, 2, 2, 2])
        block = batch.copy()
        affil = _affil((1,), (1, 2), (1, 2), (1, 2))
        result = splitting(affil, data, batch, block, algorithm="ComBat", combat_mode=2)
        concat = pd.concat(result, axis=0)
        assert concat.shape == data.shape


class TestSplittingNanAudit:
    def test_per_cell_nan_passed_to_combat(self, monkeypatch: Any) -> None:
        """Sub-DataFrames with per-cell NaN are passed to combat, which handles it.

        Failure condition: splitting tries to drop columns or otherwise
        prevent NaN from reaching the engine, losing valid data.

        This matches R HarmonizR behavior: per-cell NaN within qualifying
        blocks reaches sva::ComBat, which handles it via na.omit.
        """
        import harmonizepy.splitting as split_mod

        called_with_nan = [False]

        def tracking(data: np.ndarray, batch: np.ndarray, **kwargs: Any) -> np.ndarray:
            if np.isnan(data).any():
                called_with_nan[0] = True
            return _combat(data, batch, **kwargs)

        monkeypatch.setattr(split_mod, "combat", tracking)

        data = _make_data(6, 6)
        data.iloc[0, 0] = np.nan    # per-cell NaN, still qualifies for batch 1
        data.iloc[0, 3] = np.nan    # per-cell NaN, still qualifies for batch 2
        batch = np.array([1, 1, 1, 2, 2, 2])
        block = batch.copy()
        affil = build_affiliation_list(data, batch, block, needed_values=2)
        # Feature 0 still qualifies (2 non-NaN per batch)
        splitting(affil, data, batch, block, algorithm="ComBat", combat_mode=2)
        assert called_with_nan[0], "Engine should have received NaN data"

    def test_per_cell_nan_passed_to_limma(self, monkeypatch: Any) -> None:
        """Sub-DataFrames with per-cell NaN are passed to remove_batch_effect."""
        import harmonizepy.splitting as split_mod

        called_with_nan = [False]

        def tracking(data: np.ndarray, batch: np.ndarray) -> np.ndarray:
            if np.isnan(data).any():
                called_with_nan[0] = True
            return _remove_batch_effect(data, batch)

        monkeypatch.setattr(split_mod, "remove_batch_effect", tracking)

        data = _make_data(6, 6)
        data.iloc[0, 0] = np.nan
        data.iloc[0, 3] = np.nan
        batch = np.array([1, 1, 1, 2, 2, 2])
        block = batch.copy()
        affil = build_affiliation_list(data, batch, block, needed_values=2)
        splitting(affil, data, batch, block, algorithm="limma")
        assert called_with_nan[0], "Engine should have received NaN data"


class TestSplittingIntegration:
    def test_no_missing_data_produces_one_group(self) -> None:
        """Complete data yields a single affiliation group and one sub-DataFrame.

        Failure condition: complete data is split into multiple groups.
        """
        data = _make_data(6, 6)
        batch = np.array([1, 1, 1, 2, 2, 2])
        block = batch.copy()
        affil = build_affiliation_list(data, batch, block, needed_values=2)
        assert all(a == (1, 2) for a in affil)
        result = splitting(affil, data, batch, block, algorithm="limma")
        concat = pd.concat(result, axis=0)
        assert concat.shape == data.shape

    def test_partial_missing_produces_multiple_groups(self) -> None:
        """Partial missing data produces multiple affiliation groups.

        Failure condition: features with different missing patterns
        are forced into the same adjustment group.
        """
        data = _make_data(6, 6)
        data.iloc[0, 0:3] = np.nan
        data.iloc[1, 3:6] = np.nan
        batch = np.array([1, 1, 1, 2, 2, 2])
        block = batch.copy()
        affil = build_affiliation_list(data, batch, block, needed_values=2)
        assert affil[0] == (2,)
        assert affil[1] == (1,)
        assert all(a == (1, 2) for a in affil[2:])
        result = splitting(affil, data, batch, block, algorithm="ComBat", combat_mode=2)
        concat = pd.concat(result, axis=0)
        assert concat.shape == data.shape
