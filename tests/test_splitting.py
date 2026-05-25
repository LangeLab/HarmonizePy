"""Direct unit tests for harmonizepy.splitting.

Tests the splitting logic in isolation: affiliation grouping, sub-matrix
extraction, adjustment dispatch, and NaN preservation.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from harmonizepy.affiliation import build_affiliation_list
from harmonizepy.combat_wrapper import adjust_combat as _adjust_combat
from harmonizepy.limma_wrapper import adjust_limma as _adjust_limma
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
        data = _make_data(4, 6)
        batch = np.array([1, 1, 1, 2, 2, 2])
        block = batch.copy()
        affil = _affil((1,), (1,), (1, 2), (1, 2))
        result = splitting(affil, data, batch, block, algorithm="limma")
        assert len(result) == 2
        concat = pd.concat(result, axis=0)
        assert concat.shape == data.shape

    def test_empty_affiliation_stays_nan(self) -> None:
        data = _make_data(4, 6)
        batch = np.array([1, 1, 1, 2, 2, 2])
        block = batch.copy()
        affil = _affil((), (1, 2), (1, 2), (1, 2))
        result = splitting(affil, data, batch, block, algorithm="limma")
        concat = pd.concat(result, axis=0)
        assert concat.index[0] == data.index[0]
        assert concat.iloc[0].isna().all()

    def test_single_batch_group_copies_raw(self) -> None:
        data = _make_data(4, 6)
        batch = np.array([1, 1, 1, 2, 2, 2])
        block = batch.copy()
        affil = _affil((1,), (1,), (1,), (1,))
        result = splitting(affil, data, batch, block, algorithm="ComBat", combat_mode=2)
        concat = pd.concat(result, axis=0)
        np.testing.assert_allclose(concat.values[:, :3], data.values[:, :3])
        assert concat.iloc[:, 3:].isna().all().all()

    def test_single_feature_group_copies_raw(self) -> None:
        data = _make_data(4, 6)
        batch = np.array([1, 1, 1, 2, 2, 2])
        block = batch.copy()
        affil = _affil((1,), (1, 2), (1, 2), (1, 2))
        result = splitting(affil, data, batch, block, algorithm="ComBat", combat_mode=2)
        concat = pd.concat(result, axis=0)
        assert concat.shape == data.shape


class TestSplittingNanAudit:
    def test_combat_subframes_are_nan_free(self, monkeypatch: Any) -> None:
        import harmonizepy.splitting as splitting_mod

        called_with_nan = False

        def tracking(sub_df: pd.DataFrame, batch_labels: np.ndarray, **kwargs: Any) -> pd.DataFrame:
            nonlocal called_with_nan
            if sub_df.isna().any().any():
                called_with_nan = True
            return _adjust_combat(sub_df, batch_labels, **kwargs)

        monkeypatch.setattr(splitting_mod, "adjust_combat", tracking)

        data = _make_data(6, 6)
        data.iloc[0, 0:3] = np.nan
        data.iloc[1, 3:6] = np.nan
        batch = np.array([1, 1, 1, 2, 2, 2])
        block = batch.copy()
        affil = build_affiliation_list(data, batch, block, needed_values=2)
        splitting(affil, data, batch, block, algorithm="ComBat", combat_mode=2)
        assert not called_with_nan

    def test_limma_subframes_are_nan_free(self, monkeypatch: Any) -> None:
        import harmonizepy.splitting as splitting_mod

        called_with_nan = False

        def tracking(sub_df: pd.DataFrame, batch_labels: np.ndarray) -> pd.DataFrame:
            nonlocal called_with_nan
            if sub_df.isna().any().any():
                called_with_nan = True
            return _adjust_limma(sub_df, batch_labels)

        monkeypatch.setattr(splitting_mod, "adjust_limma", tracking)

        data = _make_data(6, 6)
        data.iloc[0, 0:3] = np.nan
        data.iloc[1, 3:6] = np.nan
        batch = np.array([1, 1, 1, 2, 2, 2])
        block = batch.copy()
        affil = build_affiliation_list(data, batch, block, needed_values=2)
        splitting(affil, data, batch, block, algorithm="limma")
        assert not called_with_nan


class TestSplittingIntegration:
    def test_no_missing_data_produces_one_group(self) -> None:
        data = _make_data(6, 6)
        batch = np.array([1, 1, 1, 2, 2, 2])
        block = batch.copy()
        affil = build_affiliation_list(data, batch, block, needed_values=2)
        assert all(a == (1, 2) for a in affil)
        result = splitting(affil, data, batch, block, algorithm="limma")
        concat = pd.concat(result, axis=0)
        assert concat.shape == data.shape

    def test_partial_missing_produces_multiple_groups(self) -> None:
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
