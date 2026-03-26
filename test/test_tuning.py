"""Tests for tsam_xarray tuning functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import tsam_xarray


def _make_da(
    n_days: int = 30,
    variables: list[str] | None = None,
    scenarios: list[str] | None = None,
) -> xr.DataArray:
    if variables is None:
        variables = ["solar", "wind"]
    time = pd.date_range("2020-01-01", periods=n_days * 24, freq="h")
    rng = np.random.default_rng(42)
    dims = ["time", "variable"]
    shape: list[int] = [len(time), len(variables)]
    coords: dict[str, list[str] | pd.DatetimeIndex] = {
        "time": time,
        "variable": variables,
    }
    if scenarios is not None:
        dims.append("scenario")
        shape.append(len(scenarios))
        coords["scenario"] = scenarios
    return xr.DataArray(rng.random(shape), dims=dims, coords=coords)


class TestFindOptimalCombination:
    def test_basic(self):
        da = _make_da()
        result = tsam_xarray.find_optimal_combination(
            da,
            time_dim="time",
            cluster_dim="variable",
            data_reduction=0.05,
            show_progress=False,
        )
        assert result.n_clusters >= 2
        assert result.n_segments >= 1
        assert result.rmse > 0
        assert result.best_result is not None

    def test_history_populated(self):
        da = _make_da()
        result = tsam_xarray.find_optimal_combination(
            da,
            time_dim="time",
            cluster_dim="variable",
            data_reduction=0.05,
            show_progress=False,
        )
        assert len(result.history) > 0
        assert all("rmse" in h for h in result.history)
        assert all("n_clusters" in h for h in result.history)
        assert all("n_segments" in h for h in result.history)

    def test_history_has_no_inf(self):
        """Failed configs should not appear in history (matches tsam)."""
        da = _make_da()
        result = tsam_xarray.find_optimal_combination(
            da,
            time_dim="time",
            cluster_dim="variable",
            data_reduction=0.05,
            show_progress=False,
        )
        assert all(np.isfinite(h["rmse"]) for h in result.history)

    def test_best_is_lowest_rmse(self):
        da = _make_da()
        result = tsam_xarray.find_optimal_combination(
            da,
            time_dim="time",
            cluster_dim="variable",
            data_reduction=0.05,
            show_progress=False,
        )
        min_rmse = min(h["rmse"] for h in result.history)
        np.testing.assert_allclose(result.rmse, min_rmse)

    def test_multi_dim_sliced(self):
        """Tuning works with auto-sliced dims."""
        da = _make_da(scenarios=["low", "high"])
        result = tsam_xarray.find_optimal_combination(
            da,
            time_dim="time",
            cluster_dim="variable",
            data_reduction=0.05,
            show_progress=False,
        )
        assert result.n_clusters >= 2
        assert "scenario" in result.best_result.cluster_representatives.dims

    def test_multi_cluster_dim(self):
        da = _make_da()
        da_3d = da.expand_dims({"region": ["north", "south"]})
        result = tsam_xarray.find_optimal_combination(
            da_3d,
            time_dim="time",
            cluster_dim=["variable", "region"],
            data_reduction=0.05,
            show_progress=False,
        )
        assert result.n_clusters >= 2
        assert "variable" in result.best_result.cluster_representatives.dims
        assert "region" in result.best_result.cluster_representatives.dims

    def test_summary(self):
        da = _make_da()
        result = tsam_xarray.find_optimal_combination(
            da,
            time_dim="time",
            cluster_dim="variable",
            data_reduction=0.05,
            show_progress=False,
        )
        summary = result.summary
        assert "rmse" in summary.columns
        assert len(summary) == len(result.history)

    def test_weights_affect_result(self):
        da = _make_da()
        r1 = tsam_xarray.find_optimal_combination(
            da,
            time_dim="time",
            cluster_dim="variable",
            data_reduction=0.05,
            show_progress=False,
        )
        r2 = tsam_xarray.find_optimal_combination(
            da,
            time_dim="time",
            cluster_dim="variable",
            data_reduction=0.05,
            weights={"solar": 10.0, "wind": 0.1},
            show_progress=False,
        )
        # Weighted should produce different RMSE
        assert r1.rmse != r2.rmse

    def test_invalid_data_reduction(self):
        da = _make_da()
        with pytest.raises(ValueError, match="No valid"):
            tsam_xarray.find_optimal_combination(
                da,
                time_dim="time",
                cluster_dim="variable",
                data_reduction=0.0001,
                show_progress=False,
            )


class TestFindBestCombination:
    def test_basic(self):
        da = _make_da()
        result = tsam_xarray.find_best_combination(
            da,
            time_dim="time",
            cluster_dim="variable",
            max_timesteps=48,
            show_progress=False,
        )
        assert result.n_clusters >= 2
        assert result.n_segments >= 1
        assert result.rmse > 0
        assert result.best_result is not None

    def test_history_is_unfiltered(self):
        """History should contain all tested combos, not just Pareto front."""
        da = _make_da()
        grid = tsam_xarray.find_best_combination(
            da,
            time_dim="time",
            cluster_dim="variable",
            max_timesteps=48,
            show_progress=False,
        )
        pareto = tsam_xarray.find_pareto_front(
            da,
            time_dim="time",
            cluster_dim="variable",
            max_timesteps=48,
            show_progress=False,
        )
        assert len(grid.history) >= len(pareto.history)

    def test_best_matches_pareto_best(self):
        """Grid search best should equal Pareto front best."""
        da = _make_da()
        grid = tsam_xarray.find_best_combination(
            da,
            time_dim="time",
            cluster_dim="variable",
            max_timesteps=48,
            show_progress=False,
        )
        pareto = tsam_xarray.find_pareto_front(
            da,
            time_dim="time",
            cluster_dim="variable",
            max_timesteps=48,
            show_progress=False,
        )
        np.testing.assert_allclose(grid.rmse, pareto.rmse)


class TestFindParetoFront:
    def test_basic(self):
        da = _make_da()
        result = tsam_xarray.find_pareto_front(
            da,
            time_dim="time",
            cluster_dim="variable",
            max_timesteps=48,
            show_progress=False,
        )
        assert result.n_clusters >= 2
        assert result.n_segments >= 1
        assert result.rmse > 0
        assert result.best_result is not None

    def test_pareto_front_is_non_dominated(self):
        """History should only contain Pareto-optimal points."""
        da = _make_da()
        result = tsam_xarray.find_pareto_front(
            da,
            time_dim="time",
            cluster_dim="variable",
            max_timesteps=48,
            show_progress=False,
        )
        # Pareto property: sorted by timesteps, RMSE must be strictly decreasing
        sorted_h = sorted(result.history, key=lambda h: h["timesteps"])
        for i in range(1, len(sorted_h)):
            assert sorted_h[i]["rmse"] < sorted_h[i - 1]["rmse"], (
                f"Pareto violation: timesteps {sorted_h[i]['timesteps']} has "
                f"RMSE {sorted_h[i]['rmse']} >= {sorted_h[i - 1]['rmse']}"
            )

    def test_history_has_no_inf(self):
        da = _make_da()
        result = tsam_xarray.find_pareto_front(
            da,
            time_dim="time",
            cluster_dim="variable",
            max_timesteps=48,
            show_progress=False,
        )
        assert all(np.isfinite(h["rmse"]) for h in result.history)

    def test_summary_matrix(self):
        da = _make_da()
        result = tsam_xarray.find_pareto_front(
            da,
            time_dim="time",
            cluster_dim="variable",
            max_timesteps=48,
            show_progress=False,
        )
        matrix = result.summary_matrix
        assert "rmse" in matrix
        assert "n_clusters" in matrix.dims
        assert "n_segments" in matrix.dims


class TestTuningResult:
    @pytest.fixture()
    def result_with_all(self):
        da = _make_da()
        return tsam_xarray.find_optimal_combination(
            da,
            time_dim="time",
            cluster_dim="variable",
            data_reduction=0.05,
            show_progress=False,
            save_all_results=True,
        )

    @pytest.fixture()
    def result_without_all(self):
        da = _make_da()
        return tsam_xarray.find_optimal_combination(
            da,
            time_dim="time",
            cluster_dim="variable",
            data_reduction=0.05,
            show_progress=False,
        )

    def test_find_by_timesteps(self, result_with_all):
        target = result_with_all.history[0]["timesteps"]
        agg = result_with_all.find_by_timesteps(target)
        assert agg is not None

    def test_find_by_timesteps_no_results(self, result_without_all):
        with pytest.raises(ValueError, match="No results available"):
            result_without_all.find_by_timesteps(10)

    def test_find_by_rmse(self, result_with_all):
        threshold = max(h["rmse"] for h in result_with_all.history)
        agg = result_with_all.find_by_rmse(threshold)
        assert agg is not None

    def test_find_by_rmse_impossible(self, result_with_all):
        with pytest.raises(ValueError, match="No configuration achieves"):
            result_with_all.find_by_rmse(0.0)

    def test_len_and_iter(self, result_with_all):
        assert len(result_with_all) == len(result_with_all.history)
        results_list = list(result_with_all)
        assert len(results_list) == len(result_with_all)

    def test_getitem(self, result_with_all):
        if len(result_with_all) > 0:
            agg = result_with_all[0]
            assert agg is not None
