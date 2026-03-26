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


@pytest.fixture(scope="module")
def da() -> xr.DataArray:
    return _make_da()


@pytest.fixture(scope="module")
def optimal_result(da):
    return tsam_xarray.find_optimal_combination(
        da,
        time_dim="time",
        cluster_dim="variable",
        data_reduction=0.05,
        show_progress=False,
    )


@pytest.fixture(scope="module")
def grid_result(da):
    return tsam_xarray.find_best_combination(
        da,
        time_dim="time",
        cluster_dim="variable",
        max_timesteps=48,
        show_progress=False,
    )


@pytest.fixture(scope="module")
def pareto_result(da):
    return tsam_xarray.find_pareto_front(
        da,
        time_dim="time",
        cluster_dim="variable",
        max_timesteps=48,
        show_progress=False,
    )


class TestFindOptimalCombination:
    def test_basic(self, optimal_result):
        assert optimal_result.n_clusters >= 2
        assert optimal_result.n_segments >= 1
        assert optimal_result.rmse > 0
        assert optimal_result.best_result is not None

    def test_history_populated(self, optimal_result):
        assert len(optimal_result.history) > 0
        assert all("rmse" in h for h in optimal_result.history)
        assert all("n_clusters" in h for h in optimal_result.history)
        assert all("n_segments" in h for h in optimal_result.history)

    def test_history_has_no_inf(self, optimal_result):
        """Failed configs should not appear in history (matches tsam)."""
        assert all(np.isfinite(h["rmse"]) for h in optimal_result.history)

    def test_best_is_lowest_rmse(self, optimal_result):
        min_rmse = min(h["rmse"] for h in optimal_result.history)
        np.testing.assert_allclose(optimal_result.rmse, min_rmse)

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

    def test_summary(self, optimal_result):
        summary = optimal_result.summary
        assert "rmse" in summary.columns
        assert len(summary) == len(optimal_result.history)

    def test_weights_affect_result(self, da, optimal_result):
        r2 = tsam_xarray.find_optimal_combination(
            da,
            time_dim="time",
            cluster_dim="variable",
            data_reduction=0.05,
            weights={"solar": 10.0, "wind": 0.1},
            show_progress=False,
        )
        # Weighted should produce different RMSE
        assert optimal_result.rmse != r2.rmse

    def test_invalid_data_reduction(self, da):
        with pytest.raises(ValueError, match="No valid"):
            tsam_xarray.find_optimal_combination(
                da,
                time_dim="time",
                cluster_dim="variable",
                data_reduction=0.0001,
                show_progress=False,
            )


class TestFindBestCombination:
    def test_basic(self, grid_result):
        assert grid_result.n_clusters >= 2
        assert grid_result.n_segments >= 1
        assert grid_result.rmse > 0
        assert grid_result.best_result is not None

    def test_history_is_unfiltered(self, grid_result, pareto_result):
        """History should contain all tested combos, not just Pareto front."""
        assert len(grid_result.history) >= len(pareto_result.history)

    def test_best_matches_pareto_best(self, grid_result, pareto_result):
        """Grid search best should equal Pareto front best."""
        np.testing.assert_allclose(grid_result.rmse, pareto_result.rmse)


class TestFindParetoFront:
    def test_basic(self, pareto_result):
        assert pareto_result.n_clusters >= 2
        assert pareto_result.n_segments >= 1
        assert pareto_result.rmse > 0
        assert pareto_result.best_result is not None

    def test_pareto_front_is_non_dominated(self, pareto_result):
        """History should only contain Pareto-optimal points."""
        sorted_h = sorted(pareto_result.history, key=lambda h: h["timesteps"])
        for i in range(1, len(sorted_h)):
            assert sorted_h[i]["rmse"] < sorted_h[i - 1]["rmse"], (
                f"Pareto violation: timesteps {sorted_h[i]['timesteps']} has "
                f"RMSE {sorted_h[i]['rmse']} >= {sorted_h[i - 1]['rmse']}"
            )

    def test_history_has_no_inf(self, pareto_result):
        assert all(np.isfinite(h["rmse"]) for h in pareto_result.history)

    def test_summary_matrix(self, pareto_result):
        matrix = pareto_result.summary_matrix
        assert "rmse" in matrix
        assert "n_clusters" in matrix.dims
        assert "n_segments" in matrix.dims


class TestTuningResult:
    @pytest.fixture(scope="class")
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

    @pytest.fixture(scope="class")
    def result_without_all(self):
        da = _make_da()
        return tsam_xarray.find_optimal_combination(
            da,
            time_dim="time",
            cluster_dim="variable",
            data_reduction=0.05,
            show_progress=False,
            save_all_results=False,
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

    def test_reconstructed_shape(self, result_with_all):
        rec = result_with_all.reconstructed
        assert "n_clusters" in rec.dims
        assert "n_segments" in rec.dims
        assert "time" in rec.dims
        assert "variable" in rec.dims

    def test_reconstructed_cached(self, result_with_all):
        rec1 = result_with_all.reconstructed
        rec2 = result_with_all.reconstructed
        assert rec1 is rec2

    def test_reconstructed_requires_all_results(self, result_without_all):
        with pytest.raises(ValueError, match="No results available"):
            _ = result_without_all.reconstructed

    def test_accuracy_shape(self, result_with_all):
        acc = result_with_all.accuracy
        assert "n_clusters" in acc.dims
        assert "n_segments" in acc.dims
        assert "variable" in acc.dims
        assert "rmse" in acc
        assert "mae" in acc
        assert "rmse_duration" in acc

    def test_accuracy_cached(self, result_with_all):
        acc1 = result_with_all.accuracy
        acc2 = result_with_all.accuracy
        assert acc1 is acc2

    def test_accuracy_requires_all_results(self, result_without_all):
        with pytest.raises(ValueError, match="No results available"):
            _ = result_without_all.accuracy
