"""Tests for tsam_xarray.find_optimal_combination()."""

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

    def test_best_is_lowest_rmse(self):
        da = _make_da()
        result = tsam_xarray.find_optimal_combination(
            da,
            time_dim="time",
            cluster_dim="variable",
            data_reduction=0.05,
            show_progress=False,
        )
        valid = [h for h in result.history if h["rmse"] < float("inf")]
        min_rmse = min(h["rmse"] for h in valid)
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
