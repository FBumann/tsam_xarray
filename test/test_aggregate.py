"""Tests for tsam_xarray.aggregate()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import tsam_xarray


def _make_da(
    n_days: int = 30,
    variables: list[str] | None = None,
    regions: list[str] | None = None,
    scenarios: list[str] | None = None,
) -> xr.DataArray:
    """Create a synthetic DataArray for testing."""
    if variables is None:
        variables = ["solar", "wind"]
    if regions is None:
        regions = ["north", "south"]

    time = pd.date_range("2020-01-01", periods=n_days * 24, freq="h")
    rng = np.random.default_rng(42)

    dims = ["time", "variable", "region"]
    shape: list[int] = [len(time), len(variables), len(regions)]
    coords: dict[str, list[str] | pd.DatetimeIndex] = {
        "time": time,
        "variable": variables,
        "region": regions,
    }

    if scenarios is not None:
        dims.append("scenario")
        shape.append(len(scenarios))
        coords["scenario"] = scenarios

    return xr.DataArray(rng.random(shape), dims=dims, coords=coords)


class TestBasicRoundtrip:
    def test_dims_and_shapes(self):
        da = _make_da()
        result = tsam_xarray.aggregate(
            da,
            n_clusters=4,
            time_dim="time",
            stack_dims=["variable", "region"],
        )
        expected = {"cluster", "timestep", "variable", "region"}
        assert set(result.typical_periods.dims) == expected
        assert result.typical_periods.sizes["cluster"] == 4
        assert result.typical_periods.sizes["timestep"] == 24

    def test_cluster_weights_sum(self):
        da = _make_da()
        result = tsam_xarray.aggregate(
            da,
            n_clusters=4,
            time_dim="time",
            stack_dims=["variable", "region"],
        )
        assert int(result.cluster_weights.sum()) == 30

    def test_cluster_assignments_shape(self):
        da = _make_da()
        result = tsam_xarray.aggregate(
            da,
            n_clusters=4,
            time_dim="time",
            stack_dims=["variable", "region"],
        )
        assert result.cluster_assignments.dims == ("period",)
        assert result.cluster_assignments.sizes["period"] == 30

    def test_accuracy_dims(self):
        da = _make_da()
        result = tsam_xarray.aggregate(
            da,
            n_clusters=4,
            time_dim="time",
            stack_dims=["variable", "region"],
        )
        for field in ("rmse", "mae", "rmse_duration"):
            metric = getattr(result.accuracy, field)
            assert set(metric.dims) == {"variable", "region"}

    def test_reconstructed_shape(self):
        da = _make_da()
        result = tsam_xarray.aggregate(
            da,
            n_clusters=4,
            time_dim="time",
            stack_dims=["variable", "region"],
        )
        expected = {"time", "variable", "region"}
        assert set(result.reconstructed.dims) == expected
        assert result.reconstructed.sizes["time"] == da.sizes["time"]

    def test_raw_is_tsam_result(self):
        da = _make_da()
        result = tsam_xarray.aggregate(
            da,
            n_clusters=4,
            time_dim="time",
            stack_dims=["variable", "region"],
        )
        from tsam.result import AggregationResult as TsamResult

        assert isinstance(result.raw, TsamResult)


class TestSingleStackDim:
    def test_one_stack_dim(self):
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            n_clusters=4,
            time_dim="time",
            stack_dims=["variable"],
        )
        assert set(result.typical_periods.dims) == {
            "cluster",
            "timestep",
            "variable",
        }
        assert set(result.accuracy.rmse.dims) == {"variable"}


class TestNoStackDims:
    def test_1d_time_series(self):
        time = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
        rng = np.random.default_rng(42)
        da = xr.DataArray(
            rng.random(len(time)),
            dims=["time"],
            coords={"time": time},
        )
        result = tsam_xarray.aggregate(da, n_clusters=4, time_dim="time")
        assert set(result.typical_periods.dims) == {
            "cluster",
            "timestep",
        }
        assert result.typical_periods.sizes["cluster"] == 4


class TestSliceDims:
    def test_slice_dims_basic(self):
        da = _make_da(scenarios=["low", "high"])
        result = tsam_xarray.aggregate(
            da,
            n_clusters=4,
            time_dim="time",
            stack_dims=["variable", "region"],
            slice_dims=["scenario"],
        )
        assert "scenario" in result.typical_periods.dims
        assert result.typical_periods.sizes["scenario"] == 2
        assert "scenario" in result.accuracy.rmse.dims

    def test_slice_dims_raw_is_dict(self):
        da = _make_da(scenarios=["low", "high"])
        result = tsam_xarray.aggregate(
            da,
            n_clusters=4,
            time_dim="time",
            stack_dims=["variable", "region"],
            slice_dims=["scenario"],
        )
        assert isinstance(result.raw, dict)
        assert len(result.raw) == 2


    def test_multiple_slice_dims(self):
        time = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
        rng = np.random.default_rng(42)
        da = xr.DataArray(
            rng.random((len(time), 2, 2, 2)),
            dims=["time", "variable", "scenario", "year"],
            coords={
                "time": time,
                "variable": ["solar", "wind"],
                "scenario": ["low", "high"],
                "year": [2020, 2021],
            },
        )
        result = tsam_xarray.aggregate(
            da,
            n_clusters=4,
            time_dim="time",
            stack_dims=["variable"],
            slice_dims=["scenario", "year"],
        )
        assert "scenario" in result.typical_periods.dims
        assert "year" in result.typical_periods.dims
        assert result.typical_periods.sizes["scenario"] == 2
        assert result.typical_periods.sizes["year"] == 2


class TestWeights:
    def test_weights_passthrough(self):
        da = _make_da()
        result = tsam_xarray.aggregate(
            da,
            n_clusters=4,
            time_dim="time",
            stack_dims=["variable", "region"],
            weights={"variable": {"solar": 2.0}},
        )
        assert result.typical_periods.sizes["cluster"] == 4


    def test_weights_with_numeric_coords(self):
        time = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
        rng = np.random.default_rng(42)
        da = xr.DataArray(
            rng.random((len(time), 3)),
            dims=["time", "level"],
            coords={"time": time, "level": [1, 2, 3]},
        )
        result = tsam_xarray.aggregate(
            da,
            n_clusters=4,
            time_dim="time",
            stack_dims=["level"],
            weights={"level": {"1": 2.0}},
        )
        assert result.typical_periods.sizes["cluster"] == 4


class TestValidation:
    def test_invalid_time_dim(self):
        da = _make_da()
        with pytest.raises(ValueError, match="time_dim"):
            tsam_xarray.aggregate(
                da,
                n_clusters=4,
                time_dim="nonexistent",
                stack_dims=["variable", "region"],
            )

    def test_invalid_stack_dim(self):
        da = _make_da()
        with pytest.raises(ValueError, match="stack_dims"):
            tsam_xarray.aggregate(
                da,
                n_clusters=4,
                time_dim="time",
                stack_dims=["nonexistent", "region"],
            )

    def test_time_dim_in_stack_dims(self):
        da = _make_da()
        with pytest.raises(ValueError, match="overlap"):
            tsam_xarray.aggregate(
                da,
                n_clusters=4,
                time_dim="time",
                stack_dims=["time", "region"],
            )

    def test_unaccounted_dims(self):
        da = _make_da()
        with pytest.raises(ValueError, match="not in time_dim"):
            tsam_xarray.aggregate(
                da,
                n_clusters=4,
                time_dim="time",
                stack_dims=["variable"],
            )

    def test_invalid_weight_dim(self):
        da = _make_da()
        with pytest.raises(ValueError, match="not in"):
            tsam_xarray.aggregate(
                da,
                n_clusters=4,
                time_dim="time",
                stack_dims=["variable", "region"],
                weights={"nonexistent": {"x": 1.0}},
            )

    def test_invalid_weight_coord(self):
        da = _make_da()
        with pytest.raises(ValueError, match="not in"):
            tsam_xarray.aggregate(
                da,
                n_clusters=4,
                time_dim="time",
                stack_dims=["variable", "region"],
                weights={"variable": {"nonexistent": 1.0}},
            )
