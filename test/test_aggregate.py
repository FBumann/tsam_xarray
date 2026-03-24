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
        da = _make_da().stack(column=["variable", "region"])
        result = tsam_xarray.aggregate(da, n_clusters=4)
        expected = {"cluster", "timestep", "variable", "region"}
        assert set(result.typical_periods.dims) == expected
        assert result.typical_periods.sizes["cluster"] == 4
        assert result.typical_periods.sizes["timestep"] == 24

    def test_cluster_weights_sum(self):
        da = _make_da().stack(column=["variable", "region"])
        result = tsam_xarray.aggregate(da, n_clusters=4)
        assert int(result.cluster_weights.sum()) == 30

    def test_cluster_assignments_shape(self):
        da = _make_da().stack(column=["variable", "region"])
        result = tsam_xarray.aggregate(da, n_clusters=4)
        assert result.cluster_assignments.dims == ("period",)
        assert result.cluster_assignments.sizes["period"] == 30

    def test_accuracy_dims(self):
        da = _make_da().stack(column=["variable", "region"])
        result = tsam_xarray.aggregate(da, n_clusters=4)
        for field in ("rmse", "mae", "rmse_duration"):
            metric = getattr(result.accuracy, field)
            assert set(metric.dims) == {"variable", "region"}

    def test_reconstructed_shape(self):
        da = _make_da().stack(column=["variable", "region"])
        result = tsam_xarray.aggregate(da, n_clusters=4)
        expected = {"time", "variable", "region"}
        assert set(result.reconstructed.dims) == expected
        assert result.reconstructed.sizes["time"] == da.sizes["time"]

    def test_raw_is_tsam_result(self):
        da = _make_da().stack(column=["variable", "region"])
        result = tsam_xarray.aggregate(da, n_clusters=4)
        from tsam.result import AggregationResult as TsamResult

        assert isinstance(result.raw, TsamResult)


class TestSimpleColumnDim:
    def test_single_dim(self):
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(da_flat, n_clusters=4)
        assert set(result.typical_periods.dims) == {
            "cluster",
            "timestep",
            "variable",
        }
        assert set(result.accuracy.rmse.dims) == {"variable"}

    def test_auto_detect_column_dim(self):
        """With 2 dims (time + one other), column_dim is auto-detected."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(da_flat, n_clusters=4)
        assert "variable" in result.typical_periods.dims


class TestNoColumnDim:
    def test_1d_time_series(self):
        time = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
        rng = np.random.default_rng(42)
        da = xr.DataArray(
            rng.random(len(time)),
            dims=["time"],
            coords={"time": time},
        )
        result = tsam_xarray.aggregate(da, n_clusters=4)
        assert set(result.typical_periods.dims) == {
            "cluster",
            "timestep",
        }
        assert result.typical_periods.sizes["cluster"] == 4


class TestAutoSliceDims:
    def test_extra_dims_auto_sliced(self):
        """Dims not time or column are automatically sliced."""
        da = _make_da(scenarios=["low", "high"]).stack(column=["variable", "region"])
        result = tsam_xarray.aggregate(da, n_clusters=4, column_dim="column")
        assert "scenario" in result.typical_periods.dims
        assert result.typical_periods.sizes["scenario"] == 2
        assert "scenario" in result.accuracy.rmse.dims

    def test_auto_slice_raw_is_dict(self):
        da = _make_da(scenarios=["low", "high"]).stack(column=["variable", "region"])
        result = tsam_xarray.aggregate(da, n_clusters=4, column_dim="column")
        assert isinstance(result.raw, dict)
        assert len(result.raw) == 2

    def test_multiple_auto_slice_dims(self):
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
        result = tsam_xarray.aggregate(da, n_clusters=4, column_dim="variable")
        assert "scenario" in result.typical_periods.dims
        assert "year" in result.typical_periods.dims
        assert result.typical_periods.sizes["scenario"] == 2
        assert result.typical_periods.sizes["year"] == 2


class TestDefaults:
    def test_default_time_dim(self):
        """time_dim defaults to 'time'."""
        da = _make_da().stack(column=["variable", "region"])
        result = tsam_xarray.aggregate(da, n_clusters=4)
        assert result.typical_periods.sizes["cluster"] == 4

    def test_explicit_time_dim(self):
        """Non-standard time dim name."""
        time = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
        rng = np.random.default_rng(42)
        da = xr.DataArray(
            rng.random((len(time), 2)),
            dims=["t", "variable"],
            coords={"t": time, "variable": ["solar", "wind"]},
        )
        result = tsam_xarray.aggregate(da, n_clusters=4, time_dim="t")
        assert result.typical_periods.sizes["cluster"] == 4


class TestWeights:
    def test_weights_passthrough(self):
        da = _make_da().stack(column=["variable", "region"])
        # tsam requires weights for ALL columns
        all_weights = {
            col: 2.0 if col == ("solar", "north") else 1.0
            for col in da.coords["column"].values.tolist()
        }
        result = tsam_xarray.aggregate(da, n_clusters=4, weights=all_weights)
        assert result.typical_periods.sizes["cluster"] == 4

    def test_weights_simple_column(self):
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            n_clusters=4,
            weights={"solar": 2.0, "wind": 1.0},
        )
        assert result.typical_periods.sizes["cluster"] == 4


class TestValidation:
    def test_invalid_time_dim(self):
        da = _make_da().stack(column=["variable", "region"])
        with pytest.raises(ValueError, match=r"time_dim.*not in"):
            tsam_xarray.aggregate(da, n_clusters=4, time_dim="nonexistent")

    def test_invalid_column_dim(self):
        da = _make_da().stack(column=["variable", "region"])
        with pytest.raises(ValueError, match="column_dim"):
            tsam_xarray.aggregate(da, n_clusters=4, column_dim="nonexistent")

    def test_column_dim_equals_time_dim(self):
        da = _make_da().stack(column=["variable", "region"])
        with pytest.raises(ValueError, match="different"):
            tsam_xarray.aggregate(da, n_clusters=4, time_dim="time", column_dim="time")

    def test_ambiguous_auto_detect(self):
        """Multiple non-time dims without column_dim raises."""
        da = _make_da()  # (time, variable, region) — ambiguous
        with pytest.raises(ValueError, match="multiple non-time"):
            tsam_xarray.aggregate(da, n_clusters=4)


class TestMultiIndexPassthrough:
    """Verify tsam preserves MultiIndex columns (undocumented behavior)."""

    def test_multiindex_columns_preserved(self):
        da = _make_da().stack(column=["variable", "region"])
        df = da.to_pandas()
        assert isinstance(df.columns, pd.MultiIndex)

        import tsam

        result = tsam.aggregate(df, n_clusters=4)
        assert isinstance(result.cluster_representatives.columns, pd.MultiIndex)
        assert isinstance(result.reconstructed.columns, pd.MultiIndex)

    def test_multiindex_accuracy_index_preserved(self):
        da = _make_da().stack(column=["variable", "region"])
        df = da.to_pandas()

        import tsam

        result = tsam.aggregate(df, n_clusters=4)
        assert isinstance(result.accuracy.rmse.index, pd.MultiIndex)
        assert isinstance(result.accuracy.mae.index, pd.MultiIndex)
