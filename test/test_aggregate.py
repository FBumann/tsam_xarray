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
            cluster_dim=["variable", "region"],
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
            cluster_dim=["variable", "region"],
        )
        assert int(result.cluster_weights.sum()) == 30

    def test_cluster_assignments_shape(self):
        da = _make_da()
        result = tsam_xarray.aggregate(
            da,
            n_clusters=4,
            time_dim="time",
            cluster_dim=["variable", "region"],
        )
        assert result.cluster_assignments.dims == ("period",)
        assert result.cluster_assignments.sizes["period"] == 30

    def test_accuracy_dims(self):
        da = _make_da()
        result = tsam_xarray.aggregate(
            da,
            n_clusters=4,
            time_dim="time",
            cluster_dim=["variable", "region"],
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
            cluster_dim=["variable", "region"],
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
            cluster_dim=["variable", "region"],
        )
        assert result.clustering is not None
        assert result.clustering.time_dim == "time"
        assert result.clustering.cluster_dim == ["variable", "region"]


class TestSingleClusterDim:
    def test_single_dim(self):
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            n_clusters=4,
            time_dim="time",
            cluster_dim="variable",
        )
        assert set(result.typical_periods.dims) == {
            "cluster",
            "timestep",
            "variable",
        }
        assert set(result.accuracy.rmse.dims) == {"variable"}

    def test_string_cluster_dim(self):
        """cluster_dim accepts a single string."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            n_clusters=4,
            time_dim="time",
            cluster_dim="variable",
        )
        assert "variable" in result.typical_periods.dims


class TestAutoSliceDims:
    def test_extra_dims_auto_sliced(self):
        """Dims not in time or cluster_dim are automatically sliced."""
        da = _make_da(scenarios=["low", "high"])
        result = tsam_xarray.aggregate(
            da,
            n_clusters=4,
            time_dim="time",
            cluster_dim=["variable", "region"],
        )
        assert "scenario" in result.typical_periods.dims
        assert result.typical_periods.sizes["scenario"] == 2
        assert "scenario" in result.accuracy.rmse.dims

    def test_auto_slice_clustering_has_per_slice_keys(self):
        da = _make_da(scenarios=["low", "high"])
        result = tsam_xarray.aggregate(
            da,
            n_clusters=4,
            time_dim="time",
            cluster_dim=["variable", "region"],
        )
        assert len(result.clustering.clusterings) == 2

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
        result = tsam_xarray.aggregate(
            da,
            n_clusters=4,
            time_dim="time",
            cluster_dim="variable",
        )
        assert "scenario" in result.typical_periods.dims
        assert "year" in result.typical_periods.dims
        assert result.typical_periods.sizes["scenario"] == 2
        assert result.typical_periods.sizes["year"] == 2


class TestExplicitTimeDim:
    def test_non_standard_time_dim(self):
        time = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
        rng = np.random.default_rng(42)
        da = xr.DataArray(
            rng.random((len(time), 2)),
            dims=["t", "variable"],
            coords={"t": time, "variable": ["solar", "wind"]},
        )
        result = tsam_xarray.aggregate(
            da, n_clusters=4, time_dim="t", cluster_dim="variable"
        )
        assert result.typical_periods.sizes["cluster"] == 4


class TestWeights:
    def test_weights_simple_dict(self):
        """Simple dict weights produce valid result."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
            weights={"solar": 2.0, "wind": 1.0},
        )
        assert result.n_clusters == 4

    def test_weights_dict_of_dicts(self):
        """Dict-of-dicts weights produce valid result."""
        da = _make_da()
        result = tsam_xarray.aggregate(
            da,
            time_dim="time",
            cluster_dim=["variable", "region"],
            n_clusters=4,
            weights={"variable": {"solar": 2.0}, "region": {"north": 1.5}},
        )
        assert result.n_clusters == 4

    def test_weights_affect_clustering(self):
        """Weighted and unweighted produce different results."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result_no_w = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        result_w = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
            weights={"solar": 10.0, "wind": 0.1},
        )
        rmse_no_w = result_no_w.accuracy.rmse
        rmse_w = result_w.accuracy.rmse
        assert float(rmse_w.sel(variable="solar")) < float(
            rmse_no_w.sel(variable="solar")
        )

    def test_weights_rejects_simple_dict_multi_dim(self):
        """Simple dict with multiple cluster_dim raises."""
        da = _make_da()
        with pytest.raises(ValueError, match="single cluster_dim"):
            tsam_xarray.aggregate(
                da,
                time_dim="time",
                cluster_dim=["variable", "region"],
                n_clusters=4,
                weights={"solar": 2.0},
            )

    def test_weights_rejects_unknown_dim(self):
        """Dict-of-dicts with unknown dim key raises."""
        da = _make_da()
        with pytest.raises(ValueError, match="unknown dims"):
            tsam_xarray.aggregate(
                da,
                time_dim="time",
                cluster_dim=["variable", "region"],
                n_clusters=4,
                weights={"scenario": {"low": 2.0}},
            )

    def test_weights_rejects_unknown_coord(self):
        """Weights with unknown coord values raise."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        with pytest.raises(ValueError, match="unknown coords"):
            tsam_xarray.aggregate(
                da_flat,
                time_dim="time",
                cluster_dim="variable",
                n_clusters=4,
                weights={"solar": 2.0, "nuclear": 3.0},
            )

    def test_weights_rejects_mixed_dict(self):
        """Mixed dict values (some dicts, some floats) raise."""
        da = _make_da()
        with pytest.raises(ValueError, match="Mixed weights"):
            tsam_xarray.aggregate(
                da,
                time_dim="time",
                cluster_dim=["variable", "region"],
                n_clusters=4,
                weights={"variable": {"solar": 2.0}, "region": 1.5},  # type: ignore[dict-item]
            )


class TestWeightTranslation:
    """Verify weight values are correctly translated."""

    def test_single_dim_values(self):
        from tsam_xarray._core import _translate_weights

        w = {"variable": {"solar": 2.0, "wind": 0.5}}
        df = pd.DataFrame(
            {"solar": [1.0], "wind": [2.0]},
            index=pd.date_range("2020-01-01", periods=1, freq="h"),
        )
        flat = _translate_weights(w, df, ["variable"])
        assert flat["solar"] == 2.0
        assert flat["wind"] == 0.5

    def test_multi_dim_broadcast(self):
        """Single-dim weights broadcast across other dims."""
        from tsam_xarray._core import _translate_weights

        w = {"variable": {"solar": 2.0, "wind": 1.0}}
        cols = pd.MultiIndex.from_tuples(
            [
                ("solar", "north"),
                ("solar", "south"),
                ("wind", "north"),
                ("wind", "south"),
            ],
            names=["variable", "region"],
        )
        df = pd.DataFrame(
            [[1, 2, 3, 4]],
            columns=cols,
            index=pd.date_range("2020-01-01", periods=1, freq="h"),
        )
        flat = _translate_weights(w, df, ["variable", "region"])
        assert flat[("solar", "north")] == 2.0
        assert flat[("solar", "south")] == 2.0
        assert flat[("wind", "north")] == 1.0
        assert flat[("wind", "south")] == 1.0

    def test_multi_dim_full(self):
        """Full multi-dim weights are multiplied."""
        from tsam_xarray._core import _translate_weights

        w = {
            "variable": {"solar": 3.0, "wind": 1.0},
            "region": {"north": 1.5, "south": 2.0},
        }
        cols = pd.MultiIndex.from_tuples(
            [
                ("solar", "north"),
                ("solar", "south"),
                ("wind", "north"),
                ("wind", "south"),
            ],
            names=["variable", "region"],
        )
        df = pd.DataFrame(
            [[1, 2, 3, 4]],
            columns=cols,
            index=pd.date_range("2020-01-01", periods=1, freq="h"),
        )
        flat = _translate_weights(w, df, ["variable", "region"])
        assert flat[("solar", "north")] == 4.5  # 3.0 * 1.5
        assert flat[("solar", "south")] == 6.0  # 3.0 * 2.0
        assert flat[("wind", "north")] == 1.5  # 1.0 * 1.5
        assert flat[("wind", "south")] == 2.0  # 1.0 * 2.0

    def test_missing_coords_default_to_one(self):
        """Coords not in weights get weight 1.0."""
        from tsam_xarray._core import _translate_weights

        w = {"variable": {"solar": 3.0}}
        df = pd.DataFrame(
            {"solar": [1.0], "wind": [2.0]},
            index=pd.date_range("2020-01-01", periods=1, freq="h"),
        )
        flat = _translate_weights(w, df, ["variable"])
        assert flat["solar"] == 3.0
        assert flat["wind"] == 1.0


class TestValidation:
    def test_invalid_time_dim(self):
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        with pytest.raises(ValueError, match=r"time_dim.*not in"):
            tsam_xarray.aggregate(
                da_flat,
                n_clusters=4,
                time_dim="nonexistent",
                cluster_dim="variable",
            )

    def test_invalid_cluster_dim(self):
        da = _make_da()
        with pytest.raises(ValueError, match="cluster_dim"):
            tsam_xarray.aggregate(
                da,
                n_clusters=4,
                time_dim="time",
                cluster_dim=["nonexistent"],
            )

    def test_cluster_dim_overlaps_time(self):
        da = _make_da()
        with pytest.raises(ValueError, match="overlap"):
            tsam_xarray.aggregate(
                da,
                n_clusters=4,
                time_dim="time",
                cluster_dim=["time", "variable"],
            )

    def test_cluster_config_weights_rejected(self):
        """ClusterConfig.weights is deprecated and not supported."""
        from tsam import ClusterConfig

        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        with pytest.raises(ValueError, match=r"ClusterConfig\.weights"):
            tsam_xarray.aggregate(
                da_flat,
                n_clusters=4,
                time_dim="time",
                cluster_dim="variable",
                cluster=ClusterConfig(weights={"solar": 2.0}),
            )


class TestMultiIndexPassthrough:
    """Verify tsam preserves MultiIndex columns."""

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


class TestSegmentation:
    def test_segment_durations_shape(self):
        """segment_durations has (cluster, timestep) dims."""
        from tsam import SegmentConfig

        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
            segments=SegmentConfig(n_segments=6),
        )
        assert result.segment_durations is not None
        assert set(result.segment_durations.dims) == {
            "cluster",
            "timestep",
        }
        assert result.segment_durations.sizes["cluster"] == 4
        assert result.segment_durations.sizes["timestep"] == 6

    def test_segment_durations_sum_to_period(self):
        """Each cluster's durations sum to timesteps per period."""
        from tsam import SegmentConfig

        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
            segments=SegmentConfig(n_segments=6),
        )
        assert result.segment_durations is not None
        # Each cluster's durations should sum to 24 (hours per day)
        for c in range(4):
            total = int(result.segment_durations.sel(cluster=c).sum())
            assert total == 24

    def test_no_segmentation_returns_none(self):
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        assert result.segment_durations is None

    def test_typical_periods_with_segments(self):
        """typical_periods timestep dim equals n_segments."""
        from tsam import SegmentConfig

        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
            segments=SegmentConfig(n_segments=6),
        )
        assert result.typical_periods.sizes["timestep"] == 6


class TestDisaggregate:
    def test_roundtrip_no_segments(self):
        """disaggregate(typical_periods) matches reconstructed."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        dis = result.disaggregate(result.typical_periods)
        xr.testing.assert_allclose(dis, result.reconstructed)

    def test_roundtrip_with_segments_ffill(self):
        """disaggregate + ffill matches reconstructed (segmented)."""
        from tsam import SegmentConfig

        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
            segments=SegmentConfig(n_segments=6),
        )
        dis = result.disaggregate(result.typical_periods)
        filled = dis.ffill(dim="time")
        xr.testing.assert_allclose(filled, result.reconstructed)

    def test_disaggregate_dims(self):
        """disaggregate replaces cluster+timestep with time."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        dis = result.disaggregate(result.typical_periods)
        assert "time" in dis.dims
        assert "cluster" not in dis.dims
        assert "timestep" not in dis.dims
        assert "variable" in dis.dims

    def test_roundtrip_multi_dim(self):
        """disaggregate works with auto-sliced dims."""
        da = _make_da(scenarios=["low", "high"])
        result = tsam_xarray.aggregate(
            da,
            time_dim="time",
            cluster_dim=["variable", "region"],
            n_clusters=4,
        )
        dis = result.disaggregate(result.typical_periods)
        assert dis.dims == result.reconstructed.dims
        np.testing.assert_allclose(dis.values, result.reconstructed.values)

    def test_roundtrip_multi_dim_with_segments(self):
        """disaggregate + ffill with sliced segmented data."""
        from tsam import SegmentConfig

        da = _make_da(scenarios=["low", "high"])
        result = tsam_xarray.aggregate(
            da,
            time_dim="time",
            cluster_dim=["variable", "region"],
            n_clusters=4,
            segments=SegmentConfig(n_segments=6),
        )
        dis = result.disaggregate(result.typical_periods)
        filled = dis.ffill(dim="time")
        assert filled.dims == result.reconstructed.dims
        np.testing.assert_allclose(filled.values, result.reconstructed.values)

    def test_disaggregate_segmented_has_nan(self):
        """Segmented disaggregate has NaN at non-boundary timesteps."""
        from tsam import SegmentConfig

        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
            segments=SegmentConfig(n_segments=6),
        )
        dis = result.disaggregate(result.typical_periods)
        assert bool(dis.isnull().any())


class Test1DDataArray:
    def test_1d_time_series(self):
        """1D DataArray with only time dim works."""
        time = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
        rng = np.random.default_rng(42)
        da = xr.DataArray(
            rng.random(len(time)),
            dims=["time"],
            coords={"time": time},
        )
        result = tsam_xarray.aggregate(
            da, time_dim="time", cluster_dim=(), n_clusters=4
        )
        assert set(result.typical_periods.dims) == {
            "cluster",
            "timestep",
        }
        assert result.n_clusters == 4

    def test_1d_disaggregate(self):
        """disaggregate works on 1D result."""
        time = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
        rng = np.random.default_rng(42)
        da = xr.DataArray(
            rng.random(len(time)),
            dims=["time"],
            coords={"time": time},
        )
        result = tsam_xarray.aggregate(
            da, time_dim="time", cluster_dim=(), n_clusters=4
        )
        dis = result.disaggregate(result.typical_periods)
        assert "time" in dis.dims
        assert dis.sizes["time"] == da.sizes["time"]


class TestDataValidation:
    def test_reserved_dim_name_cluster(self):
        time = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
        da = xr.DataArray(
            np.random.default_rng(42).random((len(time), 2)),
            dims=["time", "cluster"],
            coords={"time": time, "cluster": ["a", "b"]},
        )
        with pytest.raises(ValueError, match="reserved"):
            tsam_xarray.aggregate(
                da, time_dim="time", cluster_dim="cluster", n_clusters=4
            )

    def test_reserved_dim_name_timestep(self):
        time = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
        da = xr.DataArray(
            np.random.default_rng(42).random((len(time), 2)),
            dims=["time", "timestep"],
            coords={"time": time, "timestep": [0, 1]},
        )
        with pytest.raises(ValueError, match="reserved"):
            tsam_xarray.aggregate(
                da, time_dim="time", cluster_dim="timestep", n_clusters=4
            )

    def test_reserved_dim_name_period(self):
        time = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
        da = xr.DataArray(
            np.random.default_rng(42).random((len(time), 2, 2)),
            dims=["time", "variable", "period"],
            coords={
                "time": time,
                "variable": ["a", "b"],
                "period": [0, 1],
            },
        )
        with pytest.raises(ValueError, match="reserved"):
            tsam_xarray.aggregate(
                da,
                time_dim="time",
                cluster_dim="variable",
                n_clusters=4,
            )

    def test_dask_array_warns(self):
        pytest.importorskip("dask")
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        da_dask = da_flat.chunk({"time": 100})
        with pytest.warns(UserWarning, match="dask"):
            result = tsam_xarray.aggregate(
                da_dask,
                time_dim="time",
                cluster_dim="variable",
                n_clusters=4,
            )
        assert not hasattr(result.typical_periods.data, "dask")

    def test_nan_rejected(self):
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        da_nan = da_flat.copy()
        da_nan.values[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            tsam_xarray.aggregate(
                da_nan, time_dim="time", cluster_dim="variable", n_clusters=4
            )

    def test_non_numeric_dtype(self):
        time = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
        da = xr.DataArray(
            [["a", "b"]] * len(time),
            dims=["time", "variable"],
            coords={"time": time, "variable": ["x", "y"]},
        )
        with pytest.raises(TypeError, match="numeric"):
            tsam_xarray.aggregate(
                da, time_dim="time", cluster_dim="variable", n_clusters=4
            )

    def test_non_datetime_time_coord(self):
        da = xr.DataArray(
            np.random.default_rng(42).random((100, 2)),
            dims=["time", "variable"],
            coords={"time": np.arange(100), "variable": ["a", "b"]},
        )
        with pytest.raises(TypeError, match="datetime"):
            tsam_xarray.aggregate(
                da, time_dim="time", cluster_dim="variable", n_clusters=4
            )

    def test_irregular_time_spacing(self):
        times = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04"])
        da = xr.DataArray(
            np.random.default_rng(42).random((3, 2)),
            dims=["time", "variable"],
            coords={"time": times, "variable": ["a", "b"]},
        )
        with pytest.raises(ValueError, match="regular spacing"):
            tsam_xarray.aggregate(
                da, time_dim="time", cluster_dim="variable", n_clusters=2
            )


class TestClusteringIO:
    """Tests for save_clustering / load_clustering / apply."""

    def test_save_load_metadata(self, tmp_path):
        """JSON preserves time_dim and cluster_dim."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        assert clustering.time_dim == "time"
        assert clustering.cluster_dim == ["variable"]
        assert () in clustering.clusterings

    def test_json_file_is_valid(self, tmp_path):
        """Saved file is valid JSON with expected keys."""
        import json

        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        with open(path) as f:
            data = json.load(f)
        assert "time_dim" in data
        assert "cluster_dim" in data
        assert "clusterings" in data
        assert data["time_dim"] == "time"
        assert data["cluster_dim"] == ["variable"]

    def test_apply_same_data_matches_cluster_assignments(self, tmp_path):
        """Applying clustering to same data gives same assignments."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        new_result = clustering.apply(da_flat)
        np.testing.assert_array_equal(
            result.cluster_assignments.values,
            new_result.cluster_assignments.values,
        )

    def test_apply_same_data_matches_typical_periods(self, tmp_path):
        """Applying clustering to same data reproduces typical periods."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        new_result = clustering.apply(da_flat)
        xr.testing.assert_allclose(result.typical_periods, new_result.typical_periods)

    def test_apply_same_data_matches_accuracy(self, tmp_path):
        """Applying clustering to same data gives same accuracy."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        new_result = clustering.apply(da_flat)
        xr.testing.assert_allclose(result.accuracy.rmse, new_result.accuracy.rmse)

    def test_apply_different_data(self, tmp_path):
        """Apply to different data produces valid result."""
        da1 = _make_da(n_days=30)
        da1_flat = da1.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da1_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))

        # Apply to data with different values but same shape
        rng = np.random.default_rng(99)
        da2_flat = da1_flat.copy(data=rng.random(da1_flat.shape))
        new_result = clustering.apply(da2_flat)
        # Same structure, different values
        assert new_result.n_clusters == 4
        assert new_result.typical_periods.dims == result.typical_periods.dims
        assert new_result.is_transferred

    def test_apply_marks_is_transferred(self, tmp_path):
        """Applied results have is_transferred=True."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        assert not result.is_transferred
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        new_result = clustering.apply(da_flat)
        assert new_result.is_transferred

    def test_save_load_sliced_keys(self, tmp_path):
        """Sliced clustering preserves per-slice keys."""
        da = _make_da(scenarios=["low", "high"])
        result = tsam_xarray.aggregate(
            da,
            time_dim="time",
            cluster_dim=["variable", "region"],
            n_clusters=4,
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        assert len(clustering.clusterings) == 2
        assert ("low",) in clustering.clusterings
        assert ("high",) in clustering.clusterings

    def test_apply_sliced_dims(self, tmp_path):
        """Apply sliced clustering produces correct dims."""
        da = _make_da(scenarios=["low", "high"])
        result = tsam_xarray.aggregate(
            da,
            time_dim="time",
            cluster_dim=["variable", "region"],
            n_clusters=4,
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        new_result = clustering.apply(da)
        assert "scenario" in new_result.typical_periods.dims
        assert new_result.n_clusters == 4
        assert new_result.typical_periods.sizes["scenario"] == 2

    def test_apply_sliced_values_match(self, tmp_path):
        """Apply sliced clustering to same data reproduces values."""
        da = _make_da(scenarios=["low", "high"])
        result = tsam_xarray.aggregate(
            da,
            time_dim="time",
            cluster_dim=["variable", "region"],
            n_clusters=4,
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        new_result = clustering.apply(da)
        np.testing.assert_allclose(
            result.typical_periods.values,
            new_result.typical_periods.values,
        )

    def test_clustering_property_single(self):
        """clustering property returns correct ClusteringInfo."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        ci = result.clustering
        assert ci.time_dim == "time"
        assert ci.cluster_dim == ["variable"]
        assert len(ci.clusterings) == 1

    def test_clustering_property_sliced(self):
        """clustering property returns per-slice clusterings."""
        da = _make_da(scenarios=["low", "high"])
        result = tsam_xarray.aggregate(
            da,
            time_dim="time",
            cluster_dim=["variable", "region"],
            n_clusters=4,
        )
        ci = result.clustering
        assert len(ci.clusterings) == 2

    def test_apply_without_save(self):
        """apply() works directly from clustering property."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        new_result = result.clustering.apply(da_flat)
        np.testing.assert_array_equal(
            result.cluster_assignments.values,
            new_result.cluster_assignments.values,
        )

    def test_apply_with_multi_dim_cluster(self, tmp_path):
        """Apply works with multi-dim cluster_dim (MultiIndex)."""
        da = _make_da()
        result = tsam_xarray.aggregate(
            da,
            time_dim="time",
            cluster_dim=["variable", "region"],
            n_clusters=4,
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        new_result = clustering.apply(da)
        assert set(new_result.typical_periods.dims) == {
            "cluster",
            "timestep",
            "variable",
            "region",
        }

    def test_apply_override_cluster_dim(self):
        """apply() accepts different cluster_dim for new data."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        # Rename dim and apply with override
        da_renamed = da_flat.rename({"variable": "source"})
        new_result = result.clustering.apply(da_renamed, cluster_dim="source")
        assert "source" in new_result.typical_periods.dims

    def test_apply_rejects_missing_time_dim(self):
        """apply() raises if time_dim not in new data."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        da_bad = da_flat.rename({"time": "t"})
        with pytest.raises(ValueError, match="time_dim"):
            result.clustering.apply(da_bad)

    def test_apply_rejects_missing_cluster_dim(self):
        """apply() raises if cluster_dim not in new data."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        da_bad = da_flat.rename({"variable": "other"})
        with pytest.raises(ValueError, match="cluster_dim"):
            result.clustering.apply(da_bad)

    def test_apply_rejects_mismatched_slice_keys(self):
        """apply() raises if slice coords don't match stored."""
        da = _make_da(scenarios=["low", "high"])
        result = tsam_xarray.aggregate(
            da,
            time_dim="time",
            cluster_dim=["variable", "region"],
            n_clusters=4,
        )
        # New data with different scenarios
        da_bad = _make_da(scenarios=["a", "b"])
        with pytest.raises(ValueError, match="No stored clustering"):
            result.clustering.apply(da_bad)

    def test_save_load_nondefault_time_dim(self, tmp_path):
        """Round-trip with non-default time_dim."""
        time = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
        rng = np.random.default_rng(42)
        da = xr.DataArray(
            rng.random((len(time), 2)),
            dims=["t", "variable"],
            coords={"t": time, "variable": ["solar", "wind"]},
        )
        result = tsam_xarray.aggregate(
            da, time_dim="t", cluster_dim="variable", n_clusters=4
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        assert clustering.time_dim == "t"
        new_result = clustering.apply(da)
        assert new_result.n_clusters == result.n_clusters
        assert new_result.typical_periods.dims == result.typical_periods.dims

    def test_json_file_structure(self, tmp_path):
        """JSON has expected top-level keys and array-based clustering keys."""
        import json

        da = _make_da(scenarios=["low", "high"])
        result = tsam_xarray.aggregate(
            da,
            time_dim="time",
            cluster_dim=["variable", "region"],
            n_clusters=4,
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        with open(path) as f:
            data = json.load(f)
        assert "time_dim" in data
        assert "cluster_dim" in data
        assert "slice_dims" in data
        assert "clusterings" in data
        assert isinstance(data["clusterings"], list)
        assert isinstance(data["clusterings"][0]["key"], list)

    def test_save_load_with_segmentation(self, tmp_path):
        """Segmentation info survives JSON round-trip."""
        from tsam import SegmentConfig

        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
            segments=SegmentConfig(n_segments=6),
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        new_result = clustering.apply(da_flat)
        assert new_result.segment_durations is not None
        assert new_result.segment_durations.sizes["timestep"] == 6

    def test_apply_different_variables(self, tmp_path):
        """Cluster on subset, apply to full dataset."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        # Cluster on solar only
        da_solar = da_flat.sel(variable=["solar"])
        result = tsam_xarray.aggregate(
            da_solar,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        # Apply to both solar and wind
        new_result = clustering.apply(da_flat)
        assert set(new_result.typical_periods.coords["variable"].values) == {
            "solar",
            "wind",
        }

    def test_apply_sliced_multi_dim_cluster(self, tmp_path):
        """Save/load/apply with slicing AND multi-dim cluster_dim."""
        da = _make_da(scenarios=["low", "high"])
        result = tsam_xarray.aggregate(
            da,
            time_dim="time",
            cluster_dim=["variable", "region"],
            n_clusters=4,
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        new_result = clustering.apply(da)
        np.testing.assert_allclose(
            result.typical_periods.values,
            new_result.typical_periods.values,
        )

    def test_slice_dims_preserved_in_json(self, tmp_path):
        """slice_dims stored and loaded correctly."""
        da = _make_da(scenarios=["low", "high"])
        result = tsam_xarray.aggregate(
            da,
            time_dim="time",
            cluster_dim=["variable", "region"],
            n_clusters=4,
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        assert clustering.slice_dims == ["scenario"]

    def test_disaggregate_after_apply(self, tmp_path):
        """disaggregate works on apply() results."""
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        result = tsam_xarray.aggregate(
            da_flat,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
        )
        path = tmp_path / "clustering.json"
        result.clustering.to_json(str(path))
        clustering = tsam_xarray.load_clustering(str(path))
        new_result = clustering.apply(da_flat)
        dis = new_result.disaggregate(new_result.typical_periods)
        xr.testing.assert_allclose(dis, new_result.reconstructed)


class TestSliceEdgeCases:
    def test_cluster_count_mismatch_raises(self):
        """Mismatched cluster counts across slices raise ValueError."""
        from tsam_xarray._core import _validate_consistent_cluster_counts

        # Create two real results with different n_clusters
        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        r1 = tsam_xarray.aggregate(
            da_flat, time_dim="time", cluster_dim="variable", n_clusters=3
        )
        r2 = tsam_xarray.aggregate(
            da_flat, time_dim="time", cluster_dim="variable", n_clusters=4
        )
        with pytest.raises(ValueError, match="different cluster counts"):
            _validate_consistent_cluster_counts([r1, r2], [("low",), ("high",)])

    def test_cluster_count_consistent_passes(self):
        """Same cluster counts across slices passes validation."""
        from tsam_xarray._core import _validate_consistent_cluster_counts

        da = _make_da()
        da_flat = da.isel(region=0).drop_vars("region")
        r1 = tsam_xarray.aggregate(
            da_flat, time_dim="time", cluster_dim="variable", n_clusters=4
        )
        r2 = tsam_xarray.aggregate(
            da_flat, time_dim="time", cluster_dim="variable", n_clusters=4
        )
        _validate_consistent_cluster_counts(
            [r1, r2], [("low",), ("high",)]
        )  # should not raise
