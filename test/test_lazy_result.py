"""Laziness of AggregationResult.accuracy and .reconstructed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsam_xarray import aggregate


def _data(n_slices: int = 1) -> xr.DataArray:
    rng = np.random.default_rng(0)
    n_t = 14 * 24
    dims = ["variable", "time"]
    shape: tuple[int, ...] = (3, n_t)
    coords: dict[str, object] = {
        "variable": ["a", "b", "c"],
        "time": pd.date_range("2023-01-01", periods=n_t, freq="h"),
    }
    if n_slices > 1:
        dims = ["scenario", *dims]
        shape = (n_slices, *shape)
        coords["scenario"] = [f"s{i}" for i in range(n_slices)]
    return xr.DataArray(rng.random(shape), dims=dims, coords=coords, name="load")


@pytest.mark.parametrize("n_slices", [1, 3])
def test_accuracy_and_reconstructed_are_deferred(n_slices):
    result = aggregate(
        _data(n_slices), time_dim="time", cluster_dim="variable", n_clusters=4
    )

    assert "accuracy" not in result.__dict__
    assert "reconstructed" not in result.__dict__

    repr(result)
    assert "accuracy" not in result.__dict__

    accuracy = result.accuracy
    reconstructed = result.reconstructed
    assert "accuracy" in result.__dict__
    assert "reconstructed" in result.__dict__

    assert result.accuracy is accuracy
    assert result.reconstructed is reconstructed


def test_deferred_values_match_shapes():
    da = _data(3)
    result = aggregate(da, time_dim="time", cluster_dim="variable", n_clusters=4)

    assert result.reconstructed.dims == da.dims
    assert result.reconstructed.shape == da.shape
    assert set(result.accuracy.rmse.dims) == {"variable", "scenario"}
    assert not result.residuals.isnull().all()


def test_repr_shows_metrics_once_computed():
    result = aggregate(_data(), time_dim="time", cluster_dim="variable", n_clusters=4)
    assert "not computed" in repr(result)
    _ = result.accuracy
    assert "weighted_rmse=" in repr(result)
