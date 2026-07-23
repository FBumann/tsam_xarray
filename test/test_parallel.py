"""Parallel slice aggregation via n_jobs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsam_xarray import aggregate
from tsam_xarray._core import _resolve_n_workers


def _data(n_slices: int = 4) -> xr.DataArray:
    rng = np.random.default_rng(7)
    n_t = 21 * 24
    return xr.DataArray(
        rng.random((n_slices, 3, n_t)),
        dims=["scenario", "variable", "time"],
        coords={
            "scenario": [f"s{i}" for i in range(n_slices)],
            "variable": ["a", "b", "c"],
            "time": pd.date_range("2023-01-01", periods=n_t, freq="h"),
        },
        name="load",
    )


@pytest.mark.parametrize("n_jobs", [2, -1])
def test_parallel_matches_sequential(n_jobs):
    da = _data()
    sequential = aggregate(da, time_dim="time", cluster_dim="variable", n_clusters=4)
    parallel = aggregate(
        da, time_dim="time", cluster_dim="variable", n_clusters=4, n_jobs=n_jobs
    )

    xr.testing.assert_identical(
        sequential.cluster_representatives, parallel.cluster_representatives
    )
    xr.testing.assert_identical(
        sequential.cluster_assignments, parallel.cluster_assignments
    )
    xr.testing.assert_identical(sequential.reconstructed, parallel.reconstructed)
    xr.testing.assert_identical(sequential.accuracy.rmse, parallel.accuracy.rmse)


def test_n_jobs_without_slice_dims_is_ignored():
    da = _data(1).isel(scenario=0, drop=True)
    result = aggregate(
        da, time_dim="time", cluster_dim="variable", n_clusters=4, n_jobs=-1
    )
    assert result.n_clusters == 4


def test_resolve_n_workers():
    assert _resolve_n_workers(None, 8) == 1
    assert _resolve_n_workers(1, 8) == 1
    assert _resolve_n_workers(3, 8) == 3
    assert _resolve_n_workers(16, 8) == 8
    assert _resolve_n_workers(-1, 100) >= 1
    assert _resolve_n_workers(-1, 100) == _resolve_n_workers(-2, 100) + 1
