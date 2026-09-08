"""AggregationResult.concurrency, tsam's cross-column concurrency metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tsam
import xarray as xr

import tsam_xarray
from tsam_xarray import ConcurrencyMetrics, aggregate

HAS_CONCURRENCY = hasattr(tsam.AggregationResult, "concurrency")

requires_concurrency = pytest.mark.skipif(
    not HAS_CONCURRENCY,
    reason="tsam < 4 does not compute concurrency metrics",
)


def _data(n_slices: int = 1, variables: list[str] | None = None) -> xr.DataArray:
    if variables is None:
        variables = ["a", "b", "c"]
    rng = np.random.default_rng(0)
    n_t = 14 * 24
    dims = ["variable", "time"]
    shape: tuple[int, ...] = (len(variables), n_t)
    coords: dict[str, object] = {
        "variable": variables,
        "time": pd.date_range("2023-01-01", periods=n_t, freq="h"),
    }
    if n_slices > 1:
        dims = ["scenario", *dims]
        shape = (n_slices, *shape)
        coords["scenario"] = [f"s{i}" for i in range(n_slices)]
    return xr.DataArray(rng.random(shape), dims=dims, coords=coords, name="load")


def _aggregate(da: xr.DataArray, n_clusters: int = 4):
    return aggregate(da, time_dim="time", cluster_dim="variable", n_clusters=n_clusters)


def test_concurrency_metrics_is_exported():
    assert tsam_xarray.ConcurrencyMetrics is ConcurrencyMetrics


@pytest.mark.skipif(HAS_CONCURRENCY, reason="tsam >= 4 computes concurrency metrics")
def test_none_without_tsam_support():
    assert _aggregate(_data()).concurrency is None


@requires_concurrency
def test_scalar_without_slice_dims():
    concurrency = _aggregate(_data()).concurrency

    assert isinstance(concurrency, ConcurrencyMetrics)
    for metric in (concurrency.correlation_error, concurrency.rank_correlation_error):
        assert metric.dims == ()
        assert float(metric) >= 0


@requires_concurrency
def test_dims_follow_slice_dims():
    da = _data(n_slices=3)
    concurrency = _aggregate(da).concurrency

    for metric in (concurrency.correlation_error, concurrency.rank_correlation_error):
        assert metric.dims == ("scenario",)
        xr.testing.assert_identical(metric.coords["scenario"], da.coords["scenario"])


@requires_concurrency
def test_exact_reconstruction_has_no_concurrency_error():
    da = _data()
    n_periods = da.sizes["time"] // 24
    concurrency = _aggregate(da, n_clusters=n_periods).concurrency

    assert float(concurrency.correlation_error) == pytest.approx(0, abs=1e-9)
    assert float(concurrency.rank_correlation_error) == pytest.approx(0, abs=1e-9)


@requires_concurrency
def test_nan_for_a_single_clustered_column():
    concurrency = _aggregate(_data(variables=["a"])).concurrency

    assert np.isnan(float(concurrency.correlation_error))
    assert np.isnan(float(concurrency.rank_correlation_error))


@requires_concurrency
@pytest.mark.parametrize("n_slices", [1, 3])
def test_deferred_until_accessed(n_slices):
    result = _aggregate(_data(n_slices))

    assert "concurrency" not in result.__dict__

    concurrency = result.concurrency
    assert "concurrency" in result.__dict__
    assert result.concurrency is concurrency


@requires_concurrency
@pytest.mark.parametrize("n_slices", [1, 3])
def test_apply_reports_concurrency(n_slices):
    da = _data(n_slices)
    result = _aggregate(da)
    transferred = result.clustering.apply(da)

    assert transferred.is_transferred
    xr.testing.assert_allclose(
        transferred.concurrency.correlation_error,
        result.concurrency.correlation_error,
    )


@requires_concurrency
def test_repr_reports_both_metrics():
    text = repr(_aggregate(_data()).concurrency)

    assert text.startswith("ConcurrencyMetrics(")
    assert "correlation_error=" in text
    assert "rank_correlation_error=" in text
