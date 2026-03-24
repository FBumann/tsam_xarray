"""Core aggregation logic for tsam_xarray."""

from __future__ import annotations

import itertools
from collections.abc import Hashable
from typing import Any

import numpy as np
import pandas as pd
import tsam
import xarray as xr

from tsam_xarray._result import AccuracyMetrics, AggregationResult


def aggregate(
    da: xr.DataArray,
    n_clusters: int,
    *,
    time_dim: str = "time",
    column_dim: str | None = None,
    weights: dict[str, float] | None = None,
    **tsam_kwargs: Any,
) -> AggregationResult:
    """Aggregate an xarray DataArray using tsam.

    Parameters
    ----------
    da : xr.DataArray
        Input data with a time dimension and optional extra dimensions.
        Use ``da.stack(column=["var1", "var2"])`` to combine multiple
        dimensions into a single column dimension before calling.
    n_clusters : int
        Number of typical periods.
    time_dim : str
        Name of the time dimension (default: ``"time"``).
    column_dim : str | None
        Dimension that becomes DataFrame columns. If ``None`` and the
        DataArray has exactly two dims, the non-time dim is used. For
        1D data (time only), pass ``None``.
    weights : dict[str, float] | None
        Per-column weights passed to ``tsam.aggregate()``.
    **tsam_kwargs
        Additional keyword arguments passed to ``tsam.aggregate()``.
    """
    _validate_time_dim(da, time_dim)
    column_dim = _resolve_column_dim(da, time_dim, column_dim)
    slice_dims = _infer_slice_dims(da, time_dim, column_dim)
    _validate(da, time_dim, column_dim, slice_dims)

    if not slice_dims:
        return _aggregate_single(
            da, n_clusters, time_dim, column_dim, weights, tsam_kwargs
        )

    slice_coords = {d: da.coords[d].values for d in slice_dims}
    slice_keys = list(itertools.product(*(slice_coords[d] for d in slice_dims)))

    results: list[AggregationResult] = []
    raw_map: dict[tuple[Hashable, ...], Any] = {}

    for key in slice_keys:
        sel = dict(zip(slice_dims, key, strict=True))
        da_slice = da.sel(sel)
        r = _aggregate_single(
            da_slice, n_clusters, time_dim, column_dim, weights, tsam_kwargs
        )
        results.append(r)
        raw_map[key] = r.raw

    return _concat_results(results, slice_dims, slice_coords, raw_map)


def _resolve_column_dim(
    da: xr.DataArray,
    time_dim: str,
    column_dim: str | None,
) -> str | None:
    """Resolve column_dim, auto-detecting if not specified."""
    if column_dim is not None:
        return column_dim
    non_time = [d for d in da.dims if d != time_dim]
    if len(non_time) == 0:
        return None
    if len(non_time) == 1:
        return str(non_time[0])
    # Multiple non-time dims — can't auto-detect
    msg = (
        f"DataArray has multiple non-time dims {non_time}. "
        "Specify column_dim explicitly, or use da.stack() to "
        "combine dims first."
    )
    raise ValueError(msg)


def _infer_slice_dims(
    da: xr.DataArray,
    time_dim: str,
    column_dim: str | None,
) -> list[str]:
    """Infer slice dims: everything not time_dim or column_dim."""
    exclude = {time_dim}
    if column_dim is not None:
        exclude.add(column_dim)
    return [str(d) for d in da.dims if d not in exclude]


def _validate_time_dim(da: xr.DataArray, time_dim: str) -> None:
    if time_dim not in da.dims:
        msg = f"time_dim {time_dim!r} not in DataArray dims {set(da.dims)}"
        raise ValueError(msg)


def _validate(
    da: xr.DataArray,
    time_dim: str,
    column_dim: str | None,
    slice_dims: list[str],
) -> None:
    dims = set(da.dims)
    if column_dim is not None and column_dim not in dims:
        msg = f"column_dim {column_dim!r} not in DataArray dims {dims}"
        raise ValueError(msg)
    if column_dim == time_dim:
        msg = "column_dim and time_dim must be different"
        raise ValueError(msg)


def _to_dataframe(
    da: xr.DataArray,
    time_dim: str,
    column_dim: str | None,
) -> pd.DataFrame:
    """Convert DataArray to DataFrame for tsam."""
    if column_dim is None:
        # 1D time series
        s = da.to_pandas()
        if isinstance(s, pd.Series):
            name = da.name or "value"
            return s.to_frame(name=str(name))
        return pd.DataFrame(s)

    da_t = da.transpose(time_dim, column_dim)
    return pd.DataFrame(da_t.to_pandas())


def _representatives_to_da(
    df: pd.DataFrame,
    column_dim: str | None,
) -> xr.DataArray:
    """Convert cluster_representatives DataFrame to DataArray."""
    df = df.copy()
    df.index.names = ["cluster", "timestep"]

    if column_dim is None:
        clusters = df.index.get_level_values(0).unique()
        timesteps = df.index.get_level_values(1).unique()
        values = df.values.squeeze(axis=1).reshape(len(clusters), len(timesteps))
        return xr.DataArray(
            values,
            dims=["cluster", "timestep"],
            coords={"cluster": clusters, "timestep": timesteps},
        )

    stacked = df.stack(df.columns.names, future_stack=True)
    da: xr.DataArray = stacked.to_xarray()  # type: ignore[assignment]
    return da


def _reconstructed_to_da(
    df: pd.DataFrame,
    time_dim: str,
    column_dim: str | None,
) -> xr.DataArray:
    """Convert reconstructed DataFrame to DataArray."""
    df = df.copy()
    df.index.name = time_dim

    if column_dim is None:
        return xr.DataArray(
            df.values.squeeze(axis=1),
            dims=[time_dim],
            coords={time_dim: df.index},
        )

    stacked = df.stack(df.columns.names, future_stack=True)
    da: xr.DataArray = stacked.to_xarray()  # type: ignore[assignment]
    return da


def _metric_to_da(
    series: pd.Series[float],
    column_dim: str | None,
    column_names: list[str] | None = None,
) -> xr.DataArray:
    """Convert an accuracy metric Series to DataArray."""
    if column_dim is None:
        return xr.DataArray(float(series.iloc[0]))
    series = series.copy()
    if isinstance(series.index, pd.MultiIndex):
        # Restore MultiIndex level names (tsam drops them)
        if column_names is not None:
            series.index = series.index.set_names(column_names)
    elif series.index.name is None:
        series.index.name = column_dim
    return xr.DataArray(series.to_xarray())


def _aggregate_single(
    da: xr.DataArray,
    n_clusters: int,
    time_dim: str,
    column_dim: str | None,
    weights: dict[str, float] | None,
    tsam_kwargs: dict[str, Any],
) -> AggregationResult:
    """Run a single tsam aggregation on a DataArray."""
    df = _to_dataframe(da, time_dim, column_dim)

    tsam_result = tsam.aggregate(df, n_clusters, weights=weights, **tsam_kwargs)

    typical = _representatives_to_da(tsam_result.cluster_representatives, column_dim)
    reconstructed = _reconstructed_to_da(
        tsam_result.reconstructed, time_dim, column_dim
    )

    cw = tsam_result.cluster_weights
    cluster_ids = np.array(sorted(cw.keys()))
    cluster_weights_da = xr.DataArray(
        np.array([cw[k] for k in cluster_ids]),
        dims=["cluster"],
        coords={"cluster": cluster_ids},
    )

    assignments_da = xr.DataArray(tsam_result.cluster_assignments, dims=["period"])

    # Extract column names for restoring MultiIndex level names
    col_names: list[str] | None = None
    if isinstance(df.columns, pd.MultiIndex):
        col_names = [str(n) for n in df.columns.names]

    accuracy = AccuracyMetrics(
        rmse=_metric_to_da(tsam_result.accuracy.rmse, column_dim, col_names),
        mae=_metric_to_da(tsam_result.accuracy.mae, column_dim, col_names),
        rmse_duration=_metric_to_da(
            tsam_result.accuracy.rmse_duration, column_dim, col_names
        ),
    )

    return AggregationResult(
        typical_periods=typical,
        cluster_assignments=assignments_da,
        cluster_weights=cluster_weights_da,
        accuracy=accuracy,
        reconstructed=reconstructed,
        original=da,
        raw=tsam_result,
    )


def _make_dim_index(
    slice_coords: dict[str, Any],
    dim: str,
) -> pd.Index:
    """Create a pd.Index for a slice dimension."""
    return pd.Index(slice_coords[dim], name=dim)  # type: ignore[no-any-return]


def _concat_along_dims(
    arrays: list[xr.DataArray],
    slice_dims: list[str],
    slice_coords: dict[str, Any],
) -> xr.DataArray:
    """Concat arrays along one or more slice dims."""
    if len(slice_dims) == 1:
        return xr.concat(arrays, dim=_make_dim_index(slice_coords, slice_dims[0]))
    it = iter(arrays)

    def _nest(dims: list[str]) -> list[Any]:
        if len(dims) == 1:
            return [next(it) for _ in slice_coords[dims[0]]]
        return [_nest(dims[1:]) for _ in slice_coords[dims[0]]]

    nested: Any = _nest(slice_dims)

    def _recursive_concat(node: Any, dims: list[str]) -> xr.DataArray:
        dim = dims[0]
        idx = _make_dim_index(slice_coords, dim)
        if len(dims) == 1:
            return xr.concat(node, dim=idx)  # type: ignore[no-any-return]
        children = [_recursive_concat(child, dims[1:]) for child in node]
        return xr.concat(children, dim=idx)

    return _recursive_concat(nested, slice_dims)


def _concat_results(
    results: list[AggregationResult],
    slice_dims: list[str],
    slice_coords: dict[str, Any],
    raw_map: dict[tuple[Hashable, ...], Any],
) -> AggregationResult:
    """Concatenate per-slice results along slice dims."""

    def _field(field_name: str) -> xr.DataArray:
        arrays = [getattr(r, field_name) for r in results]
        return _concat_along_dims(arrays, slice_dims, slice_coords)

    def _acc_field(field_name: str) -> xr.DataArray:
        arrays = [getattr(r.accuracy, field_name) for r in results]
        return _concat_along_dims(arrays, slice_dims, slice_coords)

    return AggregationResult(
        typical_periods=_field("typical_periods"),
        cluster_assignments=_field("cluster_assignments"),
        cluster_weights=_field("cluster_weights"),
        accuracy=AccuracyMetrics(
            rmse=_acc_field("rmse"),
            mae=_acc_field("mae"),
            rmse_duration=_acc_field("rmse_duration"),
        ),
        reconstructed=_field("reconstructed"),
        original=_field("original"),
        raw=raw_map,
    )
