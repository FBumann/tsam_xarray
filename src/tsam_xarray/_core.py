"""Core aggregation logic for tsam_xarray."""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import tsam
import xarray as xr

from tsam_xarray._result import AccuracyMetrics, AggregationResult


def aggregate(
    da: xr.DataArray,
    *,
    time_dim: str,
    cluster_dim: Sequence[str] | str,
    n_clusters: int,
    weights: dict[str, dict[str, float]] | dict[str, float] | None = None,
    **tsam_kwargs: Any,
) -> AggregationResult:
    """Aggregate an xarray DataArray using tsam.

    Parameters
    ----------
    da : xr.DataArray
        Input data with a time dimension and optional extra dimensions.
    time_dim : str
        Name of the time dimension.
    cluster_dim : Sequence[str] | str
        Dimension(s) to cluster together. Multiple dims are stacked
        internally into a MultiIndex and unstacked in results.
        All remaining dims are sliced independently.
    n_clusters : int
        Number of typical periods.
    weights : dict[str, dict[str, float]] | dict[str, float] | None
        Per-dimension weights. For a single ``cluster_dim``, pass a
        simple dict keyed by coordinate values, e.g.
        ``{"solar": 2.0, "wind": 1.0}``. For multiple ``cluster_dim``,
        pass a dict-of-dicts keyed by dimension name, e.g.
        ``{"variable": {"solar": 2.0}, "region": {"north": 1.5}}``.
        Weights are multiplied across dimensions. Missing entries
        default to 1.0.
    **tsam_kwargs
        Additional keyword arguments passed to ``tsam.aggregate()``.
    """
    _validate_time_dim(da, time_dim)
    col_dims = _resolve_cluster_dim(cluster_dim)
    slice_dims = _infer_slice_dims(da, time_dim, col_dims)
    _validate(da, time_dim, col_dims, slice_dims)
    _validate_no_cluster_config_weights(tsam_kwargs)
    norm_weights = _normalize_weights(weights, col_dims)

    if not slice_dims:
        return _aggregate_single(
            da, n_clusters, time_dim, col_dims, norm_weights, tsam_kwargs
        )

    slice_coords = {d: da.coords[d].values for d in slice_dims}
    slice_keys = list(itertools.product(*(slice_coords[d] for d in slice_dims)))

    results: list[AggregationResult] = []
    raw_map: dict[tuple[Hashable, ...], Any] = {}

    for key in slice_keys:
        sel = dict(zip(slice_dims, key, strict=True))
        da_slice = da.sel(sel)
        r = _aggregate_single(
            da_slice, n_clusters, time_dim, col_dims, norm_weights, tsam_kwargs
        )
        results.append(r)
        raw_map[key] = r.raw

    return _concat_results(results, slice_dims, slice_coords, raw_map)


def _resolve_cluster_dim(
    cluster_dim: Sequence[str] | str,
) -> list[str]:
    """Resolve cluster_dim to a list of dimension names."""
    if isinstance(cluster_dim, str):
        return [cluster_dim]
    return list(cluster_dim)


def _infer_slice_dims(
    da: xr.DataArray,
    time_dim: str,
    col_dims: list[str],
) -> list[str]:
    """Infer slice dims: everything not time_dim or column dims."""
    exclude = {time_dim, *col_dims}
    return [str(d) for d in da.dims if d not in exclude]


def _validate_time_dim(da: xr.DataArray, time_dim: str) -> None:
    if time_dim not in da.dims:
        msg = f"time_dim {time_dim!r} not in DataArray dims {set(da.dims)}"
        raise ValueError(msg)


def _validate_no_cluster_config_weights(
    tsam_kwargs: dict[str, Any],
) -> None:
    """Reject deprecated weights in ClusterConfig."""
    cluster_config = tsam_kwargs.get("cluster")
    if cluster_config is not None and cluster_config.weights is not None:
        msg = (
            "ClusterConfig.weights is deprecated in tsam and not "
            "supported by tsam_xarray. Use the top-level 'weights' "
            "parameter of aggregate() instead."
        )
        raise ValueError(msg)


def _validate(
    da: xr.DataArray,
    time_dim: str,
    col_dims: list[str],
    slice_dims: list[str],
) -> None:
    dims = set(da.dims)
    for d in col_dims:
        if d not in dims:
            msg = f"cluster_dim entry {d!r} not in DataArray dims {dims}"
            raise ValueError(msg)
        if d == time_dim:
            msg = "cluster_dim and time_dim must not overlap"
            raise ValueError(msg)


def _to_dataframe(
    da: xr.DataArray,
    time_dim: str,
    col_dims: list[str],
) -> pd.DataFrame:
    """Convert DataArray to DataFrame for tsam."""
    if not col_dims:
        s = da.to_pandas()
        if isinstance(s, pd.Series):
            name = da.name or "value"
            return s.to_frame(name=str(name))
        return pd.DataFrame(s)

    if len(col_dims) > 1:
        da = da.stack(_column=col_dims)
        col_dim = "_column"
    else:
        col_dim = col_dims[0]

    da_t = da.transpose(time_dim, col_dim)
    return pd.DataFrame(da_t.to_pandas())


def _representatives_to_da(
    df: pd.DataFrame,
    col_dims: list[str],
) -> xr.DataArray:
    """Convert cluster_representatives DataFrame to DataArray."""
    df = df.copy()
    df.index.names = ["cluster", "timestep"]

    if not col_dims:
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
    col_dims: list[str],
) -> xr.DataArray:
    """Convert reconstructed DataFrame to DataArray."""
    df = df.copy()
    df.index.name = time_dim

    if not col_dims:
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
    col_dims: list[str],
    column_names: list[str] | None = None,
) -> xr.DataArray:
    """Convert an accuracy metric Series to DataArray."""
    if not col_dims:
        return xr.DataArray(float(series.iloc[0]))
    series = series.copy()
    if isinstance(series.index, pd.MultiIndex):
        if column_names is not None:
            series.index = series.index.set_names(column_names)
    elif series.index.name is None:
        series.index.name = col_dims[0]
    return xr.DataArray(series.to_xarray())


def _normalize_weights(
    weights: dict[str, dict[str, float]] | dict[str, float] | None,
    col_dims: list[str],
) -> dict[str, dict[str, float]] | None:
    """Normalize weights to dict-of-dicts form."""
    if weights is None:
        return None
    if not weights:
        return None
    # Check if it's already dict-of-dicts (values are dicts)
    first_val = next(iter(weights.values()))
    if isinstance(first_val, dict):
        return weights  # type: ignore[return-value]
    # Simple dict — wrap with the single cluster dim name
    if len(col_dims) != 1:
        msg = (
            "Simple dict weights require a single cluster_dim. "
            "For multiple cluster_dim, use dict-of-dicts: "
            '{"dim_name": {"coord": weight}}.'
        )
        raise ValueError(msg)
    return {col_dims[0]: weights}  # type: ignore[dict-item]


def _translate_weights(
    weights: dict[str, dict[str, float]],
    df: pd.DataFrame,
    col_dims: list[str],
) -> dict[Hashable, float]:
    """Translate per-dim weights to flat column weights for tsam."""
    flat: dict[Hashable, float] = {}
    for col in df.columns:
        w = 1.0
        if isinstance(col, tuple):
            for dim_name, coord_val in zip(col_dims, col, strict=True):
                if dim_name in weights:
                    w *= weights[dim_name].get(str(coord_val), 1.0)
        else:
            # Single dim
            dim_name = col_dims[0]
            if dim_name in weights:
                w *= weights[dim_name].get(str(col), 1.0)
        flat[col] = w
    return flat


def _aggregate_single(
    da: xr.DataArray,
    n_clusters: int,
    time_dim: str,
    col_dims: list[str],
    weights: dict[str, dict[str, float]] | None,
    tsam_kwargs: dict[str, Any],
) -> AggregationResult:
    """Run a single tsam aggregation on a DataArray."""
    df = _to_dataframe(da, time_dim, col_dims)

    flat_weights: dict[Hashable, float] | None = None
    if weights is not None:
        flat_weights = _translate_weights(weights, df, col_dims)

    tsam_result = tsam.aggregate(
        df,
        n_clusters,
        weights=flat_weights,
        **tsam_kwargs,  # type: ignore[arg-type]
    )

    typical = _representatives_to_da(tsam_result.cluster_representatives, col_dims)
    reconstructed = _reconstructed_to_da(tsam_result.reconstructed, time_dim, col_dims)

    cw = tsam_result.cluster_weights
    cluster_ids = np.array(sorted(cw.keys()))
    cluster_weights_da = xr.DataArray(
        np.array([cw[k] for k in cluster_ids]),
        dims=["cluster"],
        coords={"cluster": cluster_ids},
    )

    assignments_da = xr.DataArray(tsam_result.cluster_assignments, dims=["period"])

    col_names: list[str] | None = None
    if isinstance(df.columns, pd.MultiIndex):
        col_names = [str(n) for n in df.columns.names]

    accuracy = AccuracyMetrics(
        rmse=_metric_to_da(tsam_result.accuracy.rmse, col_dims, col_names),
        mae=_metric_to_da(tsam_result.accuracy.mae, col_dims, col_names),
        rmse_duration=_metric_to_da(
            tsam_result.accuracy.rmse_duration, col_dims, col_names
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
