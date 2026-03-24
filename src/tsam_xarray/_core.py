"""Core aggregation logic for tsam_xarray."""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import tsam
import xarray as xr

from tsam_xarray._result import AccuracyResult, AggregationResult

_SEP = "__"


def aggregate(
    da: xr.DataArray,
    n_clusters: int,
    *,
    time_dim: str,
    stack_dims: Sequence[str] = (),
    slice_dims: Sequence[str] = (),
    weights: dict[str, dict[str, float]] | None = None,
    **tsam_kwargs: Any,
) -> AggregationResult:
    """Aggregate an xarray DataArray using tsam.

    Parameters
    ----------
    da : xr.DataArray
        Input data with a time dimension and optional extra dimensions.
    n_clusters : int
        Number of typical periods.
    time_dim : str
        Name of the time dimension.
    stack_dims : Sequence[str]
        Dimensions to flatten into DataFrame columns (shared clustering).
    slice_dims : Sequence[str]
        Dimensions to loop over independently (one aggregation each).
    weights : dict[str, dict[str, float]] | None
        Per-dimension weights, e.g. ``{"variable": {"solar": 2.0}}``.
    **tsam_kwargs
        Additional keyword arguments passed to ``tsam.aggregate()``.
    """
    stack_dims_l = list(stack_dims)
    slice_dims_l = list(slice_dims)
    _validate(da, time_dim, stack_dims_l, slice_dims_l, weights)

    if not slice_dims_l:
        return _aggregate_single(
            da, n_clusters, time_dim, stack_dims_l, weights, tsam_kwargs
        )

    slice_coords = {d: da.coords[d].values for d in slice_dims_l}
    slice_keys = list(
        itertools.product(*(slice_coords[d] for d in slice_dims_l))
    )

    results: list[AggregationResult] = []
    raw_map: dict[tuple[Hashable, ...], Any] = {}

    for key in slice_keys:
        sel = dict(zip(slice_dims_l, key, strict=True))
        da_slice = da.sel(sel)
        r = _aggregate_single(
            da_slice, n_clusters, time_dim, stack_dims_l, weights, tsam_kwargs
        )
        results.append(r)
        raw_map[key] = r.raw

    return _concat_results(results, slice_dims_l, slice_coords, raw_map)


def _validate(
    da: xr.DataArray,
    time_dim: str,
    stack_dims: list[str],
    slice_dims: list[str],
    weights: dict[str, dict[str, float]] | None,
) -> None:
    dims = set(da.dims)
    if time_dim not in dims:
        msg = f"time_dim {time_dim!r} not in DataArray dims {dims}"
        raise ValueError(msg)
    for d in stack_dims:
        if d not in dims:
            msg = f"stack_dims entry {d!r} not in DataArray dims {dims}"
            raise ValueError(msg)
    for d in slice_dims:
        if d not in dims:
            msg = f"slice_dims entry {d!r} not in DataArray dims {dims}"
            raise ValueError(msg)

    all_specified = {time_dim} | set(stack_dims) | set(slice_dims)
    if len(all_specified) != 1 + len(stack_dims) + len(slice_dims):
        msg = "time_dim, stack_dims, and slice_dims must not overlap"
        raise ValueError(msg)
    unaccounted = dims - all_specified
    if unaccounted:
        msg = (
            f"Dimensions {unaccounted} not in time_dim, "
            "stack_dims, or slice_dims"
        )
        raise ValueError(msg)

    if weights is not None:
        for dim_name, dim_weights in weights.items():
            if dim_name not in stack_dims:
                msg = (
                    f"Weight key {dim_name!r} is not in "
                    f"stack_dims {stack_dims}"
                )
                raise ValueError(msg)
            valid = set(da.coords[dim_name].values.tolist())
            for coord_name in dim_weights:
                if coord_name not in valid:
                    msg = (
                        f"Weight coord {coord_name!r} not in "
                        f"{dim_name!r} coordinates"
                    )
                    raise ValueError(msg)


def _flatten(
    da: xr.DataArray,
    time_dim: str,
    stack_dims: list[str],
) -> tuple[pd.DataFrame, dict[str, tuple[Hashable, ...]]]:
    """Flatten DataArray to DataFrame with flat string column names."""
    if not stack_dims:
        s = da.to_pandas()
        if isinstance(s, pd.Series):
            name = da.name or "value"
            df: pd.DataFrame = s.to_frame(name=str(name))
        else:
            df = pd.DataFrame(s)
        return df, {c: (c,) for c in df.columns}

    ordered = [time_dim, *stack_dims]
    da_t = da.transpose(*ordered)
    da_stacked = da_t.stack(_flat=stack_dims)
    df = pd.DataFrame(da_stacked.to_pandas())

    coord_map: dict[str, tuple[Hashable, ...]] = {}
    for col in df.columns:
        if isinstance(col, tuple):
            flat_name = _SEP.join(str(c) for c in col)
            coord_map[flat_name] = col
        else:
            flat_name = str(col)
            coord_map[flat_name] = (col,)
    df.columns = pd.Index(list(coord_map.keys()))
    return df, coord_map


def _translate_weights(
    weights: dict[str, dict[str, float]],
    coord_map: dict[str, tuple[Hashable, ...]],
    stack_dims: list[str],
) -> dict[str, float]:
    """Translate per-dim weights to flat column weights."""
    flat: dict[str, float] = {}
    for flat_name, coord_tuple in coord_map.items():
        w = 1.0
        for dim_name, coord_val in zip(
            stack_dims, coord_tuple, strict=True
        ):
            if dim_name in weights:
                w *= weights[dim_name].get(str(coord_val), 1.0)
        flat[flat_name] = w
    return flat


def _unflatten_representatives(
    df: pd.DataFrame,
    stack_dims: list[str],
    coord_map: dict[str, tuple[Hashable, ...]],
) -> xr.DataArray:
    """Unflatten cluster_representatives to DataArray."""
    df = df.copy()
    df.index.names = ["cluster", "timestep"]

    if not stack_dims:
        clusters = df.index.get_level_values(0).unique()
        timesteps = df.index.get_level_values(1).unique()
        values = df.values.squeeze(axis=1).reshape(
            len(clusters), len(timesteps)
        )
        return xr.DataArray(
            values,
            dims=["cluster", "timestep"],
            coords={"cluster": clusters, "timestep": timesteps},
        )

    return _df_to_da(
        df, stack_dims, coord_map, index_dims=["cluster", "timestep"]
    )


def _unflatten_reconstructed(
    df: pd.DataFrame,
    time_dim: str,
    stack_dims: list[str],
    coord_map: dict[str, tuple[Hashable, ...]],
) -> xr.DataArray:
    """Unflatten reconstructed DataFrame to DataArray."""
    df = df.copy()
    df.index.name = time_dim

    if not stack_dims:
        return xr.DataArray(
            df.values.squeeze(axis=1),
            dims=[time_dim],
            coords={time_dim: df.index},
        )

    return _df_to_da(df, stack_dims, coord_map, index_dims=[time_dim])


def _unflatten_metric(
    series: pd.Series[float],
    stack_dims: list[str],
    coord_map: dict[str, tuple[Hashable, ...]],
) -> xr.DataArray:
    """Unflatten an accuracy metric Series to DataArray."""
    if not stack_dims:
        return xr.DataArray(float(series.iloc[0]))

    if len(stack_dims) == 1:
        idx = pd.Index(
            [coord_map[n][0] for n in series.index],
            name=stack_dims[0],
        )
    else:
        idx = pd.MultiIndex.from_tuples(
            [coord_map[name] for name in series.index],
            names=stack_dims,
        )
    series = series.copy()
    series.index = idx
    result = xr.DataArray(series.to_xarray())
    return result


def _df_to_da(
    df: pd.DataFrame,
    stack_dims: list[str],
    coord_map: dict[str, tuple[Hashable, ...]],
    index_dims: list[str],
) -> xr.DataArray:
    """Convert DataFrame with flat column names back to DataArray."""
    if len(stack_dims) == 1:
        cols: pd.Index = pd.Index(
            [coord_map[c][0] for c in df.columns],
            name=stack_dims[0],
        )
    else:
        cols = pd.MultiIndex.from_tuples(
            [coord_map[c] for c in df.columns],
            names=stack_dims,
        )
    df = df.copy()
    df.columns = cols
    stacked = df.stack(stack_dims, future_stack=True)
    result: xr.DataArray = stacked.to_xarray()  # type: ignore[assignment]
    return result


def _aggregate_single(
    da: xr.DataArray,
    n_clusters: int,
    time_dim: str,
    stack_dims: list[str],
    weights: dict[str, dict[str, float]] | None,
    tsam_kwargs: dict[str, Any],
) -> AggregationResult:
    """Run a single tsam aggregation on a DataArray."""
    df, coord_map = _flatten(da, time_dim, stack_dims)

    flat_weights: dict[str, float] | None = None
    if weights is not None:
        flat_weights = _translate_weights(weights, coord_map, stack_dims)

    tsam_result = tsam.aggregate(
        df, n_clusters, weights=flat_weights, **tsam_kwargs
    )

    typical = _unflatten_representatives(
        tsam_result.cluster_representatives, stack_dims, coord_map
    )
    reconstructed = _unflatten_reconstructed(
        tsam_result.reconstructed, time_dim, stack_dims, coord_map
    )

    cw = tsam_result.cluster_weights
    cluster_ids = np.array(sorted(cw.keys()))
    cluster_weights_da = xr.DataArray(
        np.array([cw[k] for k in cluster_ids]),
        dims=["cluster"],
        coords={"cluster": cluster_ids},
    )

    assignments_da = xr.DataArray(
        tsam_result.cluster_assignments, dims=["period"]
    )

    accuracy = AccuracyResult(
        rmse=_unflatten_metric(
            tsam_result.accuracy.rmse, stack_dims, coord_map
        ),
        mae=_unflatten_metric(
            tsam_result.accuracy.mae, stack_dims, coord_map
        ),
        rmse_duration=_unflatten_metric(
            tsam_result.accuracy.rmse_duration, stack_dims, coord_map
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
        return xr.concat(
            arrays, dim=_make_dim_index(slice_coords, slice_dims[0])
        )
    it = iter(arrays)

    def _nest(
        dims: list[str],
    ) -> list[Any]:
        if len(dims) == 1:
            return [next(it) for _ in slice_coords[dims[0]]]
        return [_nest(dims[1:]) for _ in slice_coords[dims[0]]]

    nested: Any = _nest(slice_dims)
    for i, dim in reversed(list(enumerate(slice_dims))):
        idx = _make_dim_index(slice_coords, dim)
        if i == len(slice_dims) - 1:
            nested = [
                xr.concat(group, dim=idx) for group in nested
            ]
        else:
            nested = xr.concat(nested, dim=idx)
    return nested  # type: ignore[no-any-return]


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
        accuracy=AccuracyResult(
            rmse=_acc_field("rmse"),
            mae=_acc_field("mae"),
            rmse_duration=_acc_field("rmse_duration"),
        ),
        reconstructed=_field("reconstructed"),
        original=_field("original"),
        raw=raw_map,
    )
