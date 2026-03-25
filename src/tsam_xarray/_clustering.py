"""Clustering IO and apply for tsam_xarray."""

from __future__ import annotations

import json
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xarray as xr
from tsam import ClusteringResult

from tsam_xarray._core import (
    _concat_results,
    _infer_slice_dims,
    _resolve_cluster_dim,
)


@dataclass(frozen=True)
class ClusteringInfo:
    """Reusable clustering with xarray dimension metadata.

    Wraps one or more tsam ``ClusteringResult`` objects alongside
    the dimension names needed to apply the clustering to new data.
    """

    time_dim: str
    cluster_dim: list[str]
    clusterings: dict[tuple[str, ...], ClusteringResult]
    """Per-slice clustering. Single entry ``{(): result}`` when no slicing."""

    def apply(
        self,
        da: xr.DataArray,
        *,
        time_dim: str | None = None,
        cluster_dim: Sequence[str] | str | None = None,
        **tsam_kwargs: Any,
    ) -> Any:
        """Apply this clustering to new data.

        Parameters
        ----------
        da : xr.DataArray
            New data with compatible time dimension length.
        time_dim : str | None
            Time dimension name. Defaults to the stored value.
        cluster_dim : Sequence[str] | str | None
            Cluster dimension(s). Defaults to the stored value.
            Can differ from the original if the new data has
            different dimension names.
        **tsam_kwargs
            Additional keyword arguments passed to
            ``ClusteringResult.apply()``.

        Returns
        -------
        AggregationResult
            Aggregation result using the stored clustering.
        """
        from tsam_xarray._result import AggregationResult

        td = time_dim if time_dim is not None else self.time_dim
        cd = (
            _resolve_cluster_dim(cluster_dim)
            if cluster_dim is not None
            else self.cluster_dim
        )

        _validate_apply(da, td, cd, self.clusterings)

        slice_dims = _infer_slice_dims(da, td, cd)

        if not slice_dims:
            cr = self.clusterings[()]
            return _apply_single(da, cr, td, cd, tsam_kwargs)

        import itertools

        slice_coords: dict[str, Any] = {d: da.coords[d].values for d in slice_dims}
        slice_keys = list(itertools.product(*(slice_coords[d] for d in slice_dims)))

        results: list[AggregationResult] = []
        raw_map: dict[tuple[Hashable, ...], Any] = {}

        for key in slice_keys:
            sel = dict(zip(slice_dims, key, strict=True))
            da_slice = da.sel(sel)
            str_key = tuple(str(k) for k in key)
            cr = self.clusterings[str_key]
            r = _apply_single(da_slice, cr, td, cd, tsam_kwargs)
            results.append(r)
            raw_map[key] = r.raw

        return _concat_results(results, slice_dims, slice_coords, raw_map)

    def to_json(self, path: str | Path) -> None:
        """Save clustering to JSON file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        data: dict[str, Any] = {
            "time_dim": self.time_dim,
            "cluster_dim": self.cluster_dim,
            "clusterings": {},
        }
        for key, cr in self.clusterings.items():
            str_key = "|".join(key) if key else ""
            data["clusterings"][str_key] = cr.to_dict()

        with Path(path).open("w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> ClusteringInfo:
        """Load clustering from JSON file.

        Parameters
        ----------
        path : str or Path
            Input file path.

        Returns
        -------
        ClusteringInfo
        """
        with Path(path).open() as f:
            data = json.load(f)

        clusterings: dict[tuple[str, ...], ClusteringResult] = {}
        for str_key, cr_dict in data["clusterings"].items():
            key = tuple(str_key.split("|")) if str_key else ()
            clusterings[key] = ClusteringResult.from_dict(cr_dict)

        return cls(
            time_dim=data["time_dim"],
            cluster_dim=data["cluster_dim"],
            clusterings=clusterings,
        )


def _validate_apply(
    da: xr.DataArray,
    time_dim: str,
    col_dims: list[str],
    clusterings: dict[tuple[str, ...], ClusteringResult],
) -> None:
    """Validate data is compatible with stored clustering."""
    # time_dim must exist
    if time_dim not in da.dims:
        msg = f"time_dim {time_dim!r} not in DataArray dims {set(da.dims)}"
        raise ValueError(msg)

    # cluster_dim must exist
    for d in col_dims:
        if d not in da.dims:
            msg = f"cluster_dim {d!r} not in DataArray dims {set(da.dims)}"
            raise ValueError(msg)

    # Slice dims must match stored clustering keys
    slice_dims = _infer_slice_dims(da, time_dim, col_dims)
    if slice_dims:
        import itertools

        slice_coords = {d: da.coords[d].values for d in slice_dims}
        data_keys = {
            tuple(str(k) for k in key)
            for key in itertools.product(*(slice_coords[d] for d in slice_dims))
        }
        stored_keys = set(clusterings.keys())
        missing = data_keys - stored_keys
        if missing:
            msg = (
                f"No stored clustering for slice coordinates: "
                f"{missing}. Stored keys: {stored_keys}"
            )
            raise ValueError(msg)
    elif () not in clusterings:
        msg = (
            "Data has no slice dims but clustering was "
            "created with slicing. Provide data with matching "
            "slice dimensions."
        )
        raise ValueError(msg)


def _apply_single(
    da: xr.DataArray,
    cr: ClusteringResult,
    time_dim: str,
    col_dims: list[str],
    tsam_kwargs: dict[str, Any],
) -> Any:
    """Apply a single ClusteringResult to a DataArray."""
    import numpy as np
    import pandas as pd

    from tsam_xarray._core import (
        _metric_to_da,
        _reconstructed_to_da,
        _representatives_to_da,
        _segment_durations_to_da,
        _to_dataframe,
    )
    from tsam_xarray._result import AccuracyMetrics, AggregationResult

    df = _to_dataframe(da, time_dim, col_dims)
    tsam_result = cr.apply(df, **tsam_kwargs)

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

    seg_durations = _segment_durations_to_da(tsam_result.segment_durations)

    return AggregationResult(
        typical_periods=typical,
        cluster_assignments=assignments_da,
        cluster_weights=cluster_weights_da,
        segment_durations=seg_durations,
        accuracy=accuracy,
        reconstructed=reconstructed,
        original=da,
        raw=tsam_result,
        _time_dim=time_dim,
        _cluster_dim=col_dims,
    )
