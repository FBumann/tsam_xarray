"""Clustering IO and apply for tsam_xarray."""

from __future__ import annotations

import json
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tsam
import xarray as xr

from tsam_xarray._core import (
    _concat_along_dims,
    _concat_results,
    _resolve_cluster_dim,
    _segment_durations_to_da,
)


@dataclass(frozen=True, repr=False)
class ClusteringResult:
    """Reusable clustering result with xarray dimension metadata.

    Wraps one or more tsam ``ClusteringResult`` objects alongside
    the dimension names needed to apply the clustering to new data.
    Exposes clustering metadata as xarray DataArrays.
    """

    time_dim: str
    cluster_dim: list[str]
    slice_dims: list[str]
    clusterings: dict[tuple[Hashable, ...], tsam.ClusteringResult]
    """Per-slice tsam clustering. Single entry ``{(): result}`` when no slicing."""
    time_coords: pd.DatetimeIndex | None = field(default=None, repr=False)
    """Original time coordinates. Needed for :meth:`disaggregate`."""
    _cache: dict[str, Any] = field(
        default_factory=dict, repr=False, init=False, compare=False
    )

    def __repr__(self) -> str:
        seg = f", n_segments={self.n_segments}" if self.n_segments else ""
        slices = f", slice_dims={self.slice_dims}" if self.slice_dims else ""
        return (
            f"ClusteringResult("
            f"n_clusters={self.n_clusters}, "
            f"n_periods={self.n_original_periods}, "
            f"timesteps_per_period={self.n_timesteps_per_period}, "
            f"time_dim={self.time_dim!r}, "
            f"cluster_dim={self.cluster_dim}"
            f"{slices}{seg})"
        )

    # -- scalar accessors (uniform across slices) --

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        return next(iter(self.clusterings.values())).n_clusters

    @property
    def n_original_periods(self) -> int:
        """Number of original periods (e.g., days)."""
        return next(iter(self.clusterings.values())).n_original_periods

    @property
    def n_timesteps_per_period(self) -> int:
        """Number of timesteps per period (e.g., 24 for hourly with daily periods)."""
        return next(iter(self.clusterings.values())).n_timesteps_per_period

    @property
    def n_segments(self) -> int | None:
        """Number of segments per period, or None if no segmentation."""
        return next(iter(self.clusterings.values())).n_segments

    # -- DataArray properties (cached, concatenated across slices) --

    @property
    def _slice_coords(self) -> dict[str, Any]:
        """Reconstruct slice coordinates from clusterings keys."""
        if not self.slice_dims:
            return {}
        keys = list(self.clusterings.keys())
        return {
            dim: list(dict.fromkeys(k[i] for k in keys))
            for i, dim in enumerate(self.slice_dims)
        }

    @property
    def cluster_assignments(self) -> xr.DataArray:
        """Cluster assignment for each period, as DataArray.

        Dims: ``(period, *slice_dims)``.
        """
        if "cluster_assignments" not in self._cache:
            self._cache["cluster_assignments"] = self._build_assignments()
        result: xr.DataArray = self._cache["cluster_assignments"]
        return result

    def _build_assignments(self) -> xr.DataArray:
        if not self.slice_dims:
            cr = self.clusterings[()]
            return xr.DataArray(list(cr.cluster_assignments), dims=["period"])

        import itertools

        sc = self._slice_coords
        keys = list(itertools.product(*(sc[d] for d in self.slice_dims)))
        arrays = [
            xr.DataArray(list(self.clusterings[k].cluster_assignments), dims=["period"])
            for k in keys
        ]
        return _concat_along_dims(arrays, self.slice_dims, sc)

    @property
    def cluster_occurrences(self) -> xr.DataArray:
        """Number of periods assigned to each cluster.

        Dims: ``(cluster, *slice_dims)``.
        """
        if "cluster_occurrences" not in self._cache:
            self._cache["cluster_occurrences"] = self._build_occurrences()
        result: xr.DataArray = self._cache["cluster_occurrences"]
        return result

    def _build_occurrences(self) -> xr.DataArray:
        def _single(cr: tsam.ClusteringResult) -> xr.DataArray:
            counts = np.bincount(cr.cluster_assignments, minlength=cr.n_clusters)
            return xr.DataArray(
                counts,
                dims=["cluster"],
                coords={"cluster": np.arange(cr.n_clusters)},
            )

        if not self.slice_dims:
            return _single(self.clusterings[()])

        import itertools

        sc = self._slice_coords
        keys = list(itertools.product(*(sc[d] for d in self.slice_dims)))
        arrays = [_single(self.clusterings[k]) for k in keys]
        return _concat_along_dims(arrays, self.slice_dims, sc)

    @property
    def segment_durations(self) -> xr.DataArray | None:
        """Duration of each segment per cluster, or None if no segmentation.

        Dims: ``(cluster, timestep, *slice_dims)``.
        """
        if "segment_durations" not in self._cache:
            self._cache["segment_durations"] = self._build_segment_durations()
        result: xr.DataArray | None = self._cache["segment_durations"]
        return result

    def _build_segment_durations(self) -> xr.DataArray | None:
        if not self.slice_dims:
            return _segment_durations_to_da(self.clusterings[()].segment_durations)

        import itertools

        sc = self._slice_coords
        keys = list(itertools.product(*(sc[d] for d in self.slice_dims)))
        first = _segment_durations_to_da(self.clusterings[keys[0]].segment_durations)
        if first is None:
            return None
        das: list[xr.DataArray] = [first]
        for k in keys[1:]:
            da = _segment_durations_to_da(self.clusterings[k].segment_durations)
            assert da is not None  # uniform across slices
            das.append(da)
        return _concat_along_dims(das, self.slice_dims, sc)

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

        _validate_apply(da, td, cd, self.slice_dims, self.clusterings)

        # Use stored slice_dims for canonical ordering
        slice_dims = self.slice_dims

        if not slice_dims:
            cr = self.clusterings[()]
            return _apply_single(da, cr, td, cd, tsam_kwargs)

        import itertools

        slice_coords: dict[str, Any] = {d: da.coords[d].values for d in slice_dims}
        slice_keys = list(itertools.product(*(slice_coords[d] for d in slice_dims)))

        results: list[AggregationResult] = []

        for key in slice_keys:
            sel = dict(zip(slice_dims, key, strict=True))
            da_slice = da.sel(sel)
            cr = _lookup_clustering(self.clusterings, key)
            r = _apply_single(da_slice, cr, td, cd, tsam_kwargs)
            results.append(r)

        return _concat_results(results, slice_dims, slice_coords, slice_keys)

    def disaggregate(self, data: xr.DataArray) -> xr.DataArray:
        """Map data on ``(cluster, timestep)`` back to original time.

        This is the inverse of ``aggregate()``. Use it to expand
        data computed on the compact cluster-representative grid
        (e.g., optimization results) back to the full time axis.

        Unlike ``AggregationResult.disaggregate()``, this method works
        on a ``ClusteringInfo`` loaded from JSON — no original data needed.

        Parameters
        ----------
        data : xr.DataArray
            Data with ``cluster`` and ``timestep`` dims, matching the
            shape of the original cluster representatives. Additional dims
            (including auto-sliced dims like scenario) are supported.

        Returns
        -------
        xr.DataArray
            Data with ``cluster`` and ``timestep`` replaced by the
            original ``time`` dimension.

        Raises
        ------
        ValueError
            If time coordinates are not available (e.g., loaded
            from an old JSON that predates this feature).
        """
        if self.time_coords is None:
            msg = (
                "No time coordinates available. "
                "This ClusteringResult was loaded from a JSON file "
                "that does not contain time coordinate data. "
                "Re-run aggregate() or save from a newer version."
            )
            raise ValueError(msg)

        slice_dims = self.slice_dims
        if not slice_dims:
            cr = self.clusterings[()]
            return _disaggregate_single(
                self.time_coords,
                cr,
                data,
            )

        import itertools

        slice_coords = {d: data.coords[d].values for d in slice_dims}
        keys = list(itertools.product(*(slice_coords[d] for d in slice_dims)))
        results = []
        for key in keys:
            sel = dict(zip(slice_dims, key, strict=True))
            data_slice = data.sel(sel)
            cr = _lookup_clustering(self.clusterings, key)
            results.append(_disaggregate_single(self.time_coords, cr, data_slice))

        return _concat_along_dims(results, slice_dims, slice_coords)

    def to_json(self, path: str | Path, **json_kwargs: Any) -> None:
        """Save clustering to JSON file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        **json_kwargs
            Additional keyword arguments passed to ``json.dump()``.
            Default: ``indent=2``.
        """
        entries = []
        for key, cr in self.clusterings.items():
            entries.append(
                {
                    "key": list(_native_key(key)),
                    "clustering": cr.to_dict(),
                }
            )
        data: dict[str, Any] = {
            "time_dim": self.time_dim,
            "cluster_dim": self.cluster_dim,
            "slice_dims": self.slice_dims,
            "clusterings": entries,
        }
        if self.time_coords is not None:
            data["time_coords"] = [t.isoformat() for t in self.time_coords]

        with Path(path).open("w") as f:
            json.dump(data, f, **json_kwargs)

    @classmethod
    def from_json(cls, path: str | Path) -> ClusteringResult:
        """Load clustering from JSON file.

        Parameters
        ----------
        path : str or Path
            Input file path.

        Returns
        -------
        ClusteringResult
        """
        with Path(path).open() as f:
            data = json.load(f)

        clusterings: dict[tuple[Hashable, ...], tsam.ClusteringResult] = {}
        for entry in data["clusterings"]:
            key = tuple(entry["key"])
            clusterings[key] = tsam.ClusteringResult.from_dict(entry["clustering"])

        time_coords: pd.DatetimeIndex | None = None
        if "time_coords" in data:
            time_coords = pd.DatetimeIndex(data["time_coords"])

        return cls(
            time_dim=data["time_dim"],
            cluster_dim=data["cluster_dim"],
            slice_dims=data.get("slice_dims", []),
            clusterings=clusterings,
            time_coords=time_coords,
        )


ClusteringInfo = ClusteringResult
"""Backwards-compatible alias for :class:`ClusteringResult`."""


def _native_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
    """Convert numpy scalars in key to Python builtins."""
    return tuple(k.item() if hasattr(k, "item") else k for k in key)


def _lookup_clustering(
    clusterings: dict[tuple[Hashable, ...], tsam.ClusteringResult],
    key: tuple[Any, ...],
) -> tsam.ClusteringResult:
    """Look up clustering by key."""
    native = _native_key(key)
    if native in clusterings:
        return clusterings[native]
    msg = f"No stored clustering for key {native}"
    raise KeyError(msg)


def _validate_apply(
    da: xr.DataArray,
    time_dim: str,
    col_dims: list[str],
    stored_slice_dims: list[str],
    clusterings: dict[tuple[Hashable, ...], tsam.ClusteringResult],
) -> None:
    """Validate data is compatible with stored clustering."""
    if time_dim not in da.dims:
        msg = f"time_dim {time_dim!r} not in DataArray dims {set(da.dims)}"
        raise ValueError(msg)

    for d in col_dims:
        if d not in da.dims:
            msg = f"cluster_dim {d!r} not in DataArray dims {set(da.dims)}"
            raise ValueError(msg)

    if stored_slice_dims:
        import itertools

        slice_coords = {d: da.coords[d].values for d in stored_slice_dims}
        for key in itertools.product(*(slice_coords[d] for d in stored_slice_dims)):
            try:
                _lookup_clustering(clusterings, key)
            except KeyError:
                msg = f"No stored clustering for slice coordinate {key}"
                raise ValueError(msg) from None
    elif () not in clusterings:
        msg = "Data has no slice dims but clustering was created with slicing."
        raise ValueError(msg)


def _apply_single(
    da: xr.DataArray,
    cr: tsam.ClusteringResult,
    time_dim: str,
    col_dims: list[str],
    tsam_kwargs: dict[str, Any],
) -> Any:
    """Apply a single ClusteringResult to a DataArray."""
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

    clustering_info = ClusteringResult(
        time_dim=time_dim,
        cluster_dim=col_dims,
        slice_dims=[],
        clusterings={(): tsam_result.clustering},
        time_coords=pd.DatetimeIndex(da.coords[time_dim].values),
    )

    return AggregationResult(
        cluster_representatives=typical,
        cluster_assignments=assignments_da,
        cluster_weights=cluster_weights_da,
        segment_durations=seg_durations,
        accuracy=accuracy,
        reconstructed=reconstructed,
        original=da,
        clustering=clustering_info,
        is_transferred=True,
    )


def _disaggregate_single(
    time_coords: pd.DatetimeIndex,
    cr: tsam.ClusteringResult,
    data: xr.DataArray,
) -> xr.DataArray:
    """Disaggregate a single (non-sliced) DataArray using a ClusteringResult."""
    other_dims = [str(d) for d in data.dims if d not in ("cluster", "timestep")]
    ordered = data.transpose("cluster", "timestep", *other_dims)

    clusters = ordered.coords["cluster"].values
    n_clusters = len(clusters)
    n_timesteps = ordered.sizes["timestep"]
    other_sizes = ordered.shape[2:]

    flat = ordered.values.reshape(n_clusters * n_timesteps, -1)

    if cr.segment_durations is not None:
        idx_tuples = []
        for c in clusters:
            for seg, dur in enumerate(cr.segment_durations[int(c)]):
                idx_tuples.append((int(c), seg, int(dur)))
        mi = pd.MultiIndex.from_tuples(
            idx_tuples, names=["cluster", "segment", "duration"]
        )
    else:
        mi = pd.MultiIndex.from_product(
            [clusters, range(n_timesteps)], names=["cluster", "timestep"]
        )

    df = pd.DataFrame(flat, index=mi, columns=range(flat.shape[1]))
    expanded = cr.disaggregate(df)

    n_original = len(time_coords)
    vals = expanded.values[:n_original]

    if other_dims:
        vals = vals.reshape(n_original, *other_sizes)
        result = xr.DataArray(
            vals,
            dims=["time", *other_dims],
            coords={"time": time_coords},
        )
        for d in other_dims:
            if d in data.coords:
                result = result.assign_coords({d: data.coords[d]})
    else:
        result = xr.DataArray(
            vals[:, 0],
            dims=["time"],
            coords={"time": time_coords},
        )

    return result
