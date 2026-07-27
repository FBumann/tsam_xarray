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
from tsam_xarray._dim_names import DimNames


@dataclass(frozen=True, repr=False)
class ClusteringResult:
    """Reusable clustering result with xarray dimension metadata.

    Wraps one or more tsam ``ClusteringResult`` objects alongside
    the dimension names needed to apply the clustering to new data.
    Exposes clustering metadata as cached xarray DataArrays.

    Attributes:
        time_dim: Name of the time dimension.
        cluster_dim: Dimension(s) clustered together.
        slice_dims: Dimension(s) aggregated independently.
        clusterings: Per-slice tsam clustering.
            Single entry ``{(): result}`` when no slicing.
        n_clusters: Number of clusters.
        n_original_periods: Number of original periods.
        n_timesteps_per_period: Timesteps per period.
        n_segments: Segments per period, or ``None``.
        cluster_assignments: Cluster ID per period.
            Dims: ``(period, *slice_dims)``.
        cluster_occurrences: Periods per cluster.
            Dims: ``(cluster, *slice_dims)``.
        cluster_centers: Representative period per cluster.
            Dims: ``(cluster, *slice_dims)``.
        segment_durations: Duration per segment, or ``None``.
            Dims: ``(cluster, timestep, *slice_dims)``.
        segment_assignments: Segment ID per timestep, or
            ``None``. Dims: ``(cluster, timestep,
            *slice_dims)``.
        segment_centers: Representative timestep per segment,
            or ``None``.
            Dims: ``(cluster, segment, *slice_dims)``.
        dim_names: Names of the structural output dimensions.
            See `DimNames`.
    """

    time_dim: str
    cluster_dim: list[str]
    slice_dims: list[str]
    clusterings: dict[tuple[Hashable, ...], tsam.ClusteringResult]
    dim_names: DimNames = field(default_factory=DimNames)
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
            return xr.DataArray(
                list(cr.cluster_assignments), dims=[self.dim_names.period]
            )

        import itertools

        sc = self._slice_coords
        keys = list(itertools.product(*(sc[d] for d in self.slice_dims)))
        arrays = [
            xr.DataArray(
                list(self.clusterings[k].cluster_assignments),
                dims=[self.dim_names.period],
            )
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
                dims=[self.dim_names.cluster],
                coords={self.dim_names.cluster: np.arange(cr.n_clusters)},
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
            return _segment_durations_to_da(
                self.clusterings[()].segment_durations, self.dim_names
            )

        import itertools

        sc = self._slice_coords
        keys = list(itertools.product(*(sc[d] for d in self.slice_dims)))
        first = _segment_durations_to_da(
            self.clusterings[keys[0]].segment_durations, self.dim_names
        )
        if first is None:
            return None
        das: list[xr.DataArray] = [first]
        for k in keys[1:]:
            da = _segment_durations_to_da(
                self.clusterings[k].segment_durations, self.dim_names
            )
            if da is None:
                msg = (
                    f"Slice {k} has no segment durations but the first "
                    f"slice does. Segmentation must be uniform across slices."
                )
                raise ValueError(msg)
            das.append(da)
        return _concat_along_dims(das, self.slice_dims, sc)

    @property
    def cluster_centers(self) -> xr.DataArray:
        """Representative period index for each cluster.

        Dims: ``(cluster, *slice_dims)``.
        """
        if "cluster_centers" not in self._cache:
            self._cache["cluster_centers"] = self._build_cluster_centers()
        result: xr.DataArray = self._cache["cluster_centers"]
        return result

    def _build_cluster_centers(self) -> xr.DataArray:
        def _single(cr: tsam.ClusteringResult) -> xr.DataArray:
            centers = cr.cluster_centers
            if centers is None:
                msg = "No cluster centers available."
                raise ValueError(msg)
            return xr.DataArray(
                list(centers),
                dims=[self.dim_names.cluster],
                coords={self.dim_names.cluster: np.arange(cr.n_clusters)},
            )

        if not self.slice_dims:
            return _single(self.clusterings[()])

        import itertools

        sc = self._slice_coords
        keys = list(itertools.product(*(sc[d] for d in self.slice_dims)))
        arrays = [_single(self.clusterings[k]) for k in keys]
        return _concat_along_dims(arrays, self.slice_dims, sc)

    @property
    def segment_assignments(self) -> xr.DataArray | None:
        """Segment assignment for each timestep per cluster, or None.

        Dims: ``(cluster, timestep, *slice_dims)``.
        """
        if "segment_assignments" not in self._cache:
            self._cache["segment_assignments"] = self._build_segment_assignments()
        result: xr.DataArray | None = self._cache["segment_assignments"]
        return result

    def _build_segment_assignments(self) -> xr.DataArray | None:
        def _single(cr: tsam.ClusteringResult) -> xr.DataArray | None:
            if cr.segment_assignments is None:
                return None
            return xr.DataArray(
                np.array(cr.segment_assignments),
                dims=[self.dim_names.cluster, self.dim_names.timestep],
                coords={
                    self.dim_names.cluster: np.arange(cr.n_clusters),
                    self.dim_names.timestep: np.arange(cr.n_timesteps_per_period),
                },
            )

        if not self.slice_dims:
            return _single(self.clusterings[()])

        import itertools

        sc = self._slice_coords
        keys = list(itertools.product(*(sc[d] for d in self.slice_dims)))
        first = _single(self.clusterings[keys[0]])
        if first is None:
            return None
        das: list[xr.DataArray] = [first]
        for k in keys[1:]:
            da = _single(self.clusterings[k])
            if da is None:
                msg = (
                    f"Slice {k} has no segment assignments but the first "
                    f"slice does. Segmentation must be uniform across slices."
                )
                raise ValueError(msg)
            das.append(da)
        return _concat_along_dims(das, self.slice_dims, sc)

    @property
    def segment_centers(self) -> xr.DataArray | None:
        """Representative timestep index for each segment per cluster, or None.

        Dims: ``(cluster, segment, *slice_dims)``.
        """
        if "segment_centers" not in self._cache:
            self._cache["segment_centers"] = self._build_segment_centers()
        result: xr.DataArray | None = self._cache["segment_centers"]
        return result

    def _build_segment_centers(self) -> xr.DataArray | None:
        def _single(cr: tsam.ClusteringResult) -> xr.DataArray | None:
            if cr.segment_centers is None:
                return None
            n_segments = cr.n_segments or len(cr.segment_centers[0])
            return xr.DataArray(
                np.array(cr.segment_centers),
                dims=[self.dim_names.cluster, self.dim_names.segment],
                coords={
                    self.dim_names.cluster: np.arange(cr.n_clusters),
                    self.dim_names.segment: np.arange(n_segments),
                },
            )

        if not self.slice_dims:
            return _single(self.clusterings[()])

        import itertools

        sc = self._slice_coords
        keys = list(itertools.product(*(sc[d] for d in self.slice_dims)))
        first = _single(self.clusterings[keys[0]])
        if first is None:
            return None
        das: list[xr.DataArray] = [first]
        for k in keys[1:]:
            da = _single(self.clusterings[k])
            if da is None:
                msg = (
                    f"Slice {k} has no segment centers but the first "
                    f"slice does. Segmentation must be uniform across slices."
                )
                raise ValueError(msg)
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

        Args:
            da: New data with compatible time dimension
                length.
            time_dim: Time dimension name. Defaults to the
                stored value.
            cluster_dim: Cluster dimension(s). Defaults to the
                stored value. Can differ from the original if
                the new data has different dimension names.
            **tsam_kwargs: Additional keyword arguments passed
                to ``ClusteringResult.apply()``.

        Returns:
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
            return _apply_single(da, cr, td, cd, tsam_kwargs, self.dim_names)

        import itertools

        slice_coords: dict[str, Any] = {d: da.coords[d].values for d in slice_dims}
        slice_keys = list(itertools.product(*(slice_coords[d] for d in slice_dims)))

        results: list[AggregationResult] = []

        for key in slice_keys:
            sel = dict(zip(slice_dims, key, strict=True))
            da_slice = da.sel(sel)
            cr = _lookup_clustering(self.clusterings, key)
            r = _apply_single(da_slice, cr, td, cd, tsam_kwargs, self.dim_names)
            results.append(r)

        return _concat_results(results, slice_dims, slice_coords, slice_keys)

    def disaggregate(self, data: xr.DataArray) -> xr.DataArray:
        """Map data on ``(cluster, timestep)`` back to original time.

        This is the inverse of ``aggregate()``. Use it to expand
        data computed on the compact cluster-representative grid
        (e.g., optimization results) back to the full time axis.

        Unlike ``AggregationResult.disaggregate()``, this method
        works on a ``ClusteringInfo`` loaded from JSON — no
        original data needed.

        Args:
            data: Data with ``cluster`` and ``timestep`` dims,
                matching the shape of the original cluster
                representatives. Additional dims (including
                auto-sliced dims like scenario) are supported.

        Returns:
            Data with ``cluster`` and ``timestep`` replaced by
            the original ``time`` dimension.
        """
        import itertools

        slice_dims = self.slice_dims
        slice_coords = {d: data.coords[d].values for d in slice_dims}
        keys = list(itertools.product(*(slice_coords[d] for d in slice_dims)))
        crs = [_lookup_clustering(self.clusterings, key) for key in keys]

        if _is_gatherable(crs, data, slice_dims):
            return _disaggregate_gather(
                crs, data, self.dim_names, slice_dims, slice_coords
            )

        if not slice_dims:
            return _disaggregate_single(self.clusterings[()], data, self.dim_names)

        results = []
        for key, cr in zip(keys, crs, strict=True):
            sel = dict(zip(slice_dims, key, strict=True))
            data_slice = data.sel(sel)
            results.append(_disaggregate_single(cr, data_slice, self.dim_names))

        return _concat_along_dims(results, slice_dims, slice_coords)

    def to_dict(self) -> dict[str, Any]:
        """Serialize clustering to a dictionary.

        Returns:
            Plain dict suitable for ``json.dump()`` or
            storage in databases, APIs, etc.
        """
        entries = []
        for key, cr in self.clusterings.items():
            entries.append(
                {
                    "key": list(_native_key(key)),
                    "clustering": cr.to_dict(),
                }
            )
        return {
            "time_dim": self.time_dim,
            "cluster_dim": self.cluster_dim,
            "slice_dims": self.slice_dims,
            "dim_names": {
                "cluster": self.dim_names.cluster,
                "timestep": self.dim_names.timestep,
                "period": self.dim_names.period,
                "segment": self.dim_names.segment,
            },
            "clusterings": entries,
        }

    def to_json(self, path: str | Path, **json_kwargs: Any) -> None:
        """Save clustering to JSON file.

        Args:
            path: Output file path.
            **json_kwargs: Additional keyword arguments passed
                to ``json.dump()``. Default: ``indent=2``.
        """
        with Path(path).open("w") as f:
            json.dump(self.to_dict(), f, **json_kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClusteringResult:
        """Load clustering from a dictionary.

        Args:
            data: Dict as returned by :meth:`to_dict`.

        Returns:
            The loaded ``ClusteringResult``.
        """
        # Backcompat: pre-0.6 wrappers stored the time index as an outer
        # ``time_coords`` key while the inner tsam blob (written by tsam<3.4)
        # had no ``time_index``. Forward it so disaggregate keeps datetimes.
        if "time_coords" in data:
            import warnings

            warnings.warn(
                "Loading a legacy tsam_xarray JSON with an outer 'time_coords' "
                "field; re-save with to_json() to silence this warning.",
                DeprecationWarning,
                stacklevel=2,
            )
            for entry in data["clusterings"]:
                entry["clustering"].setdefault("time_index", data["time_coords"])

        clusterings: dict[tuple[Hashable, ...], tsam.ClusteringResult] = {}
        for entry in data["clusterings"]:
            key = tuple(entry["key"])
            clusterings[key] = tsam.ClusteringResult.from_dict(entry["clustering"])

        dim_names_data = data.get("dim_names")
        dim_names = DimNames(**dim_names_data) if dim_names_data else DimNames()

        return cls(
            time_dim=data["time_dim"],
            cluster_dim=data["cluster_dim"],
            slice_dims=data.get("slice_dims", []),
            clusterings=clusterings,
            dim_names=dim_names,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ClusteringResult:
        """Load clustering from JSON file.

        Args:
            path: Input file path.

        Returns:
            The loaded ``ClusteringResult``.
        """
        with Path(path).open() as f:
            return cls.from_dict(json.load(f))


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


def _drop_missing_weights(
    cr: tsam.ClusteringResult, df: pd.DataFrame
) -> tsam.ClusteringResult:
    """Strip stored clustering weights when they reference absent columns.

    tsam's ``ClusteringResult.apply()`` hard-fails when a stored weight column
    is missing from the new data, which blocks transferring a clustering to a
    differently-composed dataset. At apply time the cluster assignments and
    centers are already fixed, so weights cannot change which periods are
    selected — their only remaining effect is on the weighted accuracy
    metrics. When a weighted column is absent we drop all weights and warn.

    Tracking upstream fix: https://github.com/FZJ-IEK3-VSA/tsam/issues/396.
    Once that lands, ``test_upstream_still_rejects_missing_weight_columns``
    fails and this workaround can be removed.
    """
    import dataclasses
    import warnings

    weights = getattr(cr, "weights", None)
    if not weights:
        return cr
    missing = set(weights) - set(df.columns)
    if not missing:
        return cr

    warnings.warn(
        f"Stored clustering weights reference columns absent from the new "
        f"data ({sorted(map(str, missing))}); dropping all weights. "
        f"Representatives and reconstruction are unaffected — only the "
        f"weighted accuracy metrics change.",
        UserWarning,
        stacklevel=3,
    )
    return dataclasses.replace(cr, weights=None)


def _apply_single(
    da: xr.DataArray,
    cr: tsam.ClusteringResult,
    time_dim: str,
    col_dims: list[str],
    tsam_kwargs: dict[str, Any],
    dim_names: DimNames,
) -> Any:
    """Apply a single ClusteringResult to a DataArray."""
    import pandas as pd

    from tsam_xarray._core import (
        _cluster_counts,
        _metric_to_da,
        _reconstructed_to_da,
        _representatives_to_da,
        _segment_durations_to_da,
        _to_dataframe,
    )
    from tsam_xarray._result import AccuracyMetrics, AggregationResult

    df = _to_dataframe(da, time_dim, col_dims)
    cr = _drop_missing_weights(cr, df)
    tsam_result = cr.apply(df, **tsam_kwargs)

    typical = _representatives_to_da(
        tsam_result.cluster_representatives, col_dims, dim_names
    )

    def _make_reconstructed() -> xr.DataArray:
        return _reconstructed_to_da(tsam_result.reconstructed, time_dim, col_dims)

    cw = _cluster_counts(tsam_result)
    cluster_ids = np.array(sorted(cw.keys()))
    cluster_counts_da = xr.DataArray(
        np.array([cw[k] for k in cluster_ids]),
        dims=[dim_names.cluster],
        coords={dim_names.cluster: cluster_ids},
    )

    assignments_da = xr.DataArray(
        tsam_result.cluster_assignments, dims=[dim_names.period]
    )

    col_names: list[str] | None = None
    if isinstance(df.columns, pd.MultiIndex):
        col_names = [str(n) for n in df.columns.names]

    def _make_accuracy() -> AccuracyMetrics:
        return AccuracyMetrics(
            rmse=_metric_to_da(tsam_result.accuracy.rmse, col_dims, col_names),
            mae=_metric_to_da(tsam_result.accuracy.mae, col_dims, col_names),
            rmse_duration=_metric_to_da(
                tsam_result.accuracy.rmse_duration, col_dims, col_names
            ),
            weighted_rmse=xr.DataArray(tsam_result.accuracy.weighted_rmse),
            weighted_mae=xr.DataArray(tsam_result.accuracy.weighted_mae),
            weighted_rmse_duration=xr.DataArray(
                tsam_result.accuracy.weighted_rmse_duration
            ),
        )

    seg_durations = _segment_durations_to_da(tsam_result.segment_durations, dim_names)

    clustering_info = ClusteringResult(
        time_dim=time_dim,
        cluster_dim=col_dims,
        slice_dims=[],
        clusterings={(): tsam_result.clustering},
        dim_names=dim_names,
    )

    return AggregationResult(
        cluster_representatives=typical,
        cluster_assignments=assignments_da,
        cluster_counts=cluster_counts_da,
        segment_durations=seg_durations,
        _accuracy_factory=_make_accuracy,
        _reconstructed_factory=_make_reconstructed,
        original=da,
        clustering=clustering_info,
        is_transferred=True,
    )


def _is_gatherable(
    crs: list[tsam.ClusteringResult],
    data: xr.DataArray,
    slice_dims: list[str],
) -> bool:
    """Whether all slices can be disaggregated by one vectorized gather.

    The gather produces a single rectangular array with one shared time axis,
    so every slice must agree on period count, period length and time index.
    Segmented clusterings additionally need a segment grid that is rectangular
    and covers each period exactly once.

    Args:
        crs: Per-slice clusterings, in the order the slices appear
            along ``slice_dims``.
        data: The payload to disaggregate.
        slice_dims: Dimension(s) aggregated independently.

    Returns:
        ``True`` if one vectorized pass covers every slice, ``False`` to
        fall back to the per-slice tsam path.
    """
    if any(d not in data.dims for d in slice_dims):
        return False
    first = crs[0]
    segmented = first.segment_durations is not None
    return all(
        (cr.segment_durations is not None) == segmented
        and cr.n_timesteps_per_period == first.n_timesteps_per_period
        and len(cr.cluster_assignments) == len(first.cluster_assignments)
        and _same_time_index(cr.time_index, first.time_index)
        and (not segmented or _has_regular_segments(cr))
        for cr in crs
    )


def _has_regular_segments(cr: tsam.ClusteringResult) -> bool:
    """Whether every cluster has the same segment count and tiles its period.

    Ragged segment counts break the rectangular scatter. Zero-length segments
    would make two segments share a start timestep, where the scatter's
    last-write-wins would have to match the pandas loop's; rather than rely on
    that, such clusterings go the pandas way.

    Args:
        cr: The clustering whose segment grid is checked.

    Returns:
        ``True`` if the scatter can express this clustering's segments.
    """
    durations = cr.segment_durations
    if durations is None or not durations:
        return False
    n_segments = len(durations[0])
    return all(
        len(d) == n_segments
        and all(step > 0 for step in d)
        and sum(d) == cr.n_timesteps_per_period
        for d in durations
    )


def _same_time_index(a: pd.Index | None, b: pd.Index | None) -> bool:
    """Whether two stored time indices would produce the same time axis.

    Args:
        a: A clustering's stored time index, or ``None``.
        b: The index to compare against, or ``None``.

    Returns:
        ``True`` if both are absent or hold equal values.
    """
    if a is None or b is None:
        return a is None and b is None
    return len(a) == len(b) and bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _validate_gather_input(
    cr: tsam.ClusteringResult,
    clusters: np.ndarray,
    n_steps: int,
) -> None:
    """Mirror tsam's disaggregate input checks for the vectorized path."""
    expected = set(cr.cluster_assignments)
    got = set(np.asarray(clusters).tolist())
    if got != expected:
        parts = []
        if expected - got:
            parts.append(f"missing clusters {sorted(expected - got)}")
        if got - expected:
            parts.append(f"unexpected clusters {sorted(got - expected)}")
        msg = (
            f"Cluster IDs in data do not match this clustering: "
            f"{', '.join(parts)}. "
            f"Expected {sorted(expected)}, got {sorted(got)}."
        )
        raise ValueError(msg)

    segmented = cr.segment_durations is not None
    kind = "segments" if segmented else "timesteps"
    n_expected = cr.n_segments if segmented else cr.n_timesteps_per_period
    if n_steps != n_expected:
        msg = f"data has {n_steps} {kind} per cluster, expected {n_expected}"
        raise ValueError(msg)


def _segment_starts(
    cr: tsam.ClusteringResult,
    cluster_ranks: np.ndarray,
) -> np.ndarray:
    """First timestep of each segment, per cluster, in payload cluster order.

    ``segment_durations`` is keyed by sorted cluster ID while the payload's
    cluster axis may run in any order, so it is looked up by label rank —
    the same mapping tsam's ``_expand_segments_to_timesteps`` builds.
    """
    durations = np.asarray(cr.segment_durations, dtype=np.intp)[cluster_ranks]
    starts: np.ndarray = np.zeros_like(durations)
    np.cumsum(durations[:, :-1], axis=1, out=starts[:, 1:])
    return starts


def _expand_segments(
    values: np.ndarray,
    starts: np.ndarray,
    n_timesteps: int,
) -> np.ndarray:
    """Scatter segment values onto a full timestep axis, NaN elsewhere.

    ``values`` is ``(slice, cluster, segment, other)``; the result is
    ``(slice, cluster, timestep, other)``. Only the first timestep of each
    segment carries a value, matching tsam — callers ``ffill`` for a step
    function. Always float64, as the NaN fill forces upcasting.
    """
    n_slices, n_clusters = values.shape[:2]
    expanded = np.full(
        (n_slices, n_clusters, n_timesteps, values.shape[3]), np.nan, dtype=np.float64
    )
    slice_idx = np.arange(n_slices)[:, None, None]
    cluster_idx = np.arange(n_clusters)[None, :, None]
    expanded[slice_idx, cluster_idx, starts] = values
    return expanded


def _disaggregate_gather(
    crs: list[tsam.ClusteringResult],
    data: xr.DataArray,
    dim_names: DimNames,
    slice_dims: list[str],
    slice_coords: dict[str, Any],
) -> xr.DataArray:
    """Disaggregate every slice with one vectorized numpy pass.

    Expanding a clustering is a gather along the cluster axis: every original
    period takes the values of its assigned representative. tsam's
    ``disaggregate()`` expresses this as an ``unstack``/``.loc``/``stack``
    round-trip through pandas, and segmented input first goes through a
    per-cluster DataFrame loop; both dominate the runtime and both are
    repeated once per slice. In numpy the segment expansion is a scatter and
    the period expansion a take, covering all slices at once.

    Output dim order matches the per-slice path: ``(*slice_dims, time,
    *other_dims)``.

    The payload is transposed straight into the layout the gather reads and
    the output is written in, then made contiguous. Gathering from a strided
    view of an awkwardly ordered payload costs several times more than the
    copy does, and the copy is free when the payload already has that layout.
    Segmented input skips it, since the scatter allocates a fresh buffer in
    that layout anyway.
    """
    cluster_dim = dim_names.cluster
    step_dim = dim_names.timestep
    other_dims = [
        str(d)
        for d in data.dims
        if d not in (cluster_dim, step_dim) and d not in slice_dims
    ]
    ordered = data.transpose(*slice_dims, cluster_dim, step_dim, *other_dims)

    clusters = ordered.coords[cluster_dim].values
    n_clusters = len(clusters)
    n_steps = ordered.sizes[step_dim]
    slice_sizes = tuple(ordered.sizes[d] for d in slice_dims)
    other_sizes = ordered.shape[len(slice_dims) + 2 :]
    n_slices = int(np.prod(slice_sizes, dtype=int))
    n_periods = len(crs[0].cluster_assignments)
    n_timesteps = crs[0].n_timesteps_per_period

    positions = pd.Index(clusters)
    cluster_ranks = np.argsort(np.argsort(clusters, kind="stable"), kind="stable")
    index = np.empty((n_slices, n_periods), dtype=np.intp)
    starts = np.empty((n_slices, n_clusters, n_steps), dtype=np.intp)
    for i, cr in enumerate(crs):
        _validate_gather_input(cr, clusters, n_steps)
        index[i] = positions.get_indexer(np.asarray(cr.cluster_assignments))
        if cr.segment_durations is not None:
            starts[i] = _segment_starts(cr, cluster_ranks)

    values = ordered.values.reshape(n_slices, n_clusters, n_steps, -1)
    if crs[0].segment_durations is not None:
        values = _expand_segments(values, starts, n_timesteps)
    else:
        values = np.ascontiguousarray(values)

    gathered = values[np.arange(n_slices)[:, None], index]
    gathered = gathered.reshape(*slice_sizes, n_periods * n_timesteps, *other_sizes)

    n_time = n_periods * n_timesteps
    stored = crs[0].time_index
    time_index: pd.Index = (
        stored
        if stored is not None and len(stored) == n_time
        else pd.RangeIndex(n_time)
    )

    result = xr.DataArray(
        gathered,
        dims=[*slice_dims, "time", *other_dims],
        coords={"time": time_index},
    )
    for d in (*slice_dims, *other_dims):
        if d in slice_coords:
            result = result.assign_coords({d: slice_coords[d]})
        elif d in data.coords:
            result = result.assign_coords({d: data.coords[d]})
    return result


def _disaggregate_single(
    cr: tsam.ClusteringResult,
    data: xr.DataArray,
    dim_names: DimNames,
) -> xr.DataArray:
    """Disaggregate one slice through tsam's ``cr.disaggregate()``.

    The fallback for clusterings the vectorized gather declines; see
    :func:`_is_gatherable`. Returns a DataFrame indexed by the original
    ``DatetimeIndex`` stored on the clustering.

    Segmented input needs a third index level to mark it as segment-level
    data, but tsam drops that level before use and reads the durations off
    the clustering itself, so it is filled with zeros rather than with
    durations looked up by cluster label.
    """
    cluster_dim = dim_names.cluster
    timestep_dim = dim_names.timestep
    other_dims = [str(d) for d in data.dims if d not in (cluster_dim, timestep_dim)]
    ordered = data.transpose(cluster_dim, timestep_dim, *other_dims)

    clusters = ordered.coords[cluster_dim].values
    n_clusters = len(clusters)
    n_timesteps = ordered.sizes[timestep_dim]
    other_sizes = ordered.shape[2:]

    flat = ordered.values.reshape(n_clusters * n_timesteps, -1)

    if cr.segment_durations is not None:
        mi = pd.MultiIndex.from_arrays(
            [
                np.repeat(clusters, n_timesteps),
                np.tile(np.arange(n_timesteps), n_clusters),
                np.zeros(n_clusters * n_timesteps, dtype=int),
            ],
            names=["cluster", "segment", "duration"],
        )
    else:
        mi = pd.MultiIndex.from_product(
            [clusters, range(n_timesteps)], names=["cluster", "timestep"]
        )

    df = pd.DataFrame(flat, index=mi, columns=range(flat.shape[1]))
    expanded = cr.disaggregate(df)
    time_coords = expanded.index

    if other_dims:
        vals = expanded.values.reshape(len(time_coords), *other_sizes)
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
            expanded.values[:, 0],
            dims=["time"],
            coords={"time": time_coords},
        )

    return result
