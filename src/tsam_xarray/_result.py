"""Result dataclasses for tsam_xarray."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from tsam_xarray._clustering import ClusteringInfo


@dataclass(frozen=True)
class AccuracyMetrics:
    """Accuracy metrics from time series aggregation."""

    rmse: xr.DataArray
    mae: xr.DataArray
    rmse_duration: xr.DataArray


@dataclass(frozen=True)
class AggregationResult:
    """Result of tsam_xarray.aggregate()."""

    typical_periods: xr.DataArray
    cluster_assignments: xr.DataArray
    cluster_weights: xr.DataArray
    segment_durations: xr.DataArray | None
    accuracy: AccuracyMetrics
    reconstructed: xr.DataArray
    original: xr.DataArray
    clustering: ClusteringInfo
    is_transferred: bool = False

    @property
    def n_clusters(self) -> int:
        """Number of typical period clusters."""
        return int(self.cluster_weights.sizes["cluster"])

    @property
    def n_timesteps_per_period(self) -> int:
        """Number of timesteps per typical period."""
        return int(self.typical_periods.sizes["timestep"])

    @property
    def n_segments(self) -> int | None:
        """Number of segments per period, if segmentation was used."""
        first_cr = next(iter(self.clustering.clusterings.values()))
        result: int | None = first_cr.n_segments
        return result

    @property
    def residuals(self) -> xr.DataArray:
        """Difference between original and reconstructed data."""
        return self.original - self.reconstructed

    def disaggregate(self, data: xr.DataArray) -> xr.DataArray:
        """Map data on ``(cluster, timestep)`` back to original time.

        This is the inverse of ``aggregate()``. Use it to expand
        external data computed on the compact typical-period grid
        (e.g., optimization results) back to the full time axis.

        Without segmentation, values are repeated for each timestep
        in the period. With segmentation, values are placed at segment
        boundaries and remaining timesteps are NaN — use
        ``.ffill(dim="time")``, ``.interpolate_na(dim="time")``, etc.

        Parameters
        ----------
        data : xr.DataArray
            Data with ``cluster`` and ``timestep`` dims, matching the
            shape of ``result.typical_periods``. Additional dims
            (including auto-sliced dims like scenario) are supported.

        Returns
        -------
        xr.DataArray
            Data with ``cluster`` and ``timestep`` replaced by the
            original ``time`` dimension.
        """
        # Use stored slice_dims for canonical ordering
        slice_dims = self.clustering.slice_dims
        if not slice_dims:
            return self._disaggregate_single(data)

        import itertools

        from tsam_xarray._core import _concat_along_dims

        slice_coords = {d: data.coords[d].values for d in slice_dims}
        keys = list(itertools.product(*(slice_coords[d] for d in slice_dims)))
        results = []
        for key in keys:
            sel = dict(zip(slice_dims, key, strict=True))
            data_slice = data.sel(sel)
            result_slice = self._make_slice_view(sel)
            results.append(result_slice._disaggregate_single(data_slice))

        return _concat_along_dims(results, slice_dims, slice_coords)

    def _make_slice_view(self, sel: dict[str, object]) -> AggregationResult:
        """Create a view of this result for a single slice."""
        from tsam_xarray._clustering import (
            ClusteringInfo,
            _lookup_clustering,
        )

        # Build key in stored slice_dims order
        key = tuple(sel[d] for d in self.clustering.slice_dims)
        cr = _lookup_clustering(self.clustering.clusterings, key)

        return AggregationResult(
            typical_periods=self.typical_periods.sel(sel),
            cluster_assignments=self.cluster_assignments.sel(sel),
            cluster_weights=self.cluster_weights.sel(sel),
            segment_durations=(
                self.segment_durations.sel(sel)
                if self.segment_durations is not None
                else None
            ),
            accuracy=AccuracyMetrics(
                rmse=self.accuracy.rmse.sel(sel),
                mae=self.accuracy.mae.sel(sel),
                rmse_duration=self.accuracy.rmse_duration.sel(sel),
            ),
            reconstructed=self.reconstructed.sel(sel),
            original=self.original.sel(sel),
            clustering=ClusteringInfo(
                time_dim=self.clustering.time_dim,
                cluster_dim=self.clustering.cluster_dim,
                slice_dims=[],
                clusterings={(): cr},
            ),
        )

    def _disaggregate_single(self, data: xr.DataArray) -> xr.DataArray:
        """Disaggregate without slice dims."""
        time_coords = self.original.coords["time"]
        assignments = self.cluster_assignments.values
        n_original_timesteps = len(time_coords)
        n_periods = len(assignments)
        n_per_period = n_original_timesteps // n_periods

        other_dims = [str(d) for d in data.dims if d not in ("cluster", "timestep")]

        if self.segment_durations is None:
            expanded = data.sel(cluster=xr.DataArray(assignments, dims=["period"]))
            flat = expanded.values.reshape(-1, *expanded.shape[2:])
            result = xr.DataArray(
                flat[:n_original_timesteps],
                dims=["time", *other_dims],
                coords={"time": time_coords},
            )
            for d in other_dims:
                if d in data.coords:
                    result = result.assign_coords({d: data.coords[d]})
            return result

        other_shape = [data.sizes[d] for d in other_dims]
        total_timesteps = n_periods * n_per_period
        out = np.full([total_timesteps, *other_shape], np.nan)

        for p_idx, cluster in enumerate(assignments):
            offset = 0
            durations = self.segment_durations.sel(cluster=int(cluster)).values
            for seg_idx, dur in enumerate(durations):
                t_start = p_idx * n_per_period + offset
                vals = data.sel(cluster=int(cluster), timestep=seg_idx).values
                out[t_start] = vals
                offset += int(dur)

        result = xr.DataArray(
            out[:n_original_timesteps],
            dims=["time", *other_dims],
            coords={"time": time_coords},
        )
        for d in other_dims:
            if d in data.coords:
                result = result.assign_coords({d: data.coords[d]})
        return result
