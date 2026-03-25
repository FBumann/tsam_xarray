"""Result dataclasses for tsam_xarray."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr


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
    raw: Any  # tsam.AggregationResult or dict of them

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
        if isinstance(self.raw, dict):
            first = next(iter(self.raw.values()))
            result: int | None = first.n_segments
        else:
            result = self.raw.n_segments
        return result

    @property
    def clustering_duration(self) -> float:
        """Time spent on clustering in seconds."""
        if isinstance(self.raw, dict):
            total: float = sum(r.clustering_duration for r in self.raw.values())
            return total
        duration: float = self.raw.clustering_duration
        return duration

    @property
    def is_transferred(self) -> bool:
        """Whether result was created via ClusteringResult.apply()."""
        if isinstance(self.raw, dict):
            return all(r.is_transferred for r in self.raw.values())
        is_transferred: bool = self.raw.is_transferred
        return is_transferred

    @property
    def residuals(self) -> xr.DataArray:
        """Difference between original and reconstructed data."""
        return self.original - self.reconstructed

    def disaggregate(self, data: xr.DataArray) -> xr.DataArray:
        """Map data on (cluster, timestep) back to original time.

        Expands the compact representation to the full time axis
        using cluster assignments. With segmentation, values are
        placed at segment boundaries and remaining timesteps are
        NaN — use ``.ffill()``, ``.interpolate_na()``, etc. to fill.

        Parameters
        ----------
        data : xr.DataArray
            Data with ``cluster`` and ``timestep`` dims. All other
            dims are passed through.

        Returns
        -------
        xr.DataArray
            Data with ``cluster`` and ``timestep`` replaced by the
            original time dimension.
        """
        time_coords = self.original.coords["time"]
        assignments = self.cluster_assignments.values
        n_original_timesteps = len(time_coords)
        n_periods = len(assignments)
        n_per_period = n_original_timesteps // n_periods

        other_dims = [d for d in data.dims if d not in ("cluster", "timestep")]

        if self.segment_durations is None:
            # No segmentation — simple repeat via assignments
            expanded = data.sel(cluster=xr.DataArray(assignments, dims=["period"]))
            # Reshape (period, timestep, ...) → (time, ...)
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

        # With segmentation — place at segment boundaries, NaN elsewhere
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
            out[: len(time_coords)],
            dims=["time", *other_dims],
            coords={"time": time_coords},
        )
        for d in other_dims:
            if d in data.coords:
                result = result.assign_coords({d: data.coords[d]})
        return result
