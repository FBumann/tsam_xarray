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
        # Identify slice dims (dims on data that aren't cluster/timestep
        # and aren't cluster_dim coords)
        slice_dims = [
            str(d)
            for d in data.dims
            if d not in ("cluster", "timestep") and d in self.cluster_assignments.dims
        ]

        if not slice_dims:
            return self._disaggregate_single(data)

        # Loop over slice dims and concat
        import itertools

        slice_coords = {d: data.coords[d].values for d in slice_dims}
        keys = list(itertools.product(*(slice_coords[d] for d in slice_dims)))
        results = []
        for key in keys:
            sel = dict(zip(slice_dims, key, strict=True))
            data_slice = data.sel(sel)
            # Use per-slice raw result for assignments/durations
            result_slice = self._make_slice_view(sel)
            results.append(result_slice._disaggregate_single(data_slice))

        # Concat along slice dims
        out = results[0]
        if len(slice_dims) == 1:
            import pandas as pd

            out = xr.concat(
                results,
                dim=pd.Index(
                    slice_coords[slice_dims[0]],
                    name=slice_dims[0],
                ),
            )
        else:
            import pandas as pd

            # Multi-dim concat
            it = iter(results)

            def _nest(dims: list[str]) -> list:  # type: ignore[type-arg]
                if len(dims) == 1:
                    return [next(it) for _ in slice_coords[dims[0]]]
                return [_nest(dims[1:]) for _ in slice_coords[dims[0]]]

            nested = _nest(slice_dims)
            for dim in reversed(slice_dims):
                idx = pd.Index(slice_coords[dim], name=dim)
                if isinstance(nested[0], list):
                    nested = [xr.concat(group, dim=idx) for group in nested]
                else:
                    out = xr.concat(nested, dim=idx)
        return out

    def _make_slice_view(self, sel: dict[str, object]) -> AggregationResult:
        """Create a view of this result for a single slice."""
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
            raw=(
                self.raw[tuple(sel.values())]
                if isinstance(self.raw, dict)
                else self.raw
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
