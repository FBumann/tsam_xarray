"""Result dataclasses for tsam_xarray."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from tsam_xarray._clustering import ClusteringResult


@dataclass(frozen=True, repr=False)
class AccuracyMetrics:
    """Accuracy metrics from time series aggregation."""

    rmse: xr.DataArray
    mae: xr.DataArray
    rmse_duration: xr.DataArray
    weighted_rmse: float = 0.0
    weighted_mae: float = 0.0
    weighted_rmse_duration: float = 0.0

    def __repr__(self) -> str:
        return (
            f"AccuracyMetrics("
            f"weighted_rmse={self.weighted_rmse:.4f}, "
            f"weighted_mae={self.weighted_mae:.4f}, "
            f"weighted_rmse_duration={self.weighted_rmse_duration:.4f})"
        )


@dataclass(frozen=True, repr=False)
class AggregationResult:
    """Result of tsam_xarray.aggregate()."""

    cluster_representatives: xr.DataArray
    cluster_assignments: xr.DataArray
    cluster_weights: xr.DataArray
    segment_durations: xr.DataArray | None
    accuracy: AccuracyMetrics
    reconstructed: xr.DataArray
    original: xr.DataArray
    clustering: ClusteringResult
    is_transferred: bool = False

    def __repr__(self) -> str:
        c = self.clustering
        slices = f", slice_dims={c.slice_dims}" if c.slice_dims else ""
        seg = f", n_segments={self.n_segments}" if self.n_segments else ""
        return (
            f"AggregationResult("
            f"n_clusters={self.n_clusters}, "
            f"n_periods={c.n_original_periods}, "
            f"cluster_dim={c.cluster_dim}"
            f"{slices}{seg}, "
            f"weighted_rmse={self.accuracy.weighted_rmse:.4f})"
        )

    @property
    def n_clusters(self) -> int:
        """Number of cluster representative clusters."""
        return int(self.cluster_weights.sizes["cluster"])

    @property
    def n_timesteps_per_period(self) -> int:
        """Number of timesteps per cluster representative."""
        return int(self.cluster_representatives.sizes["timestep"])

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
        external data computed on the compact cluster-representative grid
        (e.g., optimization results) back to the full time axis.

        Without segmentation, values are repeated for each timestep
        in the period. With segmentation, values are placed at segment
        boundaries and remaining timesteps are NaN — use
        ``.ffill(dim="time")``, ``.interpolate_na(dim="time")``, etc.

        Parameters
        ----------
        data : xr.DataArray
            Data with ``cluster`` and ``timestep`` dims, matching the
            shape of ``result.cluster_representatives``. Additional dims
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
            ClusteringResult as CR,
        )
        from tsam_xarray._clustering import (
            _lookup_clustering,
        )

        # Build key in stored slice_dims order
        key = tuple(sel[d] for d in self.clustering.slice_dims)
        cr = _lookup_clustering(self.clustering.clusterings, key)

        return AggregationResult(
            cluster_representatives=self.cluster_representatives.sel(sel),
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
                weighted_rmse=self.accuracy.weighted_rmse,
                weighted_mae=self.accuracy.weighted_mae,
                weighted_rmse_duration=self.accuracy.weighted_rmse_duration,
            ),
            reconstructed=self.reconstructed.sel(sel),
            original=self.original.sel(sel),
            clustering=CR(
                time_dim=self.clustering.time_dim,
                cluster_dim=self.clustering.cluster_dim,
                slice_dims=[],
                clusterings={(): cr},
            ),
        )

    def _disaggregate_single(self, data: xr.DataArray) -> xr.DataArray:
        """Disaggregate without slice dims."""
        import pandas as pd

        from tsam_xarray._clustering import _disaggregate_single

        time_coords = pd.DatetimeIndex(self.original.coords["time"].values)
        cr = self.clustering.clusterings[()]
        return _disaggregate_single(time_coords, cr, data)
