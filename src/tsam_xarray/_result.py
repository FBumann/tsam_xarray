"""Result dataclasses for tsam_xarray."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import xarray as xr


@dataclass(frozen=True)
class AccuracyResult:
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
    accuracy: AccuracyResult
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
            return first.n_segments
        return self.raw.n_segments  # type: ignore[no-any-return]

    @property
    def clustering_duration(self) -> float:
        """Time spent on clustering in seconds."""
        if isinstance(self.raw, dict):
            return sum(
                r.clustering_duration for r in self.raw.values()
            )
        return self.raw.clustering_duration  # type: ignore[no-any-return]

    @property
    def is_transferred(self) -> bool:
        """Whether result was created via ClusteringResult.apply()."""
        if isinstance(self.raw, dict):
            return all(
                r.is_transferred for r in self.raw.values()
            )
        return self.raw.is_transferred  # type: ignore[no-any-return]

    @property
    def residuals(self) -> xr.DataArray:
        """Difference between original and reconstructed data."""
        return self.original - self.reconstructed
