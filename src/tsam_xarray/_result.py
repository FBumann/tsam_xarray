"""Result dataclasses for tsam_xarray."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from tsam_xarray._clustering import ClusteringResult
    from tsam_xarray._dim_names import DimNames


@dataclass(frozen=True, repr=False)
class AccuracyMetrics:
    """Accuracy metrics from time series aggregation.

    Attributes:
        rmse: Per-column RMSE.
            Dims: ``(*cluster_dims, *slice_dims)``.
        mae: Per-column MAE.
            Dims: ``(*cluster_dims, *slice_dims)``.
        rmse_duration: Per-column duration-curve RMSE.
            Dims: ``(*cluster_dims, *slice_dims)``.
        weighted_rmse: RMSE weighted across columns.
            Dims: ``(*slice_dims)`` or scalar.
        weighted_mae: MAE weighted across columns.
            Dims: ``(*slice_dims)`` or scalar.
        weighted_rmse_duration: Duration-curve RMSE weighted
            across columns.
            Dims: ``(*slice_dims)`` or scalar.
    """

    rmse: xr.DataArray
    mae: xr.DataArray
    rmse_duration: xr.DataArray
    weighted_rmse: xr.DataArray
    weighted_mae: xr.DataArray
    weighted_rmse_duration: xr.DataArray

    def __repr__(self) -> str:
        def _fmt(da: xr.DataArray) -> str:
            mean = float(da.mean())
            if da.size <= 1:
                return f"{mean:.4f}"
            return f"{mean:.4f} [{float(da.min()):.4f}-{float(da.max()):.4f}]"

        return (
            f"AccuracyMetrics("
            f"weighted_rmse={_fmt(self.weighted_rmse)}, "
            f"weighted_mae={_fmt(self.weighted_mae)}, "
            f"weighted_rmse_duration="
            f"{_fmt(self.weighted_rmse_duration)})"
        )


@dataclass(frozen=True, repr=False)
class AggregationResult:
    """Result of ``tsam_xarray.aggregate()``.

    Attributes:
        cluster_representatives: Typical periods.
            Dims: ``(cluster, timestep, *cluster_dims,
            *slice_dims)``.
        cluster_assignments: Which cluster each period
            belongs to. Dims: ``(period, *slice_dims)``.
        cluster_counts: Periods per cluster.
            Dims: ``(cluster, *slice_dims)``. Formerly
            ``cluster_weights``, which remains as a deprecated
            alias (following tsam v4's rename).
        segment_durations: Duration of each segment, or
            ``None``. Dims: ``(cluster, timestep,
            *slice_dims)``.
        accuracy: Per-column and weighted accuracy metrics.
        reconstructed: Reconstructed time series
            (same shape and dim order as ``original``).
        original: The input data.
        clustering: Reusable clustering metadata.
            See `ClusteringResult`.
        is_transferred: Whether this result came from
            ``apply()`` vs ``aggregate()``.
    """

    cluster_representatives: xr.DataArray
    cluster_assignments: xr.DataArray
    cluster_counts: xr.DataArray
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
            f"weighted_rmse={float(self.accuracy.weighted_rmse.mean()):.4f})"
        )

    @property
    def dim_names(self) -> DimNames:
        """Names of the structural output dimensions. See `DimNames`."""
        return self.clustering.dim_names

    @property
    def cluster_weights(self) -> xr.DataArray:
        """Deprecated alias for `cluster_counts`.

        Renamed to match tsam v4, where the values are occurrence counts
        rather than weights. Will be removed in a future release.
        """
        warnings.warn(
            "AggregationResult.cluster_weights is deprecated; use "
            "cluster_counts instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.cluster_counts

    @property
    def n_clusters(self) -> int:
        """Number of cluster representative clusters."""
        return int(self.cluster_counts.sizes[self.dim_names.cluster])

    @property
    def n_timesteps_per_period(self) -> int:
        """Number of timesteps per cluster representative."""
        return int(self.cluster_representatives.sizes[self.dim_names.timestep])

    @property
    def n_segments(self) -> int | None:
        """Number of segments per period, if segmentation was used."""
        first_cr = next(iter(self.clustering.clusterings.values()))
        result: int | None = first_cr.n_segments
        return result

    @property
    def residuals(self) -> xr.DataArray:
        """Difference between original and reconstructed data.

        Shares the dim order of ``original`` and ``reconstructed``.
        """
        return self.original - self.reconstructed

    def compare(self, **sel: object) -> xr.DataArray:
        """Stack ``original`` and ``reconstructed`` along a ``variant`` dim.

        Returns a single DataArray on the original time axis with a new
        ``variant`` coordinate ``["original", "reconstructed"]``, ready to
        plot directly with a ``color=``/``hue="variant"`` grouping — no
        ``melt`` step. This is the canonical way to eyeball aggregation
        quality per column and per slice dim.

        Args:
            **sel: Optional label-based selection applied to both arrays
                before stacking, e.g. ``compare(variable="solar")`` to
                compare a single column.

        Returns:
            DataArray with dims ``("variant", *original.dims)``.

        Examples:
            >>> agg.compare(variable="solar").plotly.line(
            ...     x="time", color="variant"
            ... )
        """
        original = self.original
        reconstructed = self.reconstructed
        if sel:
            original = original.sel(sel)
            reconstructed = reconstructed.sel(sel)
        variant = pd.Index(["original", "reconstructed"], name="variant")
        combined = xr.concat([original, reconstructed], dim=variant)
        combined.name = self.original.name
        return combined

    def to_dataframe(self, **sel: object) -> pd.DataFrame:
        """Tidy/long-form ``original`` vs ``reconstructed`` DataFrame.

        A flat DataFrame with a ``variant`` column
        (``"original"``/``"reconstructed"``), the ``time`` axis, every
        cluster and slice dim, and a value column — ready to hand
        straight to a plotting library.

        Args:
            **sel: Optional label-based selection forwarded to
                `compare` (e.g. ``variable="solar"``).

        Returns:
            Long-form DataFrame with a ``variant`` column and one value
            column (named after the input DataArray, or ``"value"``).
        """
        combined = self.compare(**sel)
        name = combined.name
        if name is None or str(name) == "variant":
            name = "value"
        return combined.to_dataframe(name=str(name)).reset_index()

    def disaggregate(self, data: xr.DataArray) -> xr.DataArray:
        """Map data on ``(cluster, timestep)`` back to original time.

        This is the inverse of ``aggregate()``. Use it to expand
        external data computed on the compact cluster-representative
        grid (e.g., optimization results) back to the full time
        axis.

        Without segmentation, values are repeated for each timestep
        in the period. With segmentation, values are placed at
        segment boundaries and remaining timesteps are NaN — use
        ``.ffill(dim="time")``,
        ``.interpolate_na(dim="time")``, etc.

        Args:
            data: Data with ``cluster`` and ``timestep`` dims,
                matching the shape of
                ``result.cluster_representatives``. Additional
                dims (including auto-sliced dims like scenario)
                are supported.

        Returns:
            Data with ``cluster`` and ``timestep`` replaced by
            the original ``time`` dimension.
        """
        return self.clustering.disaggregate(data)
