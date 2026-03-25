"""tsam_xarray: Lightweight xarray wrapper for tsam time series aggregation."""

from tsam_xarray._clustering import ClusteringInfo
from tsam_xarray._core import aggregate
from tsam_xarray._result import AccuracyMetrics, AggregationResult

load_clustering = ClusteringInfo.from_json

__all__ = [
    "AccuracyMetrics",
    "AggregationResult",
    "ClusteringInfo",
    "aggregate",
    "load_clustering",
]
