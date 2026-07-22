"""tsam_xarray: Lightweight xarray wrapper for tsam time series aggregation."""

from tsam_xarray._clustering import ClusteringInfo, ClusteringResult
from tsam_xarray._core import aggregate
from tsam_xarray._dim_names import DimNames
from tsam_xarray._result import AccuracyMetrics, AggregationResult
from tsam_xarray._tuning import (
    TuningResult,
    find_best_combination,
    find_optimal_combination,
    find_pareto_front,
    grid_search,
)

load_clustering = ClusteringResult.from_json

__all__ = [
    "AccuracyMetrics",
    "AggregationResult",
    "ClusteringInfo",
    "ClusteringResult",
    "DimNames",
    "TuningResult",
    "aggregate",
    "find_best_combination",
    "find_optimal_combination",
    "find_pareto_front",
    "grid_search",
    "load_clustering",
]
