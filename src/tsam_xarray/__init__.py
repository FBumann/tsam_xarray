"""tsam_xarray: Lightweight xarray wrapper for tsam time series aggregation."""

from tsam_xarray._core import aggregate
from tsam_xarray._result import AccuracyResult, AggregationResult

__all__ = ["AccuracyResult", "AggregationResult", "aggregate"]
