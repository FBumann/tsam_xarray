# tsam_xarray

[![PyPI](https://img.shields.io/pypi/v/tsam-xarray)](https://pypi.org/project/tsam-xarray/)
[![Python](https://img.shields.io/pypi/pyversions/tsam-xarray)](https://pypi.org/project/tsam-xarray/)
[![CI](https://github.com/FBumann/tsam_xarray/actions/workflows/ci.yaml/badge.svg)](https://github.com/FBumann/tsam_xarray/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue)](https://tsam_xarray.readthedocs.io/)

Lightweight [xarray](https://xarray.dev/) wrapper for [tsam](https://github.com/FZJ-IEK3-VSA/tsam) time series aggregation.

**DataArray in, DataArray out** — no manual DataFrame conversions, no MultiIndex wrangling, no loop-and-concat boilerplate.

## Installation

```bash
pip install tsam-xarray
```

## Quick start

```python
import tsam_xarray
from tsam_xarray import sample_energy_data

da = sample_energy_data(n_days=30)  # (time, variable, region, scenario)

# Aggregate to 4 typical days
result = tsam_xarray.aggregate(
    da, time_dim="time", cluster_dim="variable", n_clusters=4,
)

result.typical_periods   # (cluster, timestep, variable)
result.cluster_weights   # (cluster,) — days each cluster represents
result.accuracy.rmse     # (variable,) — per-variable RMSE
result.reconstructed     # same shape as input
```

## Multi-dimensional data

```python
# Cluster variable x region together; scenario is sliced independently
result = tsam_xarray.aggregate(
    da,
    time_dim="time",
    cluster_dim=["variable", "region"],
    n_clusters=8,
)

result.typical_periods  # (scenario, cluster, timestep, variable, region)
```

## Weights

```python
# Single cluster_dim — simple dict
result = tsam_xarray.aggregate(
    da, time_dim="time", cluster_dim="variable", n_clusters=8,
    weights={"solar": 2.0, "wind": 1.0},
)

# Multiple cluster_dim — dict-of-dicts
result = tsam_xarray.aggregate(
    da, time_dim="time", cluster_dim=["variable", "region"], n_clusters=8,
    weights={"variable": {"solar": 2.0}, "region": {"north": 1.5}},
)
```

## tsam passthrough

All [tsam.aggregate()](https://github.com/FZJ-IEK3-VSA/tsam) keyword arguments pass through:

```python
from tsam import ClusterConfig, SegmentConfig

result = tsam_xarray.aggregate(
    da,
    time_dim="time",
    cluster_dim="variable",
    n_clusters=8,
    cluster=ClusterConfig(method="kmeans"),
    segments=SegmentConfig(n_segments=6),
)
```

## Documentation

Full docs with interactive examples: [tsam_xarray.readthedocs.io](https://tsam_xarray.readthedocs.io/)
