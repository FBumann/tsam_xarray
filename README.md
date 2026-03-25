# tsam_xarray

Lightweight [xarray](https://xarray.dev/) wrapper for [tsam](https://github.com/FZJ-IEK3-VSA/tsam) time series aggregation.

## Installation

```bash
pip install tsam_xarray
```

## Quick start

```python
import numpy as np
import pandas as pd
import xarray as xr
import tsam_xarray

# Create sample data: 30 days of hourly solar and wind data
time = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
da = xr.DataArray(
    np.random.default_rng(42).random((len(time), 2)),
    dims=["time", "variable"],
    coords={"time": time, "variable": ["solar", "wind"]},
)

# Aggregate to 4 typical days
result = tsam_xarray.aggregate(
    da, time_dim="time", cluster_dim="variable", n_clusters=4,
)

result.typical_periods   # (cluster, timestep, variable)
result.cluster_weights   # (cluster,) — days each represents
result.accuracy.rmse     # (variable,) — per-variable RMSE
result.reconstructed     # same shape as input
```

## Multi-dimensional data

```python
# 4D data: (time, variable, region, scenario)
da = xr.DataArray(...)

# Cluster variable × region together; scenario is sliced independently
result = tsam_xarray.aggregate(
    da,
    time_dim="time",
    cluster_dim=["variable", "region"],
    n_clusters=8,
)

result.typical_periods  # (scenario, cluster, timestep, variable, region)
```

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
