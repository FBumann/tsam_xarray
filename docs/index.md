# tsam_xarray

Lightweight [xarray](https://xarray.dev/) wrapper for **[tsam](https://github.com/FZJ-IEK3-VSA/tsam)** time series aggregation.

tsam_xarray lets you aggregate multi-dimensional xarray DataArrays using [tsam](https://tsam.readthedocs.io/)'s clustering algorithms. It handles:

- **DataFrame conversion** — stack/unstack dimensions automatically
- **Independent slicing** — aggregate per scenario, year, region, etc. in one call
- **Result assembly** — typical periods, accuracy metrics, cluster weights, and segment durations are concatenated into coherent multi-dimensional xarray objects

## Quick example

```python
import tsam_xarray

result = tsam_xarray.aggregate(
    da,
    time_dim="time",
    cluster_dim=["variable", "region"],
    n_clusters=8,
)

result.typical_periods   # (cluster, timestep, variable, region)
result.cluster_weights   # (cluster,)
result.accuracy.rmse     # (variable, region)
result.reconstructed     # same shape as input
```

All [tsam.aggregate()](https://tsam.readthedocs.io/) keyword arguments pass through — clustering methods, segmentation, extreme periods, etc.

## Installation

```bash
pip install tsam_xarray
```

## Next steps

- [Getting Started](examples/getting-started.ipynb) — basic workflow
- [Multi-Dimensional Data](examples/multi-dim.ipynb) — stacking, slicing, weights
- [API Reference](api/) — full function and class documentation
