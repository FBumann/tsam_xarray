# Data Model

This page describes the objects returned by `tsam_xarray.aggregate()` and how they relate to each other.

## AggregationResult

Returned by [`aggregate()`][tsam_xarray.aggregate]. Contains everything about
the aggregation: the compact representatives, how to map them back, and how
accurate the approximation is.

```
AggregationResult
|-- cluster_representatives   DataArray (cluster, timestep, *cluster_dims, *slice_dims)
|-- cluster_assignments       DataArray (period, *slice_dims)
|-- cluster_weights           DataArray (cluster, *slice_dims)
|-- segment_durations         DataArray (cluster, timestep, *slice_dims) | None
|-- accuracy                  AccuracyMetrics
|   |-- rmse                  DataArray (*cluster_dims, *slice_dims)
|   |-- mae                   DataArray (*cluster_dims, *slice_dims)
|   |-- rmse_duration         DataArray (*cluster_dims, *slice_dims)
|   |-- weighted_rmse         float
|   |-- weighted_mae          float
|   |-- weighted_rmse_duration float
|-- reconstructed             DataArray (same shape as input)
|-- original                  DataArray (the input data)
|-- clustering                ClusteringResult
|-- n_clusters                int (property)
|-- n_timesteps_per_period    int (property)
|-- n_segments                int | None (property)
|-- residuals                 DataArray (property: original - reconstructed)
```

### Methods

| Method | Description |
|--------|-------------|
| `disaggregate(data)` | Expand cluster-representative data back to the original time axis |

## ClusteringResult

Available via `result.clustering`, or loaded from JSON with
[`load_clustering()`][tsam_xarray.load_clustering]. This is the reusable
part of an aggregation: it knows *how* the time series was clustered,
without the original data.

```
ClusteringResult
|-- time_dim                  str
|-- cluster_dim               list[str]
|-- slice_dims                list[str]
|-- time_coords               DatetimeIndex | None
|-- n_clusters                int (property)
|-- n_original_periods        int (property)
|-- n_timesteps_per_period    int (property)
|-- n_segments                int | None (property)
|-- cluster_assignments       DataArray (period, *slice_dims)          [cached]
|-- cluster_occurrences       DataArray (cluster, *slice_dims)         [cached]
|-- cluster_centers           DataArray (cluster, *slice_dims)         [cached]
|-- segment_durations         DataArray (cluster, timestep, *slice_dims) | None [cached]
|-- segment_assignments       DataArray (cluster, timestep, *slice_dims) | None [cached]
|-- segment_centers           DataArray (cluster, segment, *slice_dims) | None  [cached]
```

### Methods

| Method | Description |
|--------|-------------|
| `apply(da)` | Apply this clustering to new data, returning a new `AggregationResult` |
| `disaggregate(data)` | Expand cluster-representative data back to the original time axis |
| `to_json(path)` | Save clustering to JSON |
| `from_json(path)` | Load clustering from JSON (class method) |

### Glossary

| Term | Meaning |
|------|---------|
| **cluster_dim** | Dimensions clustered together (stacked internally, e.g. `["variable", "region"]`) |
| **slice_dims** | Dimensions aggregated independently (e.g. `["scenario"]`) |
| **period** | One repeating unit of time (e.g., one day with hourly data) |
| **cluster** | A group of similar periods |
| **timestep** | Position within a period (e.g., hour 0-23) |
| **segment** | A contiguous block of timesteps within a period (when using segmentation) |

## Typical workflows

### Inspect after aggregation

```python
result = tsam_xarray.aggregate(da, time_dim="time", cluster_dim="variable", n_clusters=8)

# Quick overview
result                        # AggregationResult(n_clusters=8, ...)
result.accuracy               # AccuracyMetrics(weighted_rmse=0.05, ...)

# Clustering metadata
result.clustering.cluster_assignments    # which cluster each day belongs to
result.clustering.cluster_occurrences    # how many days per cluster
result.clustering.cluster_centers        # which day is the representative
```

### Save, load, and reuse

```python
# Save
result.clustering.to_json("clustering.json")

# Load and inspect (no original data needed)
clustering = tsam_xarray.load_clustering("clustering.json")
clustering.n_clusters                    # 8
clustering.cluster_assignments           # DataArray
clustering.disaggregate(optimized_data)  # expand back to full time

# Apply to new data
new_result = clustering.apply(new_da)
```
