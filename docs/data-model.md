# Data Model

## Object relationships

```mermaid
%%{init: {'theme': 'neutral', 'themeVariables': {'fontSize': '15px'}, 'flowchart': {'padding': 16, 'nodeSpacing': 30, 'rankSpacing': 50, 'htmlLabels': true}}}%%
graph TD
    A["<b>aggregate(da)</b>"] --> R["<b>AggregationResult</b>"]
    R --> CR["<b>.clustering</b>"]
    R --> ACC["<b>.accuracy</b>"]
    CR -->|"to_json / from_json"| JSON["📄 clustering.json"]
    JSON -->|"load_clustering()"| CR2["<b>ClusteringResult</b>"]
    CR2 -->|"apply(new_da)"| R2["<b>AggregationResult</b>"]
    CR2 -->|"disaggregate(data)"| D["DataArray<br/><i>full time axis</i>"]
    R -->|"disaggregate(data)"| D
```

## AggregationResult

Returned by `aggregate()`. Contains everything about one aggregation.

<div style="transform: scale(0.9); transform-origin: top left;">

```mermaid
%%{init: {'theme': 'neutral', 'themeVariables': {'fontSize': '13px'}, 'flowchart': {'padding': 12, 'nodeSpacing': 8, 'rankSpacing': 40, 'htmlLabels': true}}}%%
graph LR
    R["<b>AggregationResult</b>"]

    R --- D["<b>Data</b>"]
    R --- Meta["<b>Metadata</b>"]

    D --- F1[".cluster_representatives<br/><i>cluster, timestep, *cluster_dims, *slice_dims</i>"]
    D --- F2[".reconstructed<br/><i>same shape as input</i>"]
    D --- F3[".cluster_assignments<br/><i>period, *slice_dims</i>"]
    D --- F4[".cluster_weights<br/><i>cluster, *slice_dims</i>"]
    D --- F5[".segment_durations<br/><i>cluster, timestep, *slice_dims | None</i>"]

    Meta --- A[".accuracy<br/><b>→ AccuracyMetrics</b>"]
    Meta --- C[".clustering<br/><b>→ ClusteringResult</b>"]
```

</div>

## ClusteringResult

The reusable part — knows *how* the time series was clustered,
without the original data. Access via `result.clustering` or
`load_clustering("clustering.json")`.

All DataArray properties are **cached** on first access.

<div style="transform: scale(0.85); transform-origin: top left;">

```mermaid
%%{init: {'theme': 'neutral', 'themeVariables': {'fontSize': '13px'}, 'flowchart': {'padding': 12, 'nodeSpacing': 8, 'rankSpacing': 40, 'htmlLabels': true}}}%%
graph LR
    CR["<b>ClusteringResult</b>"]

    CR --- S["<b>Scalars</b>"]
    CR --- DA["<b>DataArray properties</b>"]
    CR --- M["<b>Methods</b>"]

    S --- S1[".n_clusters"]
    S --- S2[".n_original_periods"]
    S --- S3[".n_timesteps_per_period"]
    S --- S4[".n_segments"]

    DA --- DA1[".cluster_assignments<br/><i>period, *slice_dims</i>"]
    DA --- DA2[".cluster_occurrences<br/><i>cluster, *slice_dims</i>"]
    DA --- DA3[".cluster_centers<br/><i>cluster, *slice_dims</i>"]
    DA --- DA4[".segment_durations<br/><i>cluster, timestep, *slice_dims | None</i>"]
    DA --- DA5[".segment_assignments<br/><i>cluster, timestep, *slice_dims | None</i>"]
    DA --- DA6[".segment_centers<br/><i>cluster, segment, *slice_dims | None</i>"]

    M --- M1[".apply(da)"]
    M --- M2[".disaggregate(data)"]
    M --- M3[".to_json(path)"]
    M --- M4[".from_json(path)"]
```

</div>

## AccuracyMetrics

Per-column metrics as DataArrays, plus weighted scalars.

| Field | Type | Description |
|-------|------|-------------|
| `rmse` | DataArray | Per-column RMSE |
| `mae` | DataArray | Per-column MAE |
| `rmse_duration` | DataArray | Per-column duration-curve RMSE |
| `weighted_rmse` | float | Scalar RMSE weighted by column weights |
| `weighted_mae` | float | Scalar MAE weighted by column weights |
| `weighted_rmse_duration` | float | Scalar duration RMSE weighted by column weights |

## Glossary

| Term | Meaning |
|------|---------|
| **cluster_dim** | Dimensions clustered together (stacked internally) |
| **slice_dims** | Dimensions aggregated independently |
| **period** | One repeating unit of time (e.g., one day) |
| **cluster** | A group of similar periods |
| **timestep** | Position within a period (e.g., hour 0-23) |
| **segment** | A contiguous block of timesteps (with segmentation) |
