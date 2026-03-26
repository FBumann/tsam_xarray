"""Tuning functions for finding optimal n_clusters/n_segments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tsam import SegmentConfig
from tsam.tuning import find_clusters_for_reduction

from tsam_xarray._core import aggregate
from tsam_xarray._result import AggregationResult

Weights = dict[str, float] | dict[str, dict[str, float]] | None


@dataclass
class TuningResult:
    """Result of find_optimal_combination()."""

    n_clusters: int
    n_segments: int
    rmse: float
    best_result: AggregationResult
    history: list[dict[str, Any]] = field(repr=False)
    all_results: list[AggregationResult] = field(default_factory=list, repr=False)

    @property
    def summary(self) -> Any:
        """Summary table of all tested configurations."""
        import pandas as pd

        return pd.DataFrame(self.history).sort_values("rmse")

    @property
    def summary_matrix(self) -> Any:
        """Metrics as Dataset with ``(n_clusters, n_segments)`` dims.

        Contains ``rmse`` and ``timesteps`` as variables.
        NaN where a combination was not tested or failed.
        """
        import pandas as pd

        df = pd.DataFrame(self.history)
        df = df.replace(float("inf"), float("nan"))
        return df.set_index(["n_clusters", "n_segments"]).to_xarray()


def find_optimal_combination(
    da: Any,
    *,
    time_dim: str,
    cluster_dim: Sequence[str] | str,
    data_reduction: float,
    weights: Weights = None,
    period_duration: int | float | str = 24,
    show_progress: bool = True,
    save_all_results: bool = False,
    **tsam_kwargs: Any,
) -> TuningResult:
    """Find optimal n_clusters/n_segments for a target data reduction.

    Tests all (n_clusters, n_segments) combinations that achieve
    the target data reduction, evaluating each across all slices.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    time_dim : str
        Name of the time dimension.
    cluster_dim : Sequence[str] | str
        Dimension(s) to cluster together.
    data_reduction : float
        Target data reduction (e.g., 0.01 for 1% of original).
    weights : dict | None
        Per-coordinate weights for clustering and RMSE evaluation.
    period_duration : int | float | str
        Hours per period (default: 24 for daily).
    show_progress : bool
        Show progress bar (requires tqdm).
    **tsam_kwargs
        Additional keyword arguments passed to ``tsam.aggregate()``.

    Returns
    -------
    TuningResult
        Best combination with lowest overall RMSE.
    """

    # Infer n_timesteps_per_period from period_duration
    time_coords = da.coords[time_dim].values
    if len(time_coords) < 2:
        msg = "Need at least 2 timesteps to infer resolution"
        raise ValueError(msg)
    dt_hours = float(np.diff(time_coords[:2])[0] / np.timedelta64(1, "h"))
    if isinstance(period_duration, str):
        import pandas as pd

        period_hours = pd.Timedelta(period_duration).total_seconds() / 3600
    else:
        period_hours = float(period_duration)
    n_timesteps_per_period = int(period_hours / dt_hours)
    n_periods = len(time_coords) // n_timesteps_per_period
    n_timesteps = n_periods * n_timesteps_per_period

    # Generate candidates
    candidates: list[tuple[int, int]] = []
    for n_seg in range(1, n_timesteps_per_period + 1):
        n_clust = find_clusters_for_reduction(n_timesteps, n_seg, data_reduction)
        if n_clust >= 2:
            candidates.append((n_clust, n_seg))

    if not candidates:
        msg = (
            f"No valid (n_clusters, n_segments) combinations "
            f"for data_reduction={data_reduction}"
        )
        raise ValueError(msg)

    # Evaluate each candidate
    history: list[dict[str, Any]] = []
    all_results: list[AggregationResult] = []
    best_rmse = float("inf")
    best_result: AggregationResult | None = None
    best_n_clusters = 0
    best_n_segments = 0

    iterator: Any = candidates
    if show_progress:
        try:
            import tqdm

            iterator = tqdm.tqdm(candidates, desc="Testing configurations")
        except ImportError:
            pass

    for n_clust, n_seg in iterator:
        try:
            seg_config = SegmentConfig(n_segments=n_seg)
            result = aggregate(
                da,
                time_dim=time_dim,
                cluster_dim=cluster_dim,
                n_clusters=n_clust,
                weights=weights,
                segments=seg_config,
                period_duration=period_duration,
                **tsam_kwargs,
            )
            rmse = _compute_overall_rmse(result, weights, cluster_dim)
            history.append(
                {
                    "n_clusters": n_clust,
                    "n_segments": n_seg,
                    "rmse": rmse,
                    "timesteps": n_clust * n_seg,
                }
            )
            if save_all_results:
                all_results.append(result)
            if rmse < best_rmse:
                best_rmse = rmse
                best_result = result
                best_n_clusters = n_clust
                best_n_segments = n_seg
        except Exception:
            history.append(
                {
                    "n_clusters": n_clust,
                    "n_segments": n_seg,
                    "rmse": float("inf"),
                    "timesteps": n_clust * n_seg,
                }
            )

    if best_result is None:
        msg = "All configurations failed"
        raise RuntimeError(msg)

    return TuningResult(
        n_clusters=best_n_clusters,
        n_segments=best_n_segments,
        rmse=best_rmse,
        best_result=best_result,
        history=history,
        all_results=all_results,
    )


def find_pareto_front(
    da: Any,
    *,
    time_dim: str,
    cluster_dim: Sequence[str] | str,
    max_timesteps: int | None = None,
    weights: Weights = None,
    period_duration: int | float | str = 24,
    show_progress: bool = True,
    save_all_results: bool = False,
    **tsam_kwargs: Any,
) -> TuningResult:
    """Find the Pareto-optimal configurations (RMSE vs complexity).

    Tests a grid of (n_clusters, n_segments) combinations and returns
    the Pareto frontier — configurations where no other tested combo
    has both lower RMSE and fewer timesteps.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    time_dim : str
        Name of the time dimension.
    cluster_dim : Sequence[str] | str
        Dimension(s) to cluster together.
    max_timesteps : int | None
        Maximum total timesteps to test (n_clusters * n_segments).
        Defaults to n_timesteps_per_period (full resolution).
    weights : dict | None
        Per-coordinate weights for clustering and RMSE evaluation.
    period_duration : int | float | str
        Hours per period (default: 24).
    show_progress : bool
        Show progress bar.
    save_all_results : bool
        Keep all AggregationResults (memory-intensive).
    **tsam_kwargs
        Additional keyword arguments passed to ``tsam.aggregate()``.

    Returns
    -------
    TuningResult
        Pareto-optimal result with lowest RMSE on the frontier.
    """
    time_coords = da.coords[time_dim].values
    if len(time_coords) < 2:
        msg = "Need at least 2 timesteps"
        raise ValueError(msg)
    dt_hours = float(np.diff(time_coords[:2])[0] / np.timedelta64(1, "h"))
    if isinstance(period_duration, str):
        import pandas as pd

        period_hours = pd.Timedelta(period_duration).total_seconds() / 3600
    else:
        period_hours = float(period_duration)
    n_timesteps_per_period = int(period_hours / dt_hours)
    n_periods = len(time_coords) // n_timesteps_per_period

    if max_timesteps is None:
        max_timesteps = n_timesteps_per_period

    # Generate grid of candidates
    # Cap n_clusters at n_periods - 1 (n_periods = trivial perfect fit)
    max_clusters = n_periods - 1
    candidates: list[tuple[int, int]] = []
    for n_seg in range(1, n_timesteps_per_period + 1):
        for n_clust in range(2, min(max_clusters, max_timesteps // n_seg) + 1):
            if n_clust * n_seg <= max_timesteps:
                candidates.append((n_clust, n_seg))

    if not candidates:
        msg = f"No valid combinations for max_timesteps={max_timesteps}"
        raise ValueError(msg)

    # Evaluate (reuse find_optimal_combination's loop logic)
    history: list[dict[str, Any]] = []
    all_results_list: list[AggregationResult] = []
    best_rmse = float("inf")
    best_result: AggregationResult | None = None
    best_n_clusters = 0
    best_n_segments = 0

    iterator: Any = candidates
    if show_progress:
        try:
            import tqdm

            iterator = tqdm.tqdm(candidates, desc="Pareto front")
        except ImportError:
            pass

    for n_clust, n_seg in iterator:
        try:
            seg_config = SegmentConfig(n_segments=n_seg)
            result = aggregate(
                da,
                time_dim=time_dim,
                cluster_dim=cluster_dim,
                n_clusters=n_clust,
                weights=weights,
                segments=seg_config,
                period_duration=period_duration,
                **tsam_kwargs,
            )
            rmse = _compute_overall_rmse(result, weights, cluster_dim)
            history.append(
                {
                    "n_clusters": n_clust,
                    "n_segments": n_seg,
                    "rmse": rmse,
                    "timesteps": n_clust * n_seg,
                }
            )
            if save_all_results:
                all_results_list.append(result)
            if rmse < best_rmse:
                best_rmse = rmse
                best_result = result
                best_n_clusters = n_clust
                best_n_segments = n_seg
        except Exception:
            history.append(
                {
                    "n_clusters": n_clust,
                    "n_segments": n_seg,
                    "rmse": float("inf"),
                    "timesteps": n_clust * n_seg,
                }
            )

    if best_result is None:
        msg = "All configurations failed"
        raise RuntimeError(msg)

    return TuningResult(
        n_clusters=best_n_clusters,
        n_segments=best_n_segments,
        rmse=best_rmse,
        best_result=best_result,
        history=history,
        all_results=all_results_list,
    )


def _compute_overall_rmse(
    result: AggregationResult,
    weights: Weights,
    cluster_dim: Sequence[str] | str,
) -> float:
    """Compute weighted overall RMSE across all columns and slices.

    Weights affect the aggregation across cluster_dim columns.
    Slice dims are averaged equally.
    """
    from tsam_xarray._core import (
        _normalize_weights,
        _resolve_cluster_dim,
    )

    rmse = result.accuracy.rmse
    col_dims = _resolve_cluster_dim(cluster_dim)

    # Get per-dim weights if provided
    per_dim = _normalize_weights(weights, result.original, col_dims)

    if per_dim is None:
        # Unweighted: quadratic mean across all values
        return float(np.sqrt((rmse**2).mean()))

    # Build weight array matching rmse dims (cluster_dims only)
    w_values = np.ones_like(rmse.values, dtype=float)
    for dim_name, coord_weights in per_dim.items():
        if dim_name in rmse.dims:
            dim_idx = list(rmse.dims).index(dim_name)
            for coord_val, w in coord_weights.items():
                coord_vals = [str(c) for c in rmse.coords[dim_name].values]
                if str(coord_val) in coord_vals:
                    idx = coord_vals.index(str(coord_val))
                    indexer: list[Any] = [slice(None)] * len(rmse.dims)
                    indexer[dim_idx] = idx
                    w_values[tuple(indexer)] *= w

    # Weighted quadratic mean
    weighted_sq = w_values * rmse.values**2
    return float(np.sqrt(weighted_sq.sum() / w_values.sum()))
