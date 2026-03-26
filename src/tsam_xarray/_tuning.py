"""Tuning functions for finding optimal n_clusters/n_segments."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tsam import SegmentConfig
from tsam.tuning import find_clusters_for_reduction

from tsam_xarray._core import aggregate
from tsam_xarray._result import AggregationResult

Weights = dict[str, float] | dict[str, dict[str, float]] | None

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Result of hyperparameter tuning.

    Attributes
    ----------
    n_clusters : int
        Optimal number of typical periods.
    n_segments : int
        Optimal number of segments per period.
    rmse : float
        RMSE of the optimal configuration.
    best_result : AggregationResult
        The AggregationResult for the optimal configuration.
    history : list[dict]
        History of all tested configurations with their RMSE values.
    all_results : list[AggregationResult]
        All AggregationResults from tuning (when ``save_all_results=True``).
    """

    n_clusters: int
    n_segments: int
    rmse: float
    best_result: AggregationResult
    history: list[dict[str, Any]] = field(repr=False)
    all_results: list[AggregationResult] = field(default_factory=list, repr=False)

    @property
    def summary(self) -> Any:
        """Summary table of all tested configurations, sorted by RMSE."""
        import pandas as pd

        return pd.DataFrame(self.history).sort_values("rmse")

    @property
    def summary_matrix(self) -> Any:
        """Metrics as Dataset with ``(n_clusters, n_segments)`` dims.

        Contains ``rmse`` and ``timesteps`` as variables.
        NaN where a combination was not tested.
        """
        import pandas as pd

        df = pd.DataFrame(self.history)
        return df.set_index(["n_clusters", "n_segments"]).to_xarray()

    def find_by_timesteps(self, target: int) -> AggregationResult:
        """Find the result closest to a target timestep count.

        Requires ``save_all_results=True`` when calling the tuning function.
        """
        if not self.all_results:
            msg = (
                "No results available. Use save_all_results=True in "
                "the tuning function."
            )
            raise ValueError(msg)
        if len(self.all_results) != len(self.history):
            msg = (
                f"Results/history mismatch: {len(self.all_results)} results "
                f"vs {len(self.history)} history entries."
            )
            raise ValueError(msg)

        best_idx = 0
        best_diff = float("inf")
        for i, h in enumerate(self.history):
            diff = abs(h["timesteps"] - target)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return self.all_results[best_idx]

    def find_by_rmse(self, threshold: float) -> AggregationResult:
        """Find the smallest configuration that achieves a target RMSE.

        Returns the configuration with the fewest timesteps whose RMSE
        is at or below ``threshold``.

        Requires ``save_all_results=True`` when calling the tuning function.
        """
        if not self.all_results:
            msg = (
                "No results available. Use save_all_results=True in "
                "the tuning function."
            )
            raise ValueError(msg)
        if len(self.all_results) != len(self.history):
            msg = (
                f"Results/history mismatch: {len(self.all_results)} results "
                f"vs {len(self.history)} history entries."
            )
            raise ValueError(msg)

        candidates: list[tuple[int, int]] = []  # (timesteps, index)
        for i, h in enumerate(self.history):
            if h["rmse"] <= threshold:
                candidates.append((h["timesteps"], i))

        if not candidates:
            best_available = min(h["rmse"] for h in self.history)
            msg = (
                f"No configuration achieves RMSE <= {threshold}. "
                f"Best available: {best_available:.4f}"
            )
            raise ValueError(msg)

        candidates.sort(key=lambda x: x[0])
        return self.all_results[candidates[0][1]]

    def plot(self, show_labels: bool = True, **kwargs: Any) -> Any:
        """Plot RMSE vs timesteps."""
        import plotly.graph_objects as go

        summary = self.summary
        hover_text = [
            f"{row['n_clusters']}x{row['n_segments']}<br>"
            f"Timesteps: {row['timesteps']}<br>"
            f"RMSE: {row['rmse']:.4f}"
            for _, row in summary.iterrows()
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=summary["timesteps"],
                y=summary["rmse"],
                mode="lines+markers" if len(summary) > 1 else "markers",
                marker={"size": 10},
                hovertext=hover_text if show_labels else None,
                hoverinfo="text" if show_labels else "x+y",
                **kwargs,
            )
        )
        fig.update_layout(
            title="Tuning Results: Complexity vs Accuracy",
            xaxis_title="Timesteps (n_clusters x n_segments)",
            yaxis_title="RMSE",
            hovermode="closest",
        )
        return fig

    def __len__(self) -> int:
        return len(self.all_results)

    def __getitem__(self, index: int) -> AggregationResult:
        return self.all_results[index]

    def __iter__(self) -> Any:
        return iter(self.all_results)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _infer_time_params(
    da: Any,
    time_dim: str,
    period_duration: int | float | str,
) -> tuple[int, int, int]:
    """Return (n_timesteps_per_period, n_periods, n_timesteps)."""
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
    return n_timesteps_per_period, n_periods, n_timesteps


def _wrap_progress(
    items: list[tuple[int, int]],
    show_progress: bool,
    desc: str,
) -> Any:
    """Optionally wrap *items* in a tqdm progress bar."""
    if show_progress:
        try:
            import tqdm

            return tqdm.tqdm(items, desc=desc)
        except ImportError:
            pass
    return items


def _evaluate_candidates(
    candidates: list[tuple[int, int]],
    da: Any,
    *,
    time_dim: str,
    cluster_dim: Sequence[str] | str,
    weights: Weights,
    period_duration: int | float | str,
    show_progress: bool,
    progress_desc: str,
    save_all_results: bool,
    tsam_kwargs: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[AggregationResult],
    float,
    AggregationResult | None,
    int,
    int,
]:
    """Evaluate candidates, returning history and best config."""
    history: list[dict[str, Any]] = []
    all_results: list[AggregationResult] = []
    best_rmse = float("inf")
    best_result: AggregationResult | None = None
    best_n_clusters = 0
    best_n_segments = 0

    iterator = _wrap_progress(candidates, show_progress, progress_desc)

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
        except Exception as exc:
            logger.debug(
                "Config (n_clusters=%d, n_segments=%d) failed: %s",
                n_clust,
                n_seg,
                exc,
            )

    return (
        history,
        all_results,
        best_rmse,
        best_result,
        best_n_clusters,
        best_n_segments,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    save_all_results : bool
        Keep all AggregationResults (memory-intensive).
    **tsam_kwargs
        Additional keyword arguments passed to ``tsam.aggregate()``.

    Returns
    -------
    TuningResult
        Best combination with lowest overall RMSE.
    """
    n_timesteps_per_period, _n_periods, n_timesteps = _infer_time_params(
        da, time_dim, period_duration
    )

    # Generate candidates: for each segment count, max clusters that fits
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

    history, all_results, best_rmse, best_result, best_nc, best_ns = (
        _evaluate_candidates(
            candidates,
            da,
            time_dim=time_dim,
            cluster_dim=cluster_dim,
            weights=weights,
            period_duration=period_duration,
            show_progress=show_progress,
            progress_desc="Testing configurations",
            save_all_results=save_all_results,
            tsam_kwargs=tsam_kwargs,
        )
    )

    if best_result is None:
        msg = "All configurations failed"
        raise RuntimeError(msg)

    return TuningResult(
        n_clusters=best_nc,
        n_segments=best_ns,
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

    The returned ``history`` contains only Pareto-optimal points.
    Use ``save_all_results=True`` to iterate over all corresponding
    ``AggregationResult`` objects.

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
        Defaults to total number of timesteps in the data.
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
    n_timesteps_per_period, n_periods, n_timesteps = _infer_time_params(
        da, time_dim, period_duration
    )

    if max_timesteps is None:
        max_timesteps = n_timesteps

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

    # Evaluate all candidates
    history, all_results, _best_rmse, _best_result, _best_nc, _best_ns = (
        _evaluate_candidates(
            candidates,
            da,
            time_dim=time_dim,
            cluster_dim=cluster_dim,
            weights=weights,
            period_duration=period_duration,
            show_progress=show_progress,
            progress_desc="Pareto front",
            save_all_results=save_all_results,
            tsam_kwargs=tsam_kwargs,
        )
    )

    if not history:
        msg = "All configurations failed"
        raise RuntimeError(msg)

    # Filter to Pareto front: no other point has both fewer timesteps AND lower RMSE
    pareto_history, pareto_results = _pareto_filter(history, all_results)

    # Best on Pareto front = lowest RMSE
    best_idx = min(range(len(pareto_history)), key=lambda i: pareto_history[i]["rmse"])
    best_h = pareto_history[best_idx]

    # We need the actual AggregationResult for the best point.
    # If save_all_results is True, it's in pareto_results.
    # Otherwise, re-run the best config.
    if pareto_results:
        best_result = pareto_results[best_idx]
    else:
        # Re-run best config to get the AggregationResult
        seg_config = SegmentConfig(n_segments=best_h["n_segments"])
        best_result = aggregate(
            da,
            time_dim=time_dim,
            cluster_dim=cluster_dim,
            n_clusters=best_h["n_clusters"],
            weights=weights,
            segments=seg_config,
            period_duration=period_duration,
            **tsam_kwargs,
        )

    return TuningResult(
        n_clusters=best_h["n_clusters"],
        n_segments=best_h["n_segments"],
        rmse=best_h["rmse"],
        best_result=best_result,
        history=pareto_history,
        all_results=pareto_results,
    )


def _pareto_filter(
    history: list[dict[str, Any]],
    all_results: list[AggregationResult],
) -> tuple[list[dict[str, Any]], list[AggregationResult]]:
    """Filter history to Pareto-optimal points (fewer timesteps, lower RMSE).

    A point is Pareto-optimal if no other point has both strictly fewer
    timesteps and strictly lower (or equal) RMSE.
    """
    has_results = len(all_results) == len(history)

    # Sort by timesteps ascending
    indexed = sorted(enumerate(history), key=lambda x: x[1]["timesteps"])

    pareto_history: list[dict[str, Any]] = []
    pareto_results: list[AggregationResult] = []
    best_rmse = float("inf")

    for orig_idx, entry in indexed:
        if entry["rmse"] < best_rmse:
            pareto_history.append(entry)
            if has_results:
                pareto_results.append(all_results[orig_idx])
            best_rmse = entry["rmse"]

    return pareto_history, pareto_results


# ---------------------------------------------------------------------------
# RMSE computation
# ---------------------------------------------------------------------------


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
