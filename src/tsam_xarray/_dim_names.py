"""Configurable output dimension names for tsam_xarray."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DimNames:
    """Names of the structural output dimensions produced by aggregation.

    tsam_xarray adds four dimensions to its results that do not exist in the
    input: the cluster/representative axis, the intra-period timestep axis,
    the original-period axis (in ``cluster_assignments``), and the segment
    axis (segmented runs). By default these are ``cluster``, ``timestep``,
    ``period``, and ``segment``; override them when they would collide with
    the caller's own dimension names.

    Attributes:
        cluster: Cluster/representative axis.
        timestep: Intra-period timestep axis.
        period: Original-period axis (in ``cluster_assignments``).
        segment: Segment axis (segmented runs).
    """

    cluster: str = "cluster"
    timestep: str = "timestep"
    period: str = "period"
    segment: str = "segment"

    def __post_init__(self) -> None:
        names = self.as_tuple()
        if len(set(names)) != len(names):
            msg = f"DimNames must be unique, got {names}"
            raise ValueError(msg)

    def as_tuple(self) -> tuple[str, str, str, str]:
        """The four names as a tuple, in declaration order."""
        return (self.cluster, self.timestep, self.period, self.segment)
