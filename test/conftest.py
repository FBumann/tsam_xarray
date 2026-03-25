"""Shared fixtures for tsam_xarray tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@dataclass
class AggregateCase:
    """A test case for aggregate()."""

    id: str
    da: xr.DataArray
    time_dim: str
    cluster_dim: list[str] | str
    n_clusters: int
    expected_cluster_dims: set[str]
    expected_slice_dims: set[str]
    n_periods: int


def _make_time(n_days: int = 30) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n_days * 24, freq="h")


def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _build_cases() -> list[AggregateCase]:
    time = _make_time(30)
    rng = _rng()
    cases = []

    # 1D: time only
    cases.append(
        AggregateCase(
            id="1d",
            da=xr.DataArray(
                rng.random(len(time)),
                dims=["time"],
                coords={"time": time},
            ),
            time_dim="time",
            cluster_dim=(),
            n_clusters=4,
            expected_cluster_dims=set(),
            expected_slice_dims=set(),
            n_periods=30,
        )
    )

    # 2D: time + variable
    cases.append(
        AggregateCase(
            id="2d_single_cluster",
            da=xr.DataArray(
                rng.random((len(time), 2)),
                dims=["time", "variable"],
                coords={"time": time, "variable": ["solar", "wind"]},
            ),
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
            expected_cluster_dims={"variable"},
            expected_slice_dims=set(),
            n_periods=30,
        )
    )

    # 3D: time + variable + region (multi cluster dim)
    cases.append(
        AggregateCase(
            id="3d_multi_cluster",
            da=xr.DataArray(
                rng.random((len(time), 2, 2)),
                dims=["time", "variable", "region"],
                coords={
                    "time": time,
                    "variable": ["solar", "wind"],
                    "region": ["north", "south"],
                },
            ),
            time_dim="time",
            cluster_dim=["variable", "region"],
            n_clusters=4,
            expected_cluster_dims={"variable", "region"},
            expected_slice_dims=set(),
            n_periods=30,
        )
    )

    # 3D: time + variable + scenario (single cluster, single slice)
    cases.append(
        AggregateCase(
            id="3d_single_slice",
            da=xr.DataArray(
                rng.random((len(time), 2, 2)),
                dims=["time", "variable", "scenario"],
                coords={
                    "time": time,
                    "variable": ["solar", "wind"],
                    "scenario": ["low", "high"],
                },
            ),
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
            expected_cluster_dims={"variable"},
            expected_slice_dims={"scenario"},
            n_periods=30,
        )
    )

    # 4D: time + variable + region + scenario (multi cluster, single slice)
    cases.append(
        AggregateCase(
            id="4d_multi_cluster_single_slice",
            da=xr.DataArray(
                rng.random((len(time), 2, 2, 2)),
                dims=["time", "variable", "region", "scenario"],
                coords={
                    "time": time,
                    "variable": ["solar", "wind"],
                    "region": ["north", "south"],
                    "scenario": ["low", "high"],
                },
            ),
            time_dim="time",
            cluster_dim=["variable", "region"],
            n_clusters=4,
            expected_cluster_dims={"variable", "region"},
            expected_slice_dims={"scenario"},
            n_periods=30,
        )
    )

    # 4D: time + variable + region + scenario (single cluster, multi slice)
    cases.append(
        AggregateCase(
            id="4d_single_cluster_multi_slice",
            da=xr.DataArray(
                rng.random((len(time), 2, 2, 2)),
                dims=["time", "variable", "region", "scenario"],
                coords={
                    "time": time,
                    "variable": ["solar", "wind"],
                    "region": ["north", "south"],
                    "scenario": ["low", "high"],
                },
            ),
            time_dim="time",
            cluster_dim="variable",
            n_clusters=4,
            expected_cluster_dims={"variable"},
            expected_slice_dims={"region", "scenario"},
            n_periods=30,
        )
    )

    # 5D: + year
    da_5d = xr.DataArray(
        rng.random((len(time), 2, 2, 2, 2)),
        dims=["time", "variable", "region", "scenario", "year"],
        coords={
            "time": time,
            "variable": ["solar", "wind"],
            "region": ["north", "south"],
            "scenario": ["low", "high"],
            "year": [2020, 2021],
        },
    )
    cases.append(
        AggregateCase(
            id="5d_multi_cluster_multi_slice",
            da=da_5d,
            time_dim="time",
            cluster_dim=["variable", "region"],
            n_clusters=4,
            expected_cluster_dims={"variable", "region"},
            expected_slice_dims={"scenario", "year"},
            n_periods=30,
        )
    )

    return cases


CASES = _build_cases()
CASE_IDS = [c.id for c in CASES]


@pytest.fixture(params=CASES, ids=CASE_IDS)
def agg_case(request: pytest.FixtureRequest) -> AggregateCase:
    """Parametrized aggregate test case."""
    return request.param
