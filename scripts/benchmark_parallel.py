"""Benchmark sequential vs parallel aggregation."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xarray as xr

import tsam_xarray


def make_data(n_hours: int, n_variables: int, n_slices: int) -> xr.DataArray:
    t = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    rng = np.random.default_rng(42)
    return xr.DataArray(
        rng.random((n_hours, n_variables, n_slices)),
        dims=["time", "variable", "scenario"],
        coords={
            "time": t,
            "variable": [f"v{i}" for i in range(n_variables)],
            "scenario": [f"s{i}" for i in range(n_slices)],
        },
    )


def benchmark(da: xr.DataArray, n_clusters: int = 8) -> None:
    n_slices = da.sizes["scenario"]
    n_hours = da.sizes["time"]
    n_vars = da.sizes["variable"]
    print(f"  {n_hours}h x {n_vars} vars x {n_slices} slices")

    start = time.perf_counter()
    tsam_xarray.aggregate(
        da,
        time_dim="time",
        cluster_dim="variable",
        n_clusters=n_clusters,
        n_jobs=1,
    )
    t_seq = time.perf_counter() - start
    print(f"  Sequential: {t_seq:.2f}s ({t_seq / n_slices:.3f}s/slice)")

    start = time.perf_counter()
    tsam_xarray.aggregate(
        da,
        time_dim="time",
        cluster_dim="variable",
        n_clusters=n_clusters,
        n_jobs=-1,
    )
    t_par = time.perf_counter() - start
    speedup = t_seq / t_par
    print(f"  Parallel:   {t_par:.2f}s (speedup: {speedup:.1f}x)")
    print()


if __name__ == "__main__":
    print("=== Small data (30 days) ===")
    benchmark(make_data(720, 3, 4))

    print("=== Medium data (1 year, 4 slices) ===")
    benchmark(make_data(8760, 5, 4))

    print("=== Medium data (1 year, 10 slices) ===")
    benchmark(make_data(8760, 5, 10))

    print("=== Large data (1 year, 20 slices) ===")
    benchmark(make_data(8760, 10, 20))

    print("=== Large data (1 year, 50 slices) ===")
    benchmark(make_data(8760, 10, 50))
