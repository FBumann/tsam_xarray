"""Scaling benchmarks for tsam_xarray.aggregate() with production configs.

Run with:
    uv run --with pytest-benchmem pytest benchmarks/ -o addopts="" -p no:xdist \
        --benchmark-only --benchmark-memory

Focus configs:
- hierarchical clustering (tsam default)
- Distribution(scope="global") representation
- include_period_sums=False (tsam default)
- ExtremeConfig(method="replace")

``config_variants`` isolates the cost of each option at a fixed size;
the scale tests grow columns/slices/days under the full production config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tsam
import xarray as xr

from tsam_xarray import aggregate


def make_data(n_days: int, n_cols: int, n_slices: int) -> xr.DataArray:
    rng = np.random.default_rng(42)
    n_t = n_days * 24
    time_idx = pd.date_range("2023-01-01", periods=n_t, freq="h")
    t = np.arange(n_t)
    daily = np.sin(2 * np.pi * t / 24)
    yearly = np.sin(2 * np.pi * t / 8760)
    base = (
        daily[None, :] * rng.uniform(0.5, 2, (n_cols, 1))
        + yearly[None, :] * rng.uniform(0.5, 2, (n_cols, 1))
        + rng.normal(0, 0.3, (n_cols, n_t))
    )
    dims = ["variable", "time"]
    coords: dict[str, object] = {
        "variable": [f"var{i}" for i in range(n_cols)],
        "time": time_idx,
    }
    data = base
    if n_slices > 1:
        scale = rng.uniform(0.8, 1.2, (n_slices, 1, 1))
        data = base[None, :, :] * scale
        dims = ["scenario", *dims]
        coords["scenario"] = [f"s{i}" for i in range(n_slices)]
    return xr.DataArray(data, dims=dims, coords=coords, name="load")


def cluster_config(representation: object | None = None) -> tsam.ClusterConfig:
    return tsam.ClusterConfig(
        method="hierarchical",
        representation=representation,
        include_period_sums=False,
    )


def extremes_config() -> tsam.ExtremeConfig:
    return tsam.ExtremeConfig(method="replace", max_value=["var0"], min_value=["var1"])


CONFIGS: dict[str, dict[str, object]] = {
    "default": {},
    "dist_global": {"cluster": cluster_config(tsam.Distribution(scope="global"))},
    "extremes_replace": {"cluster": cluster_config(), "extremes": extremes_config()},
    "full": {
        "cluster": cluster_config(tsam.Distribution(scope="global")),
        "extremes": extremes_config(),
    },
}


def run_aggregate(da: xr.DataArray, n_clusters: int, config: dict[str, object]) -> None:
    aggregate(
        da,
        time_dim="time",
        cluster_dim="variable",
        n_clusters=n_clusters,
        **config,
    )


BENCH_OPTS = dict(rounds=3, iterations=1, warmup_rounds=0)

REPRESENTATIONS: dict[str, object | None] = {
    "medoid": None,
    "mean": "mean",
    "dist_cluster": tsam.Distribution(scope="cluster"),
    "dist_global": tsam.Distribution(scope="global"),
    "dist_global_minmax": tsam.Distribution(scope="global", preserve_minmax=True),
}

EXTREMES: dict[str, tsam.ExtremeConfig | None] = {
    "none": None,
    "replace": extremes_config(),
    "append": tsam.ExtremeConfig(
        method="append", max_value=["var0"], min_value=["var1"]
    ),
}


@pytest.mark.parametrize("config", list(CONFIGS))
def test_config_variants(benchmark, config):
    da = make_data(365, n_cols=8, n_slices=1)
    benchmark.pedantic(run_aggregate, args=(da, 12, CONFIGS[config]), **BENCH_OPTS)


@pytest.mark.parametrize("extremes", list(EXTREMES))
@pytest.mark.parametrize("representation", list(REPRESENTATIONS))
def test_config_grid(benchmark, representation, extremes):
    da = make_data(365, n_cols=8, n_slices=1)
    config: dict[str, object] = {
        "cluster": cluster_config(REPRESENTATIONS[representation])
    }
    if EXTREMES[extremes] is not None:
        config["extremes"] = EXTREMES[extremes]
    benchmark.pedantic(run_aggregate, args=(da, 12, config), **BENCH_OPTS)


@pytest.mark.parametrize("n_cols", [16, 64, 128])
@pytest.mark.parametrize("representation", ["medoid", "dist_global"])
def test_representation_x_columns(benchmark, representation, n_cols):
    da = make_data(365, n_cols=n_cols, n_slices=1)
    config = {"cluster": cluster_config(REPRESENTATIONS[representation])}
    benchmark.pedantic(run_aggregate, args=(da, 12, config), **BENCH_OPTS)


@pytest.mark.parametrize("preserve", ["on", "off"])
@pytest.mark.parametrize("representation", ["medoid", "dist_global"])
def test_preserve_column_means(benchmark, representation, preserve):
    da = make_data(365, n_cols=8, n_slices=1)
    config: dict[str, object] = {
        "cluster": cluster_config(REPRESENTATIONS[representation]),
        "preserve_column_means": preserve == "on",
    }
    benchmark.pedantic(run_aggregate, args=(da, 12, config), **BENCH_OPTS)


@pytest.mark.parametrize("n_cols", [16, 64, 128])
def test_full_scale_columns(benchmark, n_cols):
    da = make_data(365, n_cols=n_cols, n_slices=1)
    benchmark.pedantic(run_aggregate, args=(da, 12, CONFIGS["full"]), **BENCH_OPTS)


@pytest.mark.parametrize("n_slices", [4, 8])
def test_full_scale_slices(benchmark, n_slices):
    da = make_data(365, n_cols=8, n_slices=n_slices)
    benchmark.pedantic(run_aggregate, args=(da, 12, CONFIGS["full"]), **BENCH_OPTS)


@pytest.mark.parametrize("n_days", [365, 730])
def test_full_scale_days(benchmark, n_days):
    da = make_data(n_days, n_cols=8, n_slices=1)
    benchmark.pedantic(run_aggregate, args=(da, 12, CONFIGS["full"]), **BENCH_OPTS)
