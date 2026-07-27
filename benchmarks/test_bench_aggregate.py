"""Benchmarks for tsam_xarray: wrapper micro-benchmarks + end-to-end sentinels.

Run with:
    uv run --with pytest-benchmem pytest benchmarks/ -o addopts= -p no:xdist \
        --benchmark-only --benchmark-memory

Three layers, cheapest statistics where the repo's own code lives:

- ``test_wrapper_*`` — micro-benchmarks of the wrapper's conversion stages
  (DataFrame in, result out, slice concat). Profiling showed ~90% of an
  end-to-end run is inside tsam, which this repo cannot regress — its PRs
  can only regress these stages, so they get many rounds and wide inputs.
- ``test_config_*`` — cost ratios of tsam config options at a reduced size
  (90 days); every measured config effect is multiplicative, so ratios at
  90 days match 365 days at a quarter of the cost.
- ``test_e2e_*`` — few full-pipeline sentinels at production size for
  integration surprises (tsam version bumps, config plumbing), the
  representation-by-width interaction, and the multi-``cluster_dim``
  MultiIndex path.
- ``test_large_*`` — production-sized cases (hundreds of MiB peak) that
  guard the memory-amplification regime. Opt-in via ``--large`` (skipped
  otherwise, including in CI) — run locally before dependency bumps or
  performance work.
- ``test_user_*`` — post-aggregation operations users run on results:
  reusing a stored clustering on new data (``clustering.apply``) and
  expanding cluster-level data back to the full time axis
  (``disaggregate``). Accuracy metrics and ``reconstructed`` need no
  extra cases — tsam computes them eagerly inside ``aggregate()``, so
  every e2e case and the result-conversion micro already include them.

Cases are a deterministic grid (data seeded per call) because benchmem
matches runs by test ID — IDs and data must be stable across runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tsam
import xarray as xr

from tsam_xarray import aggregate
from tsam_xarray._core import (
    _aggregate_single,
    _concat_results,
    _result_from_tsam,
    _to_dataframe,
)
from tsam_xarray._dim_names import DimNames


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
        "variable": [f"var{i:03d}" for i in range(n_cols)],
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


def extremes_config(method: str = "replace") -> tsam.ExtremeConfig:
    return tsam.ExtremeConfig(method=method, max_value=["var000"], min_value=["var001"])


FULL_CONFIG: dict[str, object] = {
    "cluster": cluster_config(tsam.Distribution(scope="global")),
    "extremes": extremes_config(),
}

REPRESENTATIONS: dict[str, object | None] = {
    "medoid": None,
    "mean": "mean",
    "dist_cluster": tsam.Distribution(scope="cluster"),
    "dist_global": tsam.Distribution(scope="global"),
    "dist_global_minmax": tsam.Distribution(scope="global", preserve_minmax=True),
}


def run_aggregate(
    da: xr.DataArray,
    config: dict[str, object],
    cluster_dim: list[str] | str = "variable",
) -> None:
    aggregate(da, time_dim="time", cluster_dim=cluster_dim, n_clusters=12, **config)


MICRO_OPTS = dict(rounds=10, iterations=10, warmup_rounds=1)
CONFIG_OPTS = dict(rounds=5, iterations=1, warmup_rounds=1)
E2E_OPTS = dict(rounds=3, iterations=1, warmup_rounds=1)


# --- wrapper micro-benchmarks -------------------------------------------------


@pytest.fixture(scope="module")
def wide_tsam_result():
    da = make_data(365, 128, 1)
    df = _to_dataframe(da, "time", ["variable"])
    return tsam.aggregate(df, 12), da, df


@pytest.fixture(scope="module")
def slice_results():
    da = make_data(365, 8, 8)
    scenarios = da.coords["scenario"].values
    results = [
        _aggregate_single(
            da.sel(scenario=s), 12, "time", ["variable"], None, None, {}, DimNames()
        )
        for s in scenarios
    ]
    return results, {"scenario": scenarios}, [(s,) for s in scenarios]


def test_wrapper_to_dataframe(benchmark):
    da = make_data(365, 128, 1)
    benchmark.pedantic(_to_dataframe, args=(da, "time", ["variable"]), **MICRO_OPTS)


def test_wrapper_result_conversion(benchmark, wide_tsam_result):
    res, da, df = wide_tsam_result
    benchmark.pedantic(
        _result_from_tsam,
        args=(res, da, df, "time", ["variable"], DimNames()),
        **MICRO_OPTS,
    )


def test_wrapper_concat_results(benchmark, slice_results):
    results, slice_coords, slice_keys = slice_results
    benchmark.pedantic(
        _concat_results,
        args=(results, ["scenario"], slice_coords, slice_keys),
        **MICRO_OPTS,
    )


# --- config cost ratios (reduced size) ----------------------------------------


@pytest.mark.parametrize("representation", list(REPRESENTATIONS))
def test_config_representation(benchmark, representation):
    da = make_data(90, n_cols=8, n_slices=1)
    config = {"cluster": cluster_config(REPRESENTATIONS[representation])}
    benchmark.pedantic(run_aggregate, args=(da, config), **CONFIG_OPTS)


@pytest.mark.parametrize("extremes", ["replace", "append"])
def test_config_extremes(benchmark, extremes):
    da = make_data(90, n_cols=8, n_slices=1)
    config = {"cluster": cluster_config(), "extremes": extremes_config(extremes)}
    benchmark.pedantic(run_aggregate, args=(da, config), **CONFIG_OPTS)


# --- end-to-end sentinels -----------------------------------------------------


def test_e2e_default(benchmark):
    da = make_data(365, n_cols=8, n_slices=1)
    benchmark.pedantic(run_aggregate, args=(da, {}), **E2E_OPTS)


def test_e2e_full(benchmark):
    da = make_data(365, n_cols=8, n_slices=1)
    benchmark.pedantic(run_aggregate, args=(da, FULL_CONFIG), **E2E_OPTS)


@pytest.mark.parametrize("representation", ["medoid", "dist_global"])
def test_e2e_wide(benchmark, representation):
    da = make_data(365, n_cols=128, n_slices=1)
    config = {"cluster": cluster_config(REPRESENTATIONS[representation])}
    benchmark.pedantic(run_aggregate, args=(da, config), **E2E_OPTS)


def test_e2e_slices(benchmark):
    da = make_data(365, n_cols=8, n_slices=8)
    benchmark.pedantic(run_aggregate, args=(da, FULL_CONFIG), **E2E_OPTS)


def test_e2e_multidim(benchmark):
    da = make_data(365, n_cols=8, n_slices=4).rename(scenario="region")
    benchmark.pedantic(run_aggregate, args=(da, {}, ["variable", "region"]), **E2E_OPTS)


# --- user post-aggregation operations -----------------------------------------


@pytest.fixture(scope="module")
def wide_result():
    da = make_data(365, 128, 1)
    result = aggregate(da, time_dim="time", cluster_dim="variable", n_clusters=12)
    return result, da


def test_user_apply(benchmark, wide_result):
    result, da = wide_result
    benchmark.pedantic(result.clustering.apply, args=(da * 1.1,), **CONFIG_OPTS)


def test_user_disaggregate(benchmark, wide_result):
    result, _da = wide_result
    benchmark.pedantic(
        result.clustering.disaggregate,
        args=(result.cluster_representatives,),
        **CONFIG_OPTS,
    )


@pytest.fixture(scope="module")
def sliced_result():
    da = make_data(365, 32, 12)
    return aggregate(da, time_dim="time", cluster_dim="variable", n_clusters=12)


def test_user_disaggregate_sliced(benchmark, sliced_result):
    benchmark.pedantic(
        sliced_result.clustering.disaggregate,
        args=(sliced_result.cluster_representatives,),
        **CONFIG_OPTS,
    )


def test_user_disaggregate_segmented(benchmark):
    da = make_data(365, 32, 1)
    result = aggregate(
        da,
        time_dim="time",
        cluster_dim="variable",
        n_clusters=12,
        segments=tsam.SegmentConfig(n_segments=6),
    )
    benchmark.pedantic(
        result.clustering.disaggregate,
        args=(result.cluster_representatives,),
        **CONFIG_OPTS,
    )


# --- large tier (production-sized, opt-in via --large) -------------------------

LARGE_OPTS = dict(rounds=2, iterations=1, warmup_rounds=0)


@pytest.mark.large
def test_large_wide(benchmark):
    da = make_data(730, n_cols=256, n_slices=1)
    benchmark.pedantic(run_aggregate, args=(da, FULL_CONFIG), **LARGE_OPTS)


@pytest.mark.large
def test_large_scenarios(benchmark):
    da = make_data(365, n_cols=64, n_slices=8)
    benchmark.pedantic(run_aggregate, args=(da, FULL_CONFIG), **LARGE_OPTS)
