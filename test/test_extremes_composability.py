"""Composability + fuzz tests for extreme-period handling across slices.

Two guarantees are exercised here, both riding on the tsam fix in
FZJ-IEK3-VSA/tsam#410:

* **Count stability** — with additive extremes (``append`` / ``new_cluster``),
  every independently-aggregated slice must yield exactly ``n_clusters`` so the
  results stack into a rectangular array. The wrapper forces
  ``preserve_n_clusters=True`` to achieve this. Where the running tsam lacks the
  flag, the tests instead assert the documented fallback contract: aggregate()
  either succeeds with *uniform* counts or raises the specific count-mismatch
  error — never anything else.
* **Transferable replace** — ``method="replace"`` composes with ``cluster_on``
  and round-trips through ``apply()``. Gated on the tsam that stores the
  injection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tsam
import xarray as xr

import tsam_xarray
from tsam_xarray._core import (
    _clustering_supports_replace_transfer,
    _extreme_config_supports_preserve,
)

SUPPORTS_PRESERVE = _extreme_config_supports_preserve()
SUPPORTS_REPLACE_TRANSFER = _clustering_supports_replace_transfer()

VARIABLES = ["solar", "wind", "price"]


def _sliced_da(seed: int, n_scenarios: int = 3, n_days: int = 20) -> xr.DataArray:
    """A (time, variable, scenario) array — scenario is an independent slice."""
    rng = np.random.default_rng(seed)
    time = pd.date_range("2020-01-01", periods=n_days * 24, freq="h")
    return xr.DataArray(
        rng.random((len(time), len(VARIABLES), n_scenarios)),
        dims=["time", "variable", "scenario"],
        coords={
            "time": time,
            "variable": VARIABLES,
            "scenario": [f"s{i}" for i in range(n_scenarios)],
        },
    )


@pytest.mark.parametrize("seed", range(30))
def test_additive_extremes_across_slices_fuzz(seed: int) -> None:
    """Fuzz additive extremes over sliced data; assert the count contract.

    Criteria are capped so the distinct-extreme count D stays below
    ``n_clusters`` (tsam rejects ``n_clusters <= D`` under preserve).
    """
    rng = np.random.default_rng(10_000 + seed)
    method = str(rng.choice(["append", "new_cluster"]))

    def pick() -> list[str]:
        k = int(rng.integers(0, 3))  # 0-2 columns per criterion
        if k == 0:
            return []
        return list(rng.choice(VARIABLES, size=k, replace=False))

    max_value, min_value = pick(), pick()
    if not (max_value or min_value):  # ensure at least one active criterion
        max_value = ["solar"]

    extremes = tsam.ExtremeConfig(
        method=method, max_value=max_value, min_value=min_value
    )
    n_clusters = int(rng.integers(8, 12))
    da = _sliced_da(seed)

    try:
        res = tsam_xarray.aggregate(
            da,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=n_clusters,
            extremes=extremes,
        )
    except ValueError as exc:
        # The only tolerated failure is the documented count mismatch, and only
        # on a tsam that cannot pin the count.
        assert not SUPPORTS_PRESERVE, f"unexpected failure under preserve: {exc}"
        assert "different cluster counts" in str(exc)
        return

    # Success path: counts are uniform across slices (concat succeeded).
    if SUPPORTS_PRESERVE:
        assert res.clustering.n_clusters == n_clusters
        assert res.cluster_representatives.sizes["cluster"] == n_clusters
    else:
        # Fallback tsam: slices happened to agree; the shared count is
        # n_clusters plus a uniform number of extras.
        assert res.clustering.n_clusters >= n_clusters


@pytest.mark.skipif(
    not SUPPORTS_PRESERVE, reason="tsam lacks preserve_n_clusters (#410)"
)
class TestCountStability:
    """Strict count guarantees, only meaningful when the flag is available."""

    def test_sliced_append_pins_count(self) -> None:
        extremes = tsam.ExtremeConfig(
            method="append",
            max_value=["solar", "wind", "price"],
            min_value=["solar", "wind", "price"],
        )
        res = tsam_xarray.aggregate(
            _sliced_da(0, n_scenarios=4),
            time_dim="time",
            cluster_dim="variable",
            n_clusters=8,
            extremes=extremes,
        )
        assert res.clustering.n_clusters == 8

    def test_forced_even_for_single_aggregation(self) -> None:
        """The flag is forced whether or not the data is sliced."""
        rng = np.random.default_rng(0)
        time = pd.date_range("2020-01-01", periods=20 * 24, freq="h")
        da = xr.DataArray(
            rng.random((len(time), 3)),
            dims=["time", "variable"],
            coords={"time": time, "variable": VARIABLES},
        )
        extremes = tsam.ExtremeConfig(
            method="append", max_value=["solar"], min_value=["wind"]
        )
        res = tsam_xarray.aggregate(
            da, time_dim="time", cluster_dim="variable", n_clusters=6, extremes=extremes
        )
        assert res.clustering.n_clusters == 6

    def test_user_preserve_false_is_overridden(self) -> None:
        """The wrapper always forces the flag, even against an explicit False."""
        extremes = tsam.ExtremeConfig(
            method="append",
            max_value=["solar", "wind", "price"],
            min_value=["solar", "wind", "price"],
            preserve_n_clusters=False,
        )
        res = tsam_xarray.aggregate(
            _sliced_da(1, n_scenarios=3),
            time_dim="time",
            cluster_dim="variable",
            n_clusters=8,
            extremes=extremes,
        )
        assert res.clustering.n_clusters == 8


@pytest.mark.skipif(
    not SUPPORTS_REPLACE_TRANSFER,
    reason="tsam lacks transferable replace (#410)",
)
class TestReplaceTransfers:
    """`replace` now survives the transfer paths it used to be barred from."""

    def test_cluster_on_replace_allowed(self) -> None:
        da = _sliced_da(0).isel(scenario=0, drop=True)
        res = tsam_xarray.aggregate(
            da,
            time_dim="time",
            cluster_dim="variable",
            n_clusters=6,
            cluster_on=["solar", "wind"],
            extremes=tsam.ExtremeConfig(
                method="replace", max_value=["solar"], min_value=["wind"]
            ),
        )
        # passive column carried through the transfer
        assert "price" in res.cluster_representatives.coords["variable"].values

    def test_replace_roundtrips_through_apply(self) -> None:
        da = _sliced_da(2).isel(scenario=0, drop=True)
        extremes = tsam.ExtremeConfig(
            method="replace", max_value=["solar", "wind"], min_value=["solar", "wind"]
        )
        direct = tsam_xarray.aggregate(
            da, time_dim="time", cluster_dim="variable", n_clusters=8, extremes=extremes
        )
        applied = direct.clustering.apply(da)
        np.testing.assert_allclose(
            direct.cluster_representatives.values,
            applied.cluster_representatives.transpose(
                *direct.cluster_representatives.dims
            ).values,
        )
