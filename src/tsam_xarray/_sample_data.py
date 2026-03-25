"""Synthetic sample data for documentation and testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def sample_energy_data(
    n_days: int = 30,
    seed: int = 42,
) -> xr.DataArray:
    """Create a synthetic energy DataArray with realistic profiles.

    Returns an hourly DataArray with dimensions:

    - **time** — hourly timestamps
    - **variable** — ``solar``, ``wind``, ``demand``
    - **region** — ``north``, ``south``, ``east``
    - **scenario** — ``low``, ``high``

    Solar follows a daily bell curve, wind has seasonal variation
    with autocorrelation, and demand combines a daily commute pattern
    with weather-driven noise. Scenarios scale the base profiles.

    Parameters
    ----------
    n_days : int
        Number of days of hourly data (default: 30).
    seed : int
        Random seed for reproducibility (default: 42).

    Returns
    -------
    xr.DataArray
        Shape ``(n_days * 24, 3, 3, 2)`` with coords on every dim.
    """
    rng = np.random.default_rng(seed)
    hours = n_days * 24
    time = pd.date_range("2020-01-01", periods=hours, freq="h")
    hour_of_day = np.arange(hours) % 24
    day_of_year = time.dayofyear.values

    variables = ["solar", "wind", "demand"]
    regions = ["north", "south", "east"]
    scenarios = ["low", "high"]

    # --- base profiles (hours,) ---
    # Solar: bell curve peaking at noon, zero at night
    solar_base = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12)) ** 1.5
    # Seasonal envelope: weaker in winter
    solar_season = 0.6 + 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    solar = solar_base * solar_season

    # Wind: autocorrelated noise with seasonal mean
    wind = np.empty(hours)
    wind[0] = 0.5
    for t in range(1, hours):
        wind[t] = 0.9 * wind[t - 1] + 0.1 * rng.standard_normal()
    wind = (wind - wind.min()) / (wind.max() - wind.min())
    wind_season = 0.7 + 0.3 * np.cos(2 * np.pi * (day_of_year - 1) / 365)
    wind = wind * wind_season

    # Demand: daily pattern + seasonal + noise
    demand_daily = 0.5 + 0.3 * np.sin(np.pi * (hour_of_day - 5) / 12)
    demand_season = 1.0 + 0.2 * np.cos(2 * np.pi * (day_of_year - 1) / 365)
    demand = demand_daily * demand_season + 0.05 * rng.standard_normal(hours)
    demand = np.clip(demand, 0, None)

    bases = np.stack([solar, wind, demand], axis=-1)  # (hours, 3)

    # --- region modifiers ---
    region_scales = np.array(
        [
            [0.7, 1.3, 1.1],  # north: less solar, more wind, slightly more demand
            [1.3, 0.7, 0.9],  # south: more solar, less wind, less demand
            [1.0, 1.0, 1.0],  # east: baseline
        ]
    )  # (3 regions, 3 variables)

    # (hours, variables, regions)
    data_3d = bases[:, :, np.newaxis] * region_scales.T[np.newaxis, :, :]

    # --- scenario scaling ---
    scenario_scales = np.array([0.8, 1.2])  # low, high
    # (hours, variables, regions, scenarios)
    data_4d = (
        data_3d[:, :, :, np.newaxis]
        * scenario_scales[np.newaxis, np.newaxis, np.newaxis, :]
    )

    # Add a small amount of noise per cell
    data_4d += 0.02 * rng.standard_normal(data_4d.shape)
    data_4d = np.clip(data_4d, 0, None)

    return xr.DataArray(
        data_4d,
        dims=["time", "variable", "region", "scenario"],
        coords={
            "time": time,
            "variable": variables,
            "region": regions,
            "scenario": scenarios,
        },
        name="energy",
    )
