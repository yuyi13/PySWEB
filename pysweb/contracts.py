"""
Script: contracts.py
Objective: Validate daily calendars, geographic averaging weights and soil-layer geometry.
Author: Yi Yu (with assistance from Codex)
Created: 2026-09-12
Last updated: 2026-09-12
Inputs: Dated DataArrays, geographic coordinates, date ranges and layer-bottom depths.
Outputs: Validated daily arrays, month boundaries, layer depths and latitude weights.
Usage: from pysweb.contracts import daily_data, month_bounds, layer_bottoms, geographic_weights
Dependencies: numpy, pandas, xarray
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def daily_data(da, start=None, end=None, label="Input"):
    """Select a complete inclusive daily window; normalise daily timestamps."""
    if "time" not in da.coords or da.sizes.get("time", 0) == 0:
        raise ValueError(f"{label} requires a non-empty time coordinate.")
    times = pd.DatetimeIndex(da.time.values)
    if times.hasnans or not times.is_monotonic_increasing:
        raise ValueError(f"{label} time must be finite and increasing.")
    days = times.normalize()
    if days.has_duplicates:
        raise ValueError(f"{label} contains multiple timesteps on one day.")
    first = pd.Timestamp(start).normalize() if start else days[0]
    last = pd.Timestamp(end).normalize() if end else days[-1]
    if last < first:
        raise ValueError(f"{label} end precedes start.")
    expected = pd.date_range(first, last, freq="D")
    missing = expected.difference(days)
    if len(missing):
        preview = ", ".join(x.strftime("%Y-%m-%d") for x in missing[:10])
        raise ValueError(f"{label} missing requested daily timesteps: {preview}")
    return da.assign_coords(time=days).sel(time=expected)


def month_bounds(start, end):
    first = pd.Timestamp(start).normalize().replace(day=1)
    last = pd.Timestamp(end).normalize() + pd.offsets.MonthEnd(0)
    return first, last


def layer_bottoms(values):
    bottoms = np.asarray(values, dtype=float)
    if (
        bottoms.ndim != 1
        or not bottoms.size
        or not np.all(np.isfinite(bottoms))
        or np.any(bottoms <= 0)
        or np.any(np.diff(bottoms) <= 0)
    ):
        raise ValueError(
            "Layer bottoms must be finite, positive and strictly increasing (mm)."
        )
    return bottoms


def geographic_weights(da, lat_dim="lat", lon_dim="lon"):
    """Relative spherical cell areas from regular geographic cell-centre coordinates."""
    lat = np.asarray(da[lat_dim], dtype=float)
    lon = np.asarray(da[lon_dim], dtype=float)
    for name, values in [(lat_dim, lat), (lon_dim, lon)]:
        if values.ndim != 1 or not np.all(np.isfinite(values)) or not values.size:
            raise ValueError(f"Invalid coordinate {name}.")
        if len(values) > 1:
            spacing = np.diff(values)
            if spacing[0] == 0 or not np.allclose(
                spacing, spacing[0], rtol=1e-5, atol=1e-8
            ):
                raise ValueError(
                    "Area weights require a regular geographic grid; supply cell areas for other grids."
                )
    if np.any(np.abs(lat) > 90) or np.any(np.abs(lon) > 360):
        raise ValueError(
            "Area weights require geographic latitude/longitude in degrees."
        )
    weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    return xr.DataArray(
        weights,
        dims=(lat_dim, lon_dim),
        coords={lat_dim: da[lat_dim], lon_dim: da[lon_dim]},
    )
