#!/usr/bin/env python3
"""
Script: core.py
Objective: Provide package-native SSEBop physics helpers for dT, ET fraction, climatology, and daily ET calculations.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: xarray temperature and geospatial arrays plus pandas ET series.
Outputs: SSEBop helper DataArray and Series results.
Usage: Imported by pysweb.ssebop consumers.
Dependencies: numpy, pandas, xarray
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "build_doy_climatology",
    "compute_dt_daily",
    "daily_et_from_etf",
    "dt_fao56_xr",
    "et_fraction_xr",
    "tcold_fano_simple_xr",
]


class TcoldConfig(Protocol):
    """Structural config interface for FANO-style Tcold tuning."""

    dt_coeff: float
    high_ndvi_threshold: float


def dt_fao56_xr(
    tmax_k: xr.DataArray,
    tmin_k: xr.DataArray,
    elev_m: xr.DataArray,
    doy: xr.DataArray | int,
    lat_deg: xr.DataArray,
    rs_mj_m2_day: xr.DataArray | None = None,
    ea_kpa: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute dT using the SSEBop FAO56-based formulation (xarray)."""
    if doy is None:
        raise ValueError("doy is required for dT calculation")

    phi = np.deg2rad(lat_deg)
    if isinstance(doy, xr.DataArray):
        doy_arr = doy
    else:
        doy_arr = xr.zeros_like(tmax_k) + float(doy)

    delta = np.sin((2 * np.pi / 365) * doy_arr - 1.39) * 0.409
    ws = np.arccos(-np.tan(phi) * np.tan(delta))
    dr = np.cos((2 * np.pi / 365) * doy_arr) * 0.033 + 1.0
    ra = (
        (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
        * dr
        * ((1367.0 / np.pi) * 0.0820)
    )
    rso = (0.75 + 2e-5 * elev_m) * ra

    if rs_mj_m2_day is None:
        rs = rso
        fcd = 1.0
    else:
        rs = rs_mj_m2_day
        fcd = np.clip(rs / rso, 0.3, 1.0) * 1.35 - 0.35

    rns = rs * (1 - 0.23)

    if ea_kpa is None:
        ea = np.exp((17.27 * (tmin_k - 273.15)) / ((tmin_k - 273.15) + 237.3)) * 0.6108
    else:
        ea = ea_kpa

    rnl = (
        (tmax_k**4 + tmin_k**4)
        * (np.sqrt(ea) * -0.14 + 0.34)
        * (4.901e-9 * 0.5)
        * fcd
    )
    rn = rns - rnl

    pair = 101.3 * ((293.0 - 0.0065 * elev_m) / 293.0) ** 5.26
    den = (pair * (3.486 / 1.01)) / ((tmax_k + tmin_k) * 0.5)

    return rn / den * (110.0 / ((1.013 / 1000) * 86400))


def et_fraction_xr(
    lst_k: xr.DataArray,
    tcold_k: xr.DataArray,
    dt_k: xr.DataArray,
    clamp_max: float = 1.0,
    mask_max: float = 2.0,
) -> xr.DataArray:
    """Compute SSEBop ET fraction from LST, Tcold, and dT."""
    etf = (tcold_k + dt_k - lst_k) / dt_k
    etf = etf.where(etf <= mask_max)
    return etf.clip(0.0, clamp_max)


def tcold_fano_simple_xr(
    lst_k: xr.DataArray,
    ndvi: xr.DataArray,
    dt_k: xr.DataArray,
    config: TcoldConfig | None = None,
) -> xr.DataArray:
    """Simplified FANO-style Tcold estimate for gridded data."""
    dt_coeff = 0.125 if config is None else config.dt_coeff
    high_ndvi_threshold = 0.9 if config is None else config.high_ndvi_threshold
    tc = lst_k - (dt_coeff * dt_k * (high_ndvi_threshold - ndvi) * 10.0)
    return tc


def build_doy_climatology(data: xr.DataArray) -> xr.DataArray:
    """Build a day-of-year climatology from a daily DataArray."""
    if "time" not in data.dims:
        raise ValueError("Input data must have a time dimension")
    return data.groupby("time.dayofyear").mean("time", skipna=True)


def compute_dt_daily(
    tmax_k: xr.DataArray,
    tmin_k: xr.DataArray,
    elev_m: xr.DataArray,
    lat_deg: xr.DataArray,
    rs_mj_m2_day: xr.DataArray | None = None,
    ea_kpa: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute daily dT for a time series using the time coordinate DOY."""
    if "time" not in tmax_k.dims:
        raise ValueError("tmax_k must have a time dimension")
    doy = tmax_k["time"].dt.dayofyear
    doy = xr.DataArray(doy, coords={"time": tmax_k["time"]}, dims=("time",))
    return dt_fao56_xr(
        tmax_k=tmax_k,
        tmin_k=tmin_k,
        elev_m=elev_m,
        doy=doy,
        lat_deg=lat_deg,
        rs_mj_m2_day=rs_mj_m2_day,
        ea_kpa=ea_kpa,
    )


def daily_et_from_etf(
    etf_series: pd.Series,
    et_reference_series: pd.Series,
    max_gap_days: int = 32,
) -> pd.Series:
    """Interpolate ETf to daily and compute ETa (ETf * reference ET)."""
    if etf_series.empty or et_reference_series.empty:
        raise ValueError("etf_series and et_reference_series must be non-empty")

    etf = etf_series.sort_index()
    target_index = et_reference_series.sort_index().index
    etf = etf.reindex(target_index.union(etf.index))
    etf_interp = etf.interpolate(method="time")

    prev_obs = etf_interp.index.to_series().where(etf.notna()).ffill()
    next_obs = etf_interp.index.to_series().where(etf.notna()).bfill()
    gap_days = (next_obs - prev_obs).dt.days
    etf_interp = etf_interp.where(gap_days <= max_gap_days)

    etf_daily = etf_interp.reindex(target_index)
    eta = etf_daily * et_reference_series
    return eta.rename("et_ssebop")
