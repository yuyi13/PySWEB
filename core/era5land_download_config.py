#!/usr/bin/env python3
"""
Script: era5land_download_config.py
Objective: Build a pure Google Earth Engine config for ERA5-Land DAILY_AGGR downloads.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-04-16
Inputs: Start and end dates, spatial extent, and an output directory.
Outputs: A dictionary ready to serialize to a GEE downloader YAML config.
Usage: import and call build_era5land_cfg(...)
Dependencies: datetime
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

ERA5LAND_BANDS = [
    "temperature_2m_min",
    "temperature_2m_max",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_solar_radiation_downwards_sum",
    "total_precipitation_sum",
]


def _parse_date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date: {value!r}") from exc


def _validate_extent(extent: Iterable[float]) -> list[float]:
    coords = [float(value) for value in extent]
    if len(coords) != 4:
        raise ValueError("extent must contain four coordinates: min_lon, min_lat, max_lon, max_lat")
    return coords


def build_era5land_cfg(
    start_date: str,
    end_date: str,
    extent: Iterable[float],
    out_dir: str,
) -> dict:
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    if start_dt > end_dt:
        raise ValueError("start_date must be on or before end_date.")
    coords = _validate_extent(extent)

    return {
        "collection": "ECMWF/ERA5_LAND/DAILY_AGGR",
        "coords": coords,
        "download_dir": out_dir,
        "start_year": start_dt.year,
        "start_month": start_dt.month,
        "start_day": start_dt.day,
        "end_year": end_dt.year,
        "end_month": end_dt.month,
        "end_day": end_dt.day,
        "bands": list(ERA5LAND_BANDS),
        "scale": 11132,
        "crs": "EPSG:4326",
        "out_format": "tif",
        "auth_mode": "browser",
        "filename_prefix": "ERA5LandDaily",
        "daily_strategy": "first",
        "postprocess": {
            "maskval_to_na": False,
            "enforce_float32": False,
        },
    }
