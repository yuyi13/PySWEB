"""ERA5-Land download configuration and execution helpers for pysweb."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
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


def write_era5land_config(
    start_date: str,
    end_date: str,
    extent: list[float],
    output_dir: str,
) -> Path:
    cfg = build_era5land_cfg(
        start_date = start_date,
        end_date   = end_date,
        extent     = extent,
        out_dir    = output_dir,
    )
    cfg_path = Path(output_dir) / f"gee_config_era5land_{start_date}_{end_date}.yaml"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg_path


def download_era5land_daily(
    start_date: str,
    end_date: str,
    extent: list[float],
    output_dir: str,
    downloader_cls=None,
) -> Path:
    if downloader_cls is None:
        from pysweb.io.gee import GEEDownloader

        downloader_cls = GEEDownloader
    cfg_path = write_era5land_config(
        start_date = start_date,
        end_date   = end_date,
        extent     = extent,
        output_dir = output_dir,
    )
    downloader_cls(str(cfg_path)).run()
    return cfg_path
