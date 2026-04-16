#!/usr/bin/env python3
"""
Script: 1c_stack_era5land_daily.py
Objective: Stack ERA5-Land daily GeoTIFFs into daily NetCDF forcing products and derive short-reference ET.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-04-16
Inputs: Daily ERA5-Land GeoTIFFs, a DEM GeoTIFF, and a date range.
Outputs: Daily NetCDF products for precipitation, temperature, radiation, vapor pressure, and reference ET.
Usage: python workflows/1c_stack_era5land_daily.py --help
Dependencies: argparse, datetime, pathlib, numpy, rasterio, xarray, core.era5land_refet, core.era5land_stack
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import xy

from core.era5land_refet import (
    actual_vapor_pressure_from_dewpoint_c,
    compute_daily_eto_short,
    j_per_m2_to_mj_per_m2_day,
    kelvin_to_celsius,
    meters_to_mm_day,
    wind_speed_from_uv,
)
from core.era5land_stack import discover_daily_files, extract_date_from_path

REQUIRED_BANDS = (
    "temperature_2m_min",
    "temperature_2m_max",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_solar_radiation_downwards_sum",
    "total_precipitation_sum",
)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date: {value!r}") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stack ERA5-Land DAILY_AGGR GeoTIFFs into daily NetCDF forcing files."
    )
    parser.add_argument("--raw-dir", required=True, help="Directory containing daily ERA5-Land GeoTIFF downloads.")
    parser.add_argument("--dem", required=True, help="DEM GeoTIFF used to derive elevation and latitude coordinates.")
    parser.add_argument("--date-range", nargs=2, metavar=("START", "END"), required=True, help="Inclusive date range.")
    parser.add_argument("--output-dir", required=True, help="Directory for NetCDF outputs.")
    return parser.parse_args(argv)


def _read_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, rasterio.Affine, str]:
    with rasterio.open(path) as src:
        array = src.read(1, masked=True).filled(np.nan).astype(float)
        transform = src.transform
        crs = src.crs.to_string() if src.crs is not None else "EPSG:4326"
        rows = np.arange(src.height)
        cols = np.arange(src.width)
        latitudes = np.asarray(xy(transform, rows, np.zeros_like(rows), offset="center")[1], dtype=float)
        longitudes = np.asarray(xy(transform, np.zeros_like(cols), cols, offset="center")[0], dtype=float)
    return array, latitudes, longitudes, transform, crs


def _read_daily_file(path: Path) -> dict[str, np.ndarray]:
    with rasterio.open(path) as src:
        descriptions = list(src.descriptions or ())
        band_lookup = {desc: idx + 1 for idx, desc in enumerate(descriptions) if desc}
        missing = [band for band in REQUIRED_BANDS if band not in band_lookup]
        if missing:
            raise ValueError(f"{path.name} is missing expected bands: {', '.join(missing)}")

        data: dict[str, np.ndarray] = {}
        for band_name in REQUIRED_BANDS:
            data[band_name] = src.read(band_lookup[band_name], masked=True).filled(np.nan).astype(float)
        return data


def _filter_date_range(paths: list[Path], start: date, end: date) -> list[Path]:
    selected = [path for path in paths if start <= extract_date_from_path(path) <= end]
    if not selected:
        raise ValueError(f"No daily ERA5-Land files were found between {start} and {end}.")
    return selected


def _write_netcdf(
    output_path: Path,
    var_name: str,
    values: np.ndarray,
    times: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    units: str,
    long_name: str,
) -> None:
    data_array = xr.DataArray(
        values.astype(np.float32, copy=False),
        dims=("time", "lat", "lon"),
        coords={
            "time": times,
            "lat": latitudes,
            "lon": longitudes,
        },
        name=var_name,
        attrs={
            "long_name": long_name,
            "units": units,
        },
    )
    data_array.to_dataset(name=var_name).to_netcdf(output_path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start_date = _parse_date(args.date_range[0])
    end_date = _parse_date(args.date_range[1])
    if start_date > end_date:
        raise ValueError("START must be on or before END.")

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_paths = _filter_date_range(discover_daily_files(raw_dir), start_date, end_date)

    dem, latitudes, longitudes, dem_transform, dem_crs = _read_grid(Path(args.dem))
    if dem.ndim != 2:
        raise ValueError("DEM must be a single-band two-dimensional raster.")

    records = []
    for path in daily_paths:
        with rasterio.open(path) as src:
            if src.width != dem.shape[1] or src.height != dem.shape[0]:
                raise ValueError(f"{path.name} does not match the DEM grid shape.")
            if src.transform != dem_transform:
                raise ValueError(f"{path.name} does not match the DEM georeferencing.")
            if src.crs is not None and dem_crs is not None and src.crs.to_string() != dem_crs:
                raise ValueError(f"{path.name} does not match the DEM CRS.")

        daily = _read_daily_file(path)
        daily_date = extract_date_from_path(path)
        records.append((daily_date, daily))

    records.sort(key=lambda item: item[0])
    dates = np.asarray([np.datetime64(record_date) for record_date, _ in records])
    doy = np.asarray([record_date.timetuple().tm_yday for record_date, _ in records], dtype=float)[:, None, None]

    tmin_c = np.stack(
        [kelvin_to_celsius(record["temperature_2m_min"]) for _, record in records],
        axis=0,
    )
    tmax_c = np.stack(
        [kelvin_to_celsius(record["temperature_2m_max"]) for _, record in records],
        axis=0,
    )
    tdew_c = np.stack(
        [kelvin_to_celsius(record["dewpoint_temperature_2m"]) for _, record in records],
        axis=0,
    )
    ea_kpa = actual_vapor_pressure_from_dewpoint_c(tdew_c)
    wind_speed = wind_speed_from_uv(
        np.stack([record["u_component_of_wind_10m"] for _, record in records], axis=0),
        np.stack([record["v_component_of_wind_10m"] for _, record in records], axis=0),
    )
    rs_mj_m2_day = j_per_m2_to_mj_per_m2_day(
        np.stack([record["surface_solar_radiation_downwards_sum"] for _, record in records], axis=0)
    )
    precipitation_mm_day = meters_to_mm_day(
        np.stack([record["total_precipitation_sum"] for _, record in records], axis=0)
    )
    et_short_crop = compute_daily_eto_short(
        tmax_c=tmax_c,
        tmin_c=tmin_c,
        ea_kpa=ea_kpa,
        rs_mj_m2_day=rs_mj_m2_day,
        uz_m_s=wind_speed,
        zw_m=10.0,
        elev_m=dem,
        lat_deg=latitudes[:, None],
        doy=doy,
    )

    start_label = start_date.isoformat()
    end_label = end_date.isoformat()
    _write_netcdf(
        output_dir / f"precipitation_daily_{start_label}_{end_label}.nc",
        "precipitation",
        precipitation_mm_day,
        dates,
        latitudes,
        longitudes,
        "mm day-1",
        "Daily precipitation",
    )
    _write_netcdf(
        output_dir / f"tmax_daily_{start_label}_{end_label}.nc",
        "tmax",
        tmax_c,
        dates,
        latitudes,
        longitudes,
        "degree_Celsius",
        "Daily maximum air temperature",
    )
    _write_netcdf(
        output_dir / f"tmin_daily_{start_label}_{end_label}.nc",
        "tmin",
        tmin_c,
        dates,
        latitudes,
        longitudes,
        "degree_Celsius",
        "Daily minimum air temperature",
    )
    _write_netcdf(
        output_dir / f"rs_daily_{start_label}_{end_label}.nc",
        "rs",
        rs_mj_m2_day,
        dates,
        latitudes,
        longitudes,
        "MJ m-2 day-1",
        "Daily incoming shortwave radiation",
    )
    _write_netcdf(
        output_dir / f"ea_daily_{start_label}_{end_label}.nc",
        "ea",
        ea_kpa,
        dates,
        latitudes,
        longitudes,
        "kPa",
        "Daily actual vapor pressure",
    )
    _write_netcdf(
        output_dir / f"et_short_crop_daily_{start_label}_{end_label}.nc",
        "et_short_crop",
        et_short_crop,
        dates,
        latitudes,
        longitudes,
        "mm day-1",
        "Daily short-reference evapotranspiration",
    )


if __name__ == "__main__":
    main()
