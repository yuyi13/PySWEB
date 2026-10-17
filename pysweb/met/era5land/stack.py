#!/usr/bin/env python3
"""
Script: stack.py
Objective: Stack ERA5-Land daily GeoTIFF downloads into NetCDF meteorology products.
Author: Yi Yu (with assistance from Codex)
Created: 2026-04-17
Last updated: 2026-10-17
Inputs: ERA5-Land daily GeoTIFFs, DEM GeoTIFF, date range, and output directory.
Outputs: Daily NetCDF files for precipitation, air temperature, radiation, vapor pressure, and short/tall-reference ET.
Usage: python -m pysweb.met.era5land.stack --help
Dependencies: numpy, rasterio, scipy, xarray; pysweb
"""
from __future__ import annotations

import argparse
import os
import re
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import xy

from pysweb.met.era5land.refet import (
    actual_vapor_pressure_from_dewpoint_c,
    compute_daily_eto_short,
    compute_daily_etr_tall,
    j_per_m2_to_mj_per_m2_day,
    kelvin_to_celsius,
    meters_to_mm_day,
    wind_speed_from_uv,
)

DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
DAILY_RASTER_SUFFIXES = {".tif", ".tiff"}
REQUIRED_BANDS = (
    "temperature_2m_min",
    "temperature_2m_max",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_solar_radiation_downwards_sum",
    "total_precipitation_sum",
)
NETCDF_ENGINE = "scipy"
NETCDF_FORMAT = "NETCDF3_64BIT"


def extract_date_from_path(path: Path) -> date:
    match = DATE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not find an embedded date in {path.name!r}.")
    return date.fromisoformat(match.group(1))


def discover_daily_files(raw_dir: Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    daily_files: list[Path] = []
    for path in raw_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in DAILY_RASTER_SUFFIXES:
            continue
        try:
            extract_date_from_path(path)
        except ValueError:
            continue
        daily_files.append(path)

    return sorted(daily_files, key=lambda path: (extract_date_from_path(path), path.name))


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date: {value!r}") from exc


def native_download_extent(raw_dir, start_date, end_date) -> list[float]:
    """Read the complete exported ERA5 pixel footprint for DEM acquisition."""
    from rasterio.warp import transform_bounds

    paths = _filter_date_range(
        discover_daily_files(Path(raw_dir)), _parse_date(start_date), _parse_date(end_date)
    )
    with rasterio.open(paths[0]) as source:
        if source.crs is None:
            raise ValueError("ERA5 download has no CRS")
        return list(transform_bounds(source.crs, "EPSG:4326", *source.bounds, densify_pts=21))


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
        array = src.read(1, masked=True).astype(float).filled(np.nan)
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
            data[band_name] = src.read(band_lookup[band_name], masked=True).astype(float).filled(np.nan)
        return data


def _filter_date_range(paths: list[Path], start: date, end: date) -> list[Path]:
    selected = [path for path in paths if start <= extract_date_from_path(path) <= end]
    if not selected:
        raise ValueError(f"No daily ERA5-Land files were found between {start} and {end}.")
    expected_dates = {start + timedelta(days=offset) for offset in range((end - start).days + 1)}
    selected_dates = {extract_date_from_path(path) for path in selected}
    if selected_dates != expected_dates:
        missing = ", ".join(sorted(date_value.isoformat() for date_value in expected_dates - selected_dates))
        raise ValueError(f"ERA5-Land daily files do not cover every date in the range. Missing: {missing}")
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
    transform=None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    data_array = xr.DataArray(
        values.astype(np.float32, copy=False),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": latitudes, "lon": longitudes},
        name=var_name,
        attrs={"long_name": long_name, "units": units},
    )
    import rioxarray
    data_array = data_array.rio.set_spatial_dims(x_dim='lon',y_dim='lat').rio.write_crs('EPSG:4326')
    if transform is not None:
        data_array = data_array.rio.write_transform(transform)
    data_array = data_array.rio.write_coordinate_system()
    try:
        data_array.to_dataset(name=var_name).to_netcdf(
            tmp_path,
            engine=NETCDF_ENGINE,
            format=NETCDF_FORMAT,
        )
        _validate_netcdf(tmp_path, var_name)
        try:
            tmp_path.replace(output_path)
        except PermissionError as exc:
            raise PermissionError(
                f"Could not replace {output_path}. Close any program that may have "
                "the NetCDF file open, then rerun the stack step."
            ) from exc
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _validate_netcdf(path: Path, var_name: str) -> None:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"NetCDF write produced an empty file: {path}")

    try:
        with xr.open_dataset(path, engine=NETCDF_ENGINE) as ds:
            if var_name not in ds.data_vars:
                raise RuntimeError(f"NetCDF file {path} does not contain variable '{var_name}'.")
            if ds[var_name].size == 0:
                raise RuntimeError(f"NetCDF variable '{var_name}' is empty in {path}.")
    except Exception as exc:
        raise RuntimeError(f"NetCDF validation failed for {path}: {exc}") from exc


def stack_era5land_daily_inputs(
    raw_dir,
    dem,
    start_date,
    end_date,
    output_dir,
) -> None:
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    if start_dt > end_dt:
        raise ValueError("START must be on or before END.")

    raw_dir = Path(raw_dir)
    dem = Path(dem)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_paths = _filter_date_range(discover_daily_files(raw_dir), start_dt, end_dt)

    # Downloads retain native grids. Aggregate the native DEM explicitly here
    # while the meteorological calculations stay on the original ERA5 grid.
    from rasterio.warp import reproject, transform_bounds
    from rasterio.enums import Resampling
    _, latitudes, longitudes, dem_transform, dem_crs = _read_grid(daily_paths[0])
    with rasterio.open(daily_paths[0]) as target, rasterio.open(dem) as source:
        if target.crs is None or target.crs.to_epsg() != 4326:
            raise ValueError("ERA5 meteorology requires its native geographic grid")
        if source.crs is None:
            raise ValueError("DEM has no CRS")
        required = transform_bounds(target.crs, source.crs, *target.bounds, densify_pts=21)
        tolerance = max(abs(source.transform.a), abs(source.transform.e)) * 1e-6
        if (source.bounds.left > required[0] + tolerance
                or source.bounds.bottom > required[1] + tolerance
                or source.bounds.right < required[2] - tolerance
                or source.bounds.top < required[3] - tolerance):
            raise ValueError("DEM must cover complete native ERA5 pixel footprints")
        elevation = source.read(1, masked=True).astype(float).filled(np.nan)
        dem_array = np.full((target.height, target.width), np.nan)
        reproject(source=elevation,
                  destination=dem_array, src_transform=source.transform, src_crs=source.crs,
                  src_nodata=np.nan, dst_transform=target.transform, dst_crs=target.crs,
                  dst_nodata=np.nan, resampling=Resampling.average)
        support = np.zeros_like(dem_array)
        reproject(source=np.isfinite(elevation).astype(float), destination=support,
                  src_transform=source.transform, src_crs=source.crs,
                  dst_transform=target.transform, dst_crs=target.crs,
                  resampling=Resampling.average)
        dem_array[support < 1 - 1e-6] = np.nan

    from functools import partial
    write_stack = partial(_write_netcdf,transform=dem_transform)
    records = []
    for path in daily_paths:
        with rasterio.open(path) as src:
            if src.width != dem_array.shape[1] or src.height != dem_array.shape[0]:
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
        tmax_c = tmax_c,
        tmin_c = tmin_c,
        ea_kpa = ea_kpa,
        rs_mj_m2_day = rs_mj_m2_day,
        uz_m_s = wind_speed,
        zw_m = 10.0,
        elev_m = dem_array,
        lat_deg = latitudes[:, None],
        doy = doy,
    )

    et_tall_crop = compute_daily_etr_tall(
        tmax_c = tmax_c,
        tmin_c = tmin_c,
        ea_kpa = ea_kpa,
        rs_mj_m2_day = rs_mj_m2_day,
        uz_m_s = wind_speed,
        zw_m = 10.0,
        elev_m = dem_array,
        lat_deg = latitudes[:, None],
        doy = doy,
    )

    start_label = start_dt.isoformat()
    end_label = end_dt.isoformat()
    write_stack(
        output_dir / f"precipitation_daily_{start_label}_{end_label}.nc",
        "precipitation",
        precipitation_mm_day,
        dates,
        latitudes,
        longitudes,
        "mm day-1",
        "Daily precipitation",
    )
    write_stack(
        output_dir / f"tmax_daily_{start_label}_{end_label}.nc",
        "tmax",
        tmax_c,
        dates,
        latitudes,
        longitudes,
        "degree_Celsius",
        "Daily maximum air temperature",
    )
    write_stack(
        output_dir / f"tmin_daily_{start_label}_{end_label}.nc",
        "tmin",
        tmin_c,
        dates,
        latitudes,
        longitudes,
        "degree_Celsius",
        "Daily minimum air temperature",
    )
    write_stack(
        output_dir / f"rs_daily_{start_label}_{end_label}.nc",
        "rs",
        rs_mj_m2_day,
        dates,
        latitudes,
        longitudes,
        "MJ m-2 day-1",
        "Daily incoming shortwave radiation",
    )
    write_stack(
        output_dir / f"ea_daily_{start_label}_{end_label}.nc",
        "ea",
        ea_kpa,
        dates,
        latitudes,
        longitudes,
        "kPa",
        "Daily actual vapor pressure",
    )
    write_stack(
        output_dir / f"et_short_crop_daily_{start_label}_{end_label}.nc",
        "et_short_crop",
        et_short_crop,
        dates,
        latitudes,
        longitudes,
        "mm day-1",
        "Daily short-reference evapotranspiration",
    )

    write_stack(
        output_dir / f"et_tall_crop_daily_{start_label}_{end_label}.nc",
        "et_tall_crop", et_tall_crop, dates, latitudes, longitudes,
        "mm day-1", "Daily alfalfa-reference evapotranspiration",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    stack_era5land_daily_inputs(
        raw_dir = args.raw_dir,
        dem = args.dem,
        start_date = args.date_range[0],
        end_date = args.date_range[1],
        output_dir = args.output_dir,
    )
