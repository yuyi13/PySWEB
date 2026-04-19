#!/usr/bin/env python3
"""
Script: preprocess.py
Objective: Preprocess forcing, soil, and GSSM reference SSM inputs into aligned NetCDF files for SWB runs.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Command-line arguments or keyword arguments describing date range, extent, forcing inputs, Earth Engine assets, and output location.
Outputs: NetCDF forcing files, soil-property layers, and optional reference SSM products on a common grid.
Usage: Imported as `pysweb.swb.preprocess` or run as a module entry point.
Dependencies: argparse, concurrent.futures, earthengine-api, multiprocessing, numpy, pandas, pyproj, requests, rioxarray, rasterio, xarray
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import tempfile
import types
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import ee
import numpy as np
import pandas as pd
import requests
import rioxarray
import xarray as xr
import pysweb.soil.api as soil_api
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
GSSM_SCALE_FACTOR = 1000.0
GSSM_EXPORT_SCALE_M = 1000.0
_RAIN_WORKER_STATE: Dict[str, object] = {}
_ET_WORKER_STATE: Dict[str, object] = {}


@dataclass(frozen = True)
class TargetGrid:
    template: xr.DataArray
    latitudes: np.ndarray
    longitudes: np.ndarray
    lat_dim: str
    lon_dim: str
    crs: str


def _pool_context():
    try:
        return mp.get_context("fork")
    except ValueError:
        return None


def _missing_date_preview(missing_dates: pd.DatetimeIndex) -> str:
    preview = ", ".join(timestamp.strftime("%Y-%m-%d") for timestamp in missing_dates[:10])
    if len(missing_dates) > 10:
        preview = f"{preview}, ..."
    return preview


def _require_daily_coverage(
    da: xr.DataArray,
    expected_dates: Sequence[pd.Timestamp],
    label: str,
    *,
    require_complete: bool = True,
) -> xr.DataArray:
    if "time" not in da.coords:
        raise ValueError(f"{label} data must include a 'time' coordinate.")

    expected_index = pd.DatetimeIndex(expected_dates).sort_values().normalize()
    actual_times = pd.DatetimeIndex(da.coords["time"].values)
    actual_index = actual_times.normalize()
    duplicate_dates = pd.DatetimeIndex(actual_index[actual_index.duplicated(keep = False)]).unique()
    if not duplicate_dates.empty:
        raise ValueError(f"{label} data contains multiple timesteps for: {_missing_date_preview(duplicate_dates)}")

    positions_by_day = {day: idx for idx, day in enumerate(actual_index)}
    selected_days = [day for day in expected_index if day in positions_by_day]
    if require_complete:
        missing_dates = expected_index.difference(pd.DatetimeIndex(selected_days))
        if not missing_dates.empty:
            raise ValueError(f"{label} data missing requested days: {_missing_date_preview(missing_dates)}")
    elif not selected_days:
        return da.isel(time = slice(0, 0)).assign_coords(time = expected_index[:0])

    selected_index = pd.DatetimeIndex(selected_days)
    positions = [positions_by_day[day] for day in selected_index]
    selected = da.isel(time = positions)
    return selected.assign_coords(time = selected_index)


def _compute_monthly_effective_rainfall_smith(monthly_precip_mm: np.ndarray) -> np.ndarray:
    monthly = np.asarray(monthly_precip_mm, dtype = float)
    monthly_eff = np.full_like(monthly, np.nan, dtype = float)
    valid = np.isfinite(monthly)
    if not np.any(valid):
        return monthly_eff

    precip = np.maximum(monthly[valid], 0.0)
    effective = np.empty_like(precip)
    low_mask = precip <= 250.0
    effective[low_mask] = precip[low_mask] * (125.0 - 0.2 * precip[low_mask]) / 125.0
    effective[~low_mask] = 125.0 + 0.1 * precip[~low_mask]
    monthly_eff[valid] = np.clip(effective, 0.0, precip)
    return monthly_eff


def compute_effective_precipitation_smith(rain: xr.DataArray, dtype: str) -> xr.DataArray:
    if "time" not in rain.dims:
        raise ValueError("Precipitation data must include a 'time' dimension.")
    if rain.ndim != 3:
        raise ValueError("Precipitation data must include exactly 3 dimensions (time, lat, lon).")
    spatial_dims = [dim for dim in rain.dims if dim != "time"]
    if len(spatial_dims) != 2:
        raise ValueError("Precipitation data must include exactly two spatial dimensions.")

    rain = rain.transpose("time", spatial_dims[0], spatial_dims[1])
    rain_values = np.asarray(rain.values, dtype = float)
    time_index = pd.to_datetime(rain.coords["time"].values)
    month_codes = (time_index.year.to_numpy(dtype = int) * 100) + time_index.month.to_numpy(dtype = int)
    _, inverse = np.unique(month_codes, return_inverse = True)

    daily_valid = np.isfinite(rain_values)
    daily_rain_nonneg = np.zeros_like(rain_values, dtype = float)
    daily_rain_nonneg[daily_valid] = np.maximum(rain_values[daily_valid], 0.0)

    daily_eff = np.full_like(rain_values, np.nan, dtype = float)
    n_month = int(inverse.max()) + 1 if inverse.size else 0
    for month_idx in range(n_month):
        month_mask = inverse == month_idx
        if not np.any(month_mask):
            continue

        month_daily = daily_rain_nonneg[month_mask, :, :]
        month_valid = daily_valid[month_mask, :, :]
        monthly_total = month_daily.sum(axis = 0, dtype = float)
        monthly_valid_count = month_valid.sum(axis = 0)
        monthly_total[monthly_valid_count == 0] = np.nan

        monthly_eff = _compute_monthly_effective_rainfall_smith(monthly_total)
        monthly_ratio = np.zeros_like(monthly_total, dtype = float)
        valid_month = np.isfinite(monthly_total) & np.isfinite(monthly_eff) & (monthly_total > 0.0)
        monthly_ratio[valid_month] = monthly_eff[valid_month] / monthly_total[valid_month]

        month_daily_eff = month_daily * monthly_ratio[None, :, :]
        month_daily_eff[~month_valid] = np.nan
        daily_eff[month_mask, :, :] = month_daily_eff

    effective = xr.DataArray(
        daily_eff.astype(dtype, copy = False),
        dims = rain.dims,
        coords = rain.coords,
        name = "effective_precipitation",
        attrs = {
            "long_name": "Daily effective precipitation",
            "units": "mm day-1",
            "method": "Smith (1992) CROPWAT monthly effective rainfall scaled by daily precipitation share",
        },
    )
    return effective


def _subset_to_extent(
    da: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    extent: Tuple[float, float, float, float],
) -> xr.DataArray:
    min_lon, min_lat, max_lon, max_lat = extent
    lat_values = da.coords[lat_dim].values
    lat_slice = slice(max_lat, min_lat) if lat_values[0] > lat_values[-1] else slice(min_lat, max_lat)
    subset = da.sel({lat_dim: lat_slice, lon_dim: slice(min_lon, max_lon)})
    if subset.sizes[lat_dim] == 0 or subset.sizes[lon_dim] == 0:
        raise ValueError("Extent selection removed all grid cells; adjust --extent bounds.")
    return subset


def _transform_extent(
    extent: Tuple[float, float, float, float],
    src_crs: str,
    dst_crs: str,
) -> Tuple[float, float, float, float]:
    if src_crs == dst_crs:
        return extent
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy = True)
    min_lon, min_lat, max_lon, max_lat = extent
    corners = [
        transformer.transform(min_lon, min_lat),
        transformer.transform(min_lon, max_lat),
        transformer.transform(max_lon, min_lat),
        transformer.transform(max_lon, max_lat),
    ]
    xs, ys = zip(*corners)
    return (min(xs), min(ys), max(xs), max(ys))


def _clip_to_extent(da: xr.DataArray, grid: TargetGrid, extent: Tuple[float, float, float, float]) -> xr.DataArray:
    data_crs = da.rio.crs
    if data_crs is None:
        return da
    minx, miny, maxx, maxy = _transform_extent(extent, grid.crs, str(data_crs))
    try:
        res_x, res_y = da.rio.resolution()
        buffer_x = abs(res_x) * 2
        buffer_y = abs(res_y) * 2
    except Exception:
        buffer_x = buffer_y = 0.0
    return da.rio.clip_box(
        minx = minx - buffer_x,
        miny = miny - buffer_y,
        maxx = maxx + buffer_x,
        maxy = maxy + buffer_y,
    )


def _build_target_coordinates(
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    res: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (max_lon > min_lon and max_lat > min_lat):
        raise ValueError("Invalid extent bounds for building target grid.")

    lon_steps = int(round((max_lon - min_lon) / res))
    lat_steps = int(round((max_lat - min_lat) / res))
    if lon_steps < 1 or lat_steps < 1:
        raise ValueError("Extent must span at least one target grid cell in each axis.")

    lon_span = lon_steps * res
    lat_span = lat_steps * res
    if not np.isclose(lon_span, max_lon - min_lon, atol = res * 1e-4):
        raise ValueError("Longitude extent is not aligned with requested resolution.")
    if not np.isclose(lat_span, max_lat - min_lat, atol = res * 1e-4):
        raise ValueError("Latitude extent is not aligned with requested resolution.")

    target_lon = min_lon + (np.arange(lon_steps, dtype = float) + 0.5) * res
    target_lat = max_lat - (np.arange(lat_steps, dtype = float) + 0.5) * res

    return np.round(target_lat, 6), np.round(target_lon, 6)


def _transform_from_center_coords(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    res_lon: float,
    res_lat: float,
):
    west = float(np.min(longitudes)) - (res_lon / 2.0)
    north = float(np.max(latitudes)) + (res_lat / 2.0)
    return from_origin(west, north, res_lon, res_lat)


def _compute_transform(latitudes: np.ndarray, longitudes: np.ndarray):
    lat_arr = np.asarray(latitudes, dtype = float)
    lon_arr = np.asarray(longitudes, dtype = float)
    if lat_arr.size < 2 or lon_arr.size < 2:
        raise ValueError("At least two coordinate values are required to infer transform.")

    res_lat = float(np.abs(np.diff(np.sort(lat_arr)).mean()))
    res_lon = float(np.abs(np.diff(np.sort(lon_arr)).mean()))
    if res_lat == 0.0 or res_lon == 0.0:
        raise ValueError("Detected zero spatial resolution in forcing data.")

    return _transform_from_center_coords(lat_arr, lon_arr, res_lon, res_lat)


def _grid_resolution(latitudes: np.ndarray, longitudes: np.ndarray) -> Tuple[float, float]:
    lat_arr = np.asarray(latitudes, dtype = float)
    lon_arr = np.asarray(longitudes, dtype = float)
    if lat_arr.size < 2 or lon_arr.size < 2:
        raise ValueError("Target grid resolution cannot be inferred from single-cell coordinates.")
    res_lat = float(np.abs(np.diff(np.sort(lat_arr)).mean()))
    res_lon = float(np.abs(np.diff(np.sort(lon_arr)).mean()))
    return res_lon, res_lat


def _prepare_template(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    res_lon: float,
    res_lat: float,
    lat_dim: str,
    lon_dim: str,
    crs: str,
) -> xr.DataArray:
    transform = _transform_from_center_coords(latitudes, longitudes, res_lon, res_lat)
    template = xr.DataArray(
        np.zeros((latitudes.size, longitudes.size), dtype = np.uint8),
        coords = {lat_dim: latitudes, lon_dim: longitudes},
        dims = (lat_dim, lon_dim),
        name = "template_mask",
    )
    template = template.rio.write_crs(crs)
    template = template.rio.write_transform(transform)
    template = template.rio.set_spatial_dims(x_dim = lon_dim, y_dim = lat_dim)
    return template


def _crop_raster_to_grid(da: xr.DataArray, grid: TargetGrid, buffer_pixels: int = 2) -> xr.DataArray:
    min_lon = float(np.min(grid.longitudes))
    max_lon = float(np.max(grid.longitudes))
    min_lat = float(np.min(grid.latitudes))
    max_lat = float(np.max(grid.latitudes))

    try:
        res_x, res_y = da.rio.resolution()
        buffer_x = abs(res_x) * buffer_pixels
        buffer_y = abs(res_y) * buffer_pixels
    except Exception:
        buffer_x = buffer_y = 0.0

    data_crs = da.rio.crs
    if data_crs is not None and str(data_crs) != grid.crs:
        minx, miny, maxx, maxy = _transform_extent((min_lon, min_lat, max_lon, max_lat), grid.crs, str(data_crs))
    else:
        minx, miny, maxx, maxy = min_lon, min_lat, max_lon, max_lat

    return da.rio.clip_box(
        minx = minx - buffer_x,
        miny = miny - buffer_y,
        maxx = maxx + buffer_x,
        maxy = maxy + buffer_y,
    )


def _reproject_to_template(
    data: xr.DataArray,
    grid: TargetGrid,
    resampling: Resampling = Resampling.bilinear,
) -> xr.DataArray:
    if grid.lat_dim in data.dims and grid.lon_dim in data.dims:
        lat_name = grid.lat_dim
        lon_name = grid.lon_dim
    elif "y" in data.dims and "x" in data.dims:
        lat_name = "y"
        lon_name = "x"
    else:
        raise ValueError("DataArray must have latitude/longitude or y/x dimensions for reprojection.")

    data = data.sortby(lat_name, ascending = False)
    lat_vals = data.coords[lat_name].values
    lon_vals = data.coords[lon_name].values
    if lat_vals.size < 2 or lon_vals.size < 2:
        transform = grid.template.rio.transform()
    else:
        transform = _compute_transform(lat_vals, lon_vals)

    if data.rio.crs is None:
        data = data.rio.write_crs(grid.crs)
    data = data.rio.write_transform(transform)
    data = data.rio.set_spatial_dims(x_dim = lon_name, y_dim = lat_name)
    data = data.rio.reproject_match(grid.template, resampling = resampling)

    rename_map = {}
    if "y" in data.dims:
        rename_map["y"] = grid.lat_dim
    if "x" in data.dims:
        rename_map["x"] = grid.lon_dim
    if rename_map:
        data = data.rename(rename_map)
    return data.assign_coords(
        {
            grid.lat_dim: grid.template.coords[grid.lat_dim].values,
            grid.lon_dim: grid.template.coords[grid.lon_dim].values,
        }
    )


def _broadcast_single_pixel(data: xr.DataArray, grid: TargetGrid) -> xr.DataArray | None:
    if grid.lat_dim in data.dims and grid.lon_dim in data.dims:
        lat_name = grid.lat_dim
        lon_name = grid.lon_dim
    elif "y" in data.dims and "x" in data.dims:
        lat_name = "y"
        lon_name = "x"
    else:
        return None

    if data.sizes.get(lat_name, 0) != 1 or data.sizes.get(lon_name, 0) != 1:
        return None

    single = data.isel({lat_name: 0, lon_name: 0})
    if "time" in single.dims:
        values = np.asarray(single.values, dtype = float)
        return xr.DataArray(
            np.broadcast_to(values[:, None, None], (values.shape[0], grid.latitudes.size, grid.longitudes.size)),
            dims = ("time", grid.lat_dim, grid.lon_dim),
            coords = {
                "time": single.coords["time"].values,
                grid.lat_dim: grid.latitudes,
                grid.lon_dim: grid.longitudes,
            },
            name = data.name,
            attrs = data.attrs,
        )

    return xr.DataArray(
        np.full((grid.latitudes.size, grid.longitudes.size), float(single.values), dtype = float),
        dims = (grid.lat_dim, grid.lon_dim),
        coords = {grid.lat_dim: grid.latitudes, grid.lon_dim: grid.longitudes},
        name = data.name,
        attrs = data.attrs,
    )


def _generate_month_paths(root: Path, pattern: str, start: pd.Timestamp, end: pd.Timestamp) -> List[Path]:
    months = pd.period_range(start = start, end = end, freq = "M")
    paths = []
    for period in months:
        path = root / str(period.year) / pattern.format(year = period.year, month = period.month)
        if not path.exists():
            raise FileNotFoundError(f"Missing precipitation file: {path}")
        paths.append(path)
    return paths


def _generate_year_paths(root: Path, pattern: str, start: pd.Timestamp, end: pd.Timestamp) -> List[Path]:
    paths = []
    for year in range(start.year, end.year + 1):
        path = root / pattern.format(year = year)
        if not path.exists():
            raise FileNotFoundError(f"Missing precipitation file: {path}")
        paths.append(path)
    return paths


def _generate_daily_paths(root: Path, pattern: str, dates: Sequence[pd.Timestamp]) -> List[Path]:
    paths = []
    missing = []
    for date in dates:
        path = root / pattern.format(year = date.year, month = date.month, day = date.day, date = date)
        if not path.exists():
            missing.append(str(path))
        else:
            paths.append(path)
    if missing:
        suffix = " ..." if len(missing) > 5 else ""
        raise FileNotFoundError(f"Missing ET raster(s): {', '.join(missing[:5])}{suffix}")
    return paths


def _ensure_date_inputs(args: argparse.Namespace) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if args.date_range:
        if args.start_date or args.end_date:
            raise ValueError("--date-range cannot be combined with --start-date or --end-date.")
        args.start_date, args.end_date = args.date_range
    if not args.start_date or not args.end_date:
        raise ValueError("Both --start-date and --end-date (or --date-range) are required.")

    start = pd.to_datetime(args.start_date)
    end = pd.to_datetime(args.end_date)
    if end < start:
        raise ValueError("End date precedes start date.")
    return start, end


def _build_target_grid(args: argparse.Namespace) -> TargetGrid:
    if args.extent is None:
        raise ValueError("--extent must be provided (MIN_LON MIN_LAT MAX_LON MAX_LAT).")
    min_lon, min_lat, max_lon, max_lat = args.extent
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError("--extent requires MIN_LON < MAX_LON and MIN_LAT < MAX_LAT.")

    res = float(args.sm_res)
    if res <= 0.0:
        raise ValueError("--sm-res must be positive.")

    target_lat, target_lon = _build_target_coordinates(min_lon, max_lon, min_lat, max_lat, res)
    template = _prepare_template(target_lat, target_lon, res, res, args.lat_dim, args.lon_dim, args.crs)
    return TargetGrid(
        template = template,
        latitudes = template.coords[args.lat_dim].values,
        longitudes = template.coords[args.lon_dim].values,
        lat_dim = args.lat_dim,
        lon_dim = args.lon_dim,
        crs = args.crs,
    )


def _standard_encoding(dtype: str) -> Dict[str, object]:
    storage_dtype = np.dtype(dtype)
    return {
        "zlib": True,
        "complevel": 4,
        "dtype": storage_dtype.name,
        "_FillValue": storage_dtype.type(np.nan),
    }


def _print_progress(label: str, index: int, total: int) -> None:
    if total <= 0:
        return
    bar_len = 20
    filled = int(bar_len * index / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    end_char = "\n" if index == total else "\r"
    print(f"{label} [{bar}] {index}/{total}", end = end_char, flush = True)


def _load_rain_file_for_window(
    path: Path,
    rain_var: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    extent: Optional[Tuple[float, float, float, float]],
    grid: TargetGrid,
    lat_dim: str,
    lon_dim: str,
) -> Optional[xr.DataArray]:
    expected_dates = pd.date_range(start = start, end = end, freq = "D")
    with xr.open_dataset(path) as ds:
        if rain_var not in ds:
            raise KeyError(f"Variable '{rain_var}' not found in precipitation dataset: {path}")
        da = _require_daily_coverage(
            ds[rain_var],
            expected_dates,
            "Precipitation",
            require_complete = False,
        )
        if da.sizes.get("time", 0) == 0:
            return None
        if extent:
            if da.rio.crs is not None:
                da = _clip_to_extent(da, grid, extent)
            else:
                da = _subset_to_extent(da, lat_dim, lon_dim, extent)
        return da.load()


def _init_rain_worker(
    grid: TargetGrid,
    rain_var: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    extent: Optional[Tuple[float, float, float, float]],
    lat_dim: str,
    lon_dim: str,
) -> None:
    global _RAIN_WORKER_STATE
    _RAIN_WORKER_STATE = {
        "grid": grid,
        "rain_var": rain_var,
        "start": start,
        "end": end,
        "extent": extent,
        "lat_dim": lat_dim,
        "lon_dim": lon_dim,
    }


def _load_rain_file_task(task: Tuple[int, str]) -> Tuple[int, Optional[xr.DataArray]]:
    idx, path_str = task
    state = _RAIN_WORKER_STATE
    da = _load_rain_file_for_window(
        path = Path(path_str),
        rain_var = str(state["rain_var"]),
        start = pd.Timestamp(state["start"]),
        end = pd.Timestamp(state["end"]),
        extent = state["extent"],
        grid = state["grid"],
        lat_dim = str(state["lat_dim"]),
        lon_dim = str(state["lon_dim"]),
    )
    return idx, da


def _prepare_et_file_component(
    path: Path,
    var_name: str,
    out_name: str,
    long_name: str,
    units: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    grid: TargetGrid,
    extent: Optional[Tuple[float, float, float, float]],
    lat_dim: str,
    lon_dim: str,
    dtype: str,
) -> xr.DataArray:
    expected_dates = pd.date_range(start = start_date, end = end_date, freq = "D")
    with xr.open_dataset(path) as ds:
        if var_name not in ds:
            raise KeyError(f"Variable '{var_name}' not found in ET dataset: {path}")
        da = ds[var_name]
        if "time" not in da.coords:
            raise ValueError(f"ET variable '{var_name}' must include a time coordinate.")
        da = _require_daily_coverage(da, expected_dates, f"ET variable '{var_name}'")
        if extent:
            if da.rio.crs is not None:
                da = _clip_to_extent(da, grid, extent)
            else:
                da = _subset_to_extent(da, lat_dim, lon_dim, extent)
        da = da.load()

    broadcast = _broadcast_single_pixel(da, grid)
    if broadcast is not None:
        da = broadcast
    else:
        da = _reproject_to_template(da, grid, resampling = Resampling.bilinear)
    da = da.astype(dtype)
    da.name = out_name
    da.attrs.update({"long_name": long_name, "units": units})
    return da


def _prepare_et_daily_layer(
    path: Path,
    date: pd.Timestamp,
    grid: TargetGrid,
    band_name: str,
    dtype: str,
) -> xr.DataArray:
    da_full = rioxarray.open_rasterio(path, masked = True)
    if da_full.rio.crs is None:
        da_full = da_full.rio.write_crs(grid.crs)

    descriptions = getattr(da_full.rio, "descriptions", None) or ()
    band_index = None
    if band_name in descriptions:
        band_index = descriptions.index(band_name)
    elif "band" in da_full.coords:
        band_labels = [str(val) for val in da_full.coords["band"].values.tolist()]
        if band_name in band_labels:
            band_index = band_labels.index(band_name)

    if band_index is None:
        if "band" in da_full.coords and da_full.coords["band"].size > 0:
            warnings.warn(
                f"{band_name} band not found in {path}; defaulting to first band.",
                RuntimeWarning,
                stacklevel = 2,
            )
            band_index = 0
        else:
            raise ValueError(f"{band_name} band not found in {path}")

    da = da_full.isel(band = band_index)
    if "band" in da.dims:
        da = da.squeeze("band", drop = True)
    da = _crop_raster_to_grid(da, grid, buffer_pixels = 2)
    da = da.astype("float64") * 86400.0
    da = _reproject_to_template(da, grid, resampling = Resampling.bilinear)
    da = da.astype(dtype)
    da = da.assign_coords(time = date).expand_dims("time")
    da.attrs.update({"units": "mm day-1"})
    return da


def _init_et_worker(grid: TargetGrid, band_name: str, dtype: str) -> None:
    global _ET_WORKER_STATE
    _ET_WORKER_STATE = {
        "grid": grid,
        "band_name": band_name,
        "dtype": dtype,
    }


def _load_et_daily_task(task: Tuple[int, str, str]) -> Tuple[int, xr.DataArray]:
    idx, path_str, date_str = task
    state = _ET_WORKER_STATE
    da = _prepare_et_daily_layer(
        path = Path(path_str),
        date = pd.Timestamp(date_str),
        grid = state["grid"],
        band_name = str(state["band_name"]),
        dtype = str(state["dtype"]),
    )
    return idx, da


def process_precipitation(args: argparse.Namespace, grid: TargetGrid, start: pd.Timestamp, end: pd.Timestamp) -> xr.DataArray:
    expected_dates = pd.date_range(start = start, end = end, freq = "D")
    if args.rain_file:
        path = Path(args.rain_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Precipitation file not found: {path}")
        with xr.open_dataset(path) as ds:
            if args.rain_var not in ds:
                raise KeyError(f"Variable '{args.rain_var}' not found in precipitation dataset: {path}")
            rain = _require_daily_coverage(ds[args.rain_var], expected_dates, "Precipitation")
            if args.extent:
                extent = tuple(args.extent)
                if rain.rio.crs is not None:
                    rain = _clip_to_extent(rain, grid, extent)
                else:
                    rain = _subset_to_extent(rain, grid.lat_dim, grid.lon_dim, extent)
            rain = rain.load()
        if rain.sizes.get("time", 0) == 0:
            raise ValueError("No precipitation data found in the requested date range.")
        rain = _require_daily_coverage(rain, expected_dates, "Precipitation")
    else:
        month_pattern = args.rain_filename_pattern or "ANUClimate_v2-0_rain_daily_{year}{month:02d}.nc"
        rain_root = Path(args.rain_root).expanduser().resolve()
        if "{month" in month_pattern:
            month_paths = _generate_month_paths(rain_root, month_pattern, start, end)
        else:
            month_paths = _generate_year_paths(rain_root, month_pattern, start, end)

        data_arrays: List[xr.DataArray] = []
        total_paths = len(month_paths)
        extent = tuple(args.extent) if args.extent else None
        if args.workers > 1 and total_paths > 1:
            effective_workers = min(args.workers, total_paths)
            print(
                f"Rain processing with process workers: requested={args.workers}, "
                f"effective={effective_workers}, files={total_paths}",
                flush = True,
            )
            with ProcessPoolExecutor(
                max_workers = effective_workers,
                mp_context = _pool_context(),
                initializer = _init_rain_worker,
                initargs = (
                    grid,
                    args.rain_var,
                    start,
                    end,
                    extent,
                    args.lat_dim,
                    args.lon_dim,
                ),
            ) as executor:
                futures = [
                    executor.submit(_load_rain_file_task, (idx, str(path)))
                    for idx, path in enumerate(month_paths, start = 1)
                ]
                completed = 0
                for future in as_completed(futures):
                    _, da = future.result()
                    if da is not None:
                        data_arrays.append(da)
                    completed += 1
                    _print_progress("Rain files", completed, total_paths)
        else:
            for idx, path in enumerate(month_paths, start = 1):
                _print_progress("Rain files", idx, total_paths)
                da = _load_rain_file_for_window(
                    path = path,
                    rain_var = args.rain_var,
                    start = start,
                    end = end,
                    extent = extent,
                    grid = grid,
                    lat_dim = args.lat_dim,
                    lon_dim = args.lon_dim,
                )
                if da is not None:
                    data_arrays.append(da)

        if not data_arrays:
            raise ValueError("No precipitation data found in the requested date range.")
        rain = xr.concat(data_arrays, dim = "time").sortby("time")
        rain = _require_daily_coverage(rain, expected_dates, "Precipitation")

    broadcast = _broadcast_single_pixel(rain, grid)
    rain = broadcast if broadcast is not None else _reproject_to_template(rain, grid, resampling = Resampling.bilinear)
    rain = rain.astype(args.dtype)
    rain.name = "precipitation"
    rain.attrs.update({"long_name": "Daily precipitation", "units": "mm day-1"})
    return rain


def process_et(args: argparse.Namespace, grid: TargetGrid, dates: Sequence[pd.Timestamp]) -> Dict[str, xr.DataArray]:
    if args.et_file:
        path = Path(args.et_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"ET file not found: {path}")

        extent = tuple(args.extent) if args.extent else None
        tasks = [(args.t_var, "t", "Daily plant transpiration", "mm day-1")]
        if args.e_var:
            tasks.append((args.e_var, "e", "Daily soil evaporation", "mm day-1"))
        else:
            tasks.append((args.et_var, "et", "Daily evapotranspiration", "mm day-1"))
        tasks.append((args.ndvi_var, "ndvi", "Daily NDVI", "1"))

        components: Dict[str, xr.DataArray] = {}
        with xr.open_dataset(path) as ds:
            if args.t_var not in ds:
                raise KeyError(
                    f"Variable '{args.t_var}' not found in ET dataset: {path}. "
                    "Provide a transpiration variable generated from SSEBop (e.g., T)."
                )
            if args.e_var and args.e_var not in ds:
                raise KeyError(f"Variable '{args.e_var}' not found in ET dataset: {path}")
            if not args.e_var and args.et_var not in ds:
                raise KeyError(f"Variable '{args.et_var}' not found in ET dataset: {path}")
            has_ndvi = args.ndvi_var in ds

        active_tasks = [task for task in tasks if task[1] != "ndvi" or has_ndvi]
        if args.workers > 1 and len(active_tasks) > 1:
            effective_workers = min(args.workers, len(active_tasks))
            print(
                f"ET file processing with process workers: requested={args.workers}, "
                f"effective={effective_workers}, components={len(active_tasks)}",
                flush = True,
            )
            with ProcessPoolExecutor(max_workers = effective_workers, mp_context = _pool_context()) as executor:
                future_to_name = {
                    executor.submit(
                        _prepare_et_file_component,
                        path,
                        var_name,
                        out_name,
                        long_name,
                        units,
                        dates[0],
                        dates[-1],
                        grid,
                        extent,
                        args.lat_dim,
                        args.lon_dim,
                        args.dtype,
                    ): out_name
                    for var_name, out_name, long_name, units in active_tasks
                }
                for future in as_completed(future_to_name):
                    components[future_to_name[future]] = future.result()
        else:
            for var_name, out_name, long_name, units in active_tasks:
                components[out_name] = _prepare_et_file_component(
                    path = path,
                    var_name = var_name,
                    out_name = out_name,
                    long_name = long_name,
                    units = units,
                    start_date = dates[0],
                    end_date = dates[-1],
                    grid = grid,
                    extent = extent,
                    lat_dim = args.lat_dim,
                    lon_dim = args.lon_dim,
                    dtype = args.dtype,
                )

        if "e" in components:
            et = (components["e"] + components["t"]).astype(args.dtype)
            et.name = "et"
            et.attrs.update({"long_name": "Daily evapotranspiration", "units": "mm day-1", "formula": "et = e + t"})
            components["et"] = et
        return components

    et_root = Path(args.et_root).expanduser().resolve()
    et_pattern = args.et_filename_pattern or "GLDAS_2.2_ET_SM_{year:04d}-{month:02d}-{day:02d}.tif"
    et_paths = _generate_daily_paths(et_root, et_pattern, dates)
    total_days = len(dates)
    band_name = args.et_var or "Evap_tavg"

    layers: List[xr.DataArray] = []
    if args.workers > 1 and total_days > 1:
        effective_workers = min(args.workers, total_days)
        print(
            f"ET daily processing with process workers: requested={args.workers}, "
            f"effective={effective_workers}, days={total_days}",
            flush = True,
        )
        ordered_layers: List[Optional[xr.DataArray]] = [None] * total_days
        with ProcessPoolExecutor(
            max_workers = effective_workers,
            mp_context = _pool_context(),
            initializer = _init_et_worker,
            initargs = (grid, band_name, args.dtype),
        ) as executor:
            futures = [
                executor.submit(_load_et_daily_task, (idx, str(path), date.strftime("%Y-%m-%d")))
                for idx, (path, date) in enumerate(zip(et_paths, dates), start = 1)
            ]
            completed = 0
            for future in as_completed(futures):
                idx, da = future.result()
                ordered_layers[idx - 1] = da
                completed += 1
                _print_progress("ET days", completed, total_days)
        layers = [layer for layer in ordered_layers if layer is not None]
    else:
        for idx, (path, date) in enumerate(zip(et_paths, dates), start = 1):
            _print_progress("ET days", idx, total_days)
            layers.append(
                _prepare_et_daily_layer(
                    path = path,
                    date = date,
                    grid = grid,
                    band_name = band_name,
                    dtype = args.dtype,
                )
            )

    et = xr.concat(layers, dim = "time")
    et.name = "et"
    et.attrs.update({"long_name": "Daily evapotranspiration", "units": "mm day-1"})
    return {"et": et}


def _parse_gssm_band_date(name: str) -> pd.Timestamp:
    match = re.fullmatch(r"band_(\d{4})_(\d{2})_(\d{2})_classification", name)
    if match is None:
        raise ValueError(f"Unsupported GSSM band name: {name}")
    year, month, day = match.groups()
    return pd.Timestamp(f"{year}-{month}-{day}")


def _initialize_ee(gee_project: str) -> None:
    try:
        ee.Initialize(project = gee_project)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Earth Engine with project '{gee_project}'.") from exc


def _validate_reference_source(reference_source: str) -> None:
    if reference_source != "gssm1km":
        raise ValueError(f"Unsupported reference_source '{reference_source}'. Only 'gssm1km' is supported.")


def _matching_gssm_image_ids(indices: Sequence[str], year: int) -> List[str]:
    prefix = f"SM{year}"
    return sorted(index for index in indices if index.startswith(prefix))


def _rename_reference_ssm(da: xr.DataArray) -> xr.DataArray:
    renamed = da.rename("reference_ssm")
    renamed.attrs.update(
        {
            "long_name": "Reference surface soil moisture",
            "units": "m3 m-3",
            "source": "gssm1km",
        }
    )
    return renamed


def _region_json_from_extent(extent: Tuple[float, float, float, float]) -> Dict[str, object]:
    min_lon, min_lat, max_lon, max_lat = [float(value) for value in extent]
    return {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]],
        "geodesic": False,
        "evenOdd": True,
    }


def _open_downloaded_raster(raw_bytes: bytes, name: str) -> xr.DataArray:
    with tempfile.TemporaryDirectory(prefix = f"{name}_") as temp_dir:
        temp_path = Path(temp_dir) / f"{name}.tif"
        temp_path.write_bytes(raw_bytes)
        try:
            return rioxarray.open_rasterio(temp_path, masked = True).load()
        except Exception:
            if not zipfile.is_zipfile(temp_path):
                raise
            with zipfile.ZipFile(temp_path) as archive:
                tif_members = [member for member in archive.namelist() if member.lower().endswith(".tif")]
                if not tif_members:
                    raise RuntimeError(f"Earth Engine download for '{name}' did not contain a GeoTIFF.")
                extracted_path = Path(temp_dir) / Path(tif_members[0]).name
                extracted_path.write_bytes(archive.read(tif_members[0]))
            return rioxarray.open_rasterio(extracted_path, masked = True).load()


def _download_ee_multiband_image(
    image: ee.Image,
    extent: Tuple[float, float, float, float],
    name: str,
    scale_m: float,
) -> xr.DataArray:
    band_names = image.bandNames().getInfo()
    url = image.getDownloadURL(
        {
            "name": name,
            "region": _region_json_from_extent(extent),
            "crs": "EPSG:4326",
            "scale": scale_m,
            "filePerBand": False,
            "format": "GEO_TIFF",
        }
    )
    with requests.get(url, timeout = 300) as response:
        response.raise_for_status()
        da = _open_downloaded_raster(response.content, name)
    if "band" in da.dims:
        da = da.assign_coords(band = np.array(band_names, dtype = object))
    da.name = name
    return da


def _load_reference_ssm(
    *,
    extent: Tuple[float, float, float, float],
    dates: Sequence[pd.Timestamp],
    reference_ssm_asset: str,
    gee_project: str,
) -> xr.DataArray:
    _initialize_ee(gee_project)
    region = ee.Geometry.Rectangle(list(extent), proj = "EPSG:4326", geodesic = False)
    collection = ee.ImageCollection(reference_ssm_asset)
    all_indices = collection.aggregate_array("system:index").getInfo()

    year_stacks = []
    for year in sorted({date.year for date in dates}):
        year_indices = _matching_gssm_image_ids(all_indices, year)
        if not year_indices:
            raise ValueError(f"No gssm1km image found for year {year}.")

        year_images = [
            ee.Image(collection.filter(ee.Filter.eq("system:index", image_id)).first()).clip(region)
            for image_id in year_indices
        ]
        year_image = ee.ImageCollection(year_images).mosaic()
        selected = []
        for band_name in year_image.bandNames().getInfo():
            band_date = _parse_gssm_band_date(band_name)
            if dates[0] <= band_date <= dates[-1]:
                selected.append((band_name, band_date))

        if not selected:
            continue

        raw_year = _download_ee_multiband_image(
            year_image.select([band_name for band_name, _ in selected]),
            extent,
            f"gssm_raw_{year}",
            GSSM_EXPORT_SCALE_M,
        )
        raw_year = raw_year.rename({"band": "time"}).assign_coords(time = [band_date for _, band_date in selected])
        year_stacks.append(raw_year)

    if not year_stacks:
        raise ValueError("Requested dates do not overlap any gssm1km daily bands.")

    raw = xr.concat(year_stacks, dim = "time").sortby("time")
    raw = _require_daily_coverage(raw, dates, "gssm1km reference SSM")
    return _rename_reference_ssm(raw / GSSM_SCALE_FACTOR)


def write_dataarray(da: xr.DataArray, output_path: Path):
    da = da.copy()
    da.attrs.pop("_FillValue", None)
    output_path.parent.mkdir(parents = True, exist_ok = True)
    encoding = {da.name: _standard_encoding(str(da.dtype))}
    da.to_dataset(name = da.name).to_netcdf(output_path, encoding = encoding)


def _write_preprocess_outputs(
    output_dir: Path,
    outputs: Dict[str, xr.DataArray],
    soil_arrays: Dict[str, xr.DataArray],
) -> Dict[str, Path]:
    written: Dict[str, Path] = {}
    for filename, da in outputs.items():
        path = output_dir / filename
        write_dataarray(da, path)
        written[filename] = path

    for name, da in soil_arrays.items():
        filename = f"soil_{name}.nc"
        path = output_dir / filename
        write_dataarray(da, path)
        written[filename] = path
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = "Preprocess spatial datasets into aligned SWB NetCDF files.")
    parser.add_argument("--start-date", type = str, help = "Simulation start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type = str, help = "Simulation end date (YYYY-MM-DD).")
    parser.add_argument("--date-range", nargs = 2, metavar = ("START", "END"), help = "Shortcut for start/end date.")
    parser.add_argument(
        "--extent",
        nargs = 4,
        type = float,
        metavar = ("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help = "Spatial extent.",
    )
    parser.add_argument("--sm-res", type = float, required = True, help = "Target spatial resolution (degrees).")
    parser.add_argument("--crs", default = "EPSG:4326", help = "Target CRS.")
    parser.add_argument("--lat-dim", default = "lat", help = "Latitude dimension name.")
    parser.add_argument("--lon-dim", default = "lon", help = "Longitude dimension name.")
    parser.add_argument("--dtype", choices = ("float32", "float64"), default = "float32", help = "Storage dtype.")
    parser.add_argument("--workers", type = int, default = 1, help = "Number of worker processes for rain/ET preprocessing.")

    parser.add_argument("--rain-file", help = "Single NetCDF file containing daily precipitation.")
    parser.add_argument(
        "--rain-root",
        default = "/g/data/gh70/ANUClimate/v2-0/stable/day/rain",
        help = "Base directory for precipitation NetCDFs.",
    )
    parser.add_argument("--rain-var", default = "rain", help = "Variable name in precipitation files.")
    parser.add_argument("--rain-filename-pattern", help = "Filename pattern with {year} and {month} placeholders.")

    parser.add_argument("--et-file", help = "Single NetCDF file containing daily ET and transpiration inputs.")
    parser.add_argument(
        "--et-root",
        default = "/g/data/yx97/GEE_collections/NASA/GLDAS/daily",
        help = "Directory containing daily ET GeoTIFFs.",
    )
    parser.add_argument("--e-var", help = "Variable name for daily soil evaporation in --et-file NetCDF.")
    parser.add_argument("--et-var", default = "Evap_tavg", help = "Band or variable name for evapotranspiration.")
    parser.add_argument("--t-var", default = "T", help = "Variable name for daily transpiration in --et-file NetCDF.")
    parser.add_argument(
        "--ndvi-var",
        default = "ndvi_interp",
        help = "Variable name for daily NDVI in --et-file NetCDF.",
    )
    parser.add_argument(
        "--tc-var",
        default = "Tc",
        help = "Deprecated compatibility argument; transpiration coefficient is no longer generated.",
    )
    parser.add_argument("--et-filename-pattern", help = "ET filename pattern with {year}, {month}, {day} placeholders.")

    parser.add_argument(
        "--output-dir",
        default = "/g/data/yx97/users_unikey/yiyu0116/sweb_model/2_spatial_preprocess",
        help = "Directory for processed outputs.",
    )
    parser.add_argument(
        "--soil-source",
        default = "openlandmap",
        help = (
            "Soil source backend. Supported values: openlandmap, mlcons, slga, custom. "
            "Implemented: openlandmap; placeholders: mlcons, slga, custom."
        ),
    )
    parser.add_argument("--reference-source", default = "gssm1km", help = "Reference SSM source. Only 'gssm1km' is supported.")
    parser.add_argument(
        "--reference-ssm-asset",
        default = "users/qianrswaterr/GlobalSSM1km0509",
        help = "Earth Engine ImageCollection asset for the reference SSM source.",
    )
    parser.add_argument("--gee-project", default = "yiyu-research", help = "Google Earth Engine project for preprocessing.")
    parser.add_argument("--skip-reference-ssm", action = "store_true", help = "Skip reference SSM preprocessing.")
    return parser


def _build_args(kwargs: Dict[str, object]) -> argparse.Namespace:
    parser = build_parser()
    defaults: Dict[str, object] = {}
    valid_names = set()
    for action in parser._actions:
        if action.dest == "help":
            continue
        valid_names.add(action.dest)
        defaults[action.dest] = None if action.default is argparse.SUPPRESS else action.default

    unknown = sorted(set(kwargs).difference(valid_names))
    if unknown:
        raise TypeError(f"Unexpected preprocess_inputs arguments: {', '.join(unknown)}")

    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def preprocess_inputs(**kwargs):
    args = _build_args(kwargs)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")

    soil_api.validate_soil_source(args.soil_source)
    _validate_reference_source(args.reference_source)
    start, end = _ensure_date_inputs(args)
    dates = pd.date_range(start = start, end = end, freq = "D")
    grid = _build_target_grid(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents = True, exist_ok = True)

    rain = process_precipitation(args, grid, start, end)
    effective_precip = compute_effective_precipitation_smith(rain, args.dtype)
    et_components = process_et(args, grid, dates)

    soil_outputs = soil_api.load_soil_properties(
        soil_source = args.soil_source,
        args = args,
        grid = grid,
        reproject_to_template = _reproject_to_template,
    )
    soil_arrays = soil_outputs.arrays

    outputs: Dict[str, xr.DataArray] = {
        f"rain_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc": rain,
        f"effective_precip_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc": effective_precip,
        f"et_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc": et_components["et"],
    }
    if "t" in et_components:
        outputs[f"t_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc"] = et_components["t"]
    if "e" in et_components:
        outputs[f"e_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc"] = et_components["e"]
    if "ndvi" in et_components:
        outputs[f"ndvi_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc"] = et_components["ndvi"]
    if not args.skip_reference_ssm:
        reference_ssm = _load_reference_ssm(
            extent = tuple(args.extent),
            dates = dates,
            reference_ssm_asset = args.reference_ssm_asset,
            gee_project = args.gee_project,
        )
        reference_ssm = _reproject_to_template(reference_ssm, grid, resampling = Resampling.bilinear)
        outputs[f"reference_ssm_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc"] = reference_ssm

    return _write_preprocess_outputs(output_dir, outputs, soil_arrays)


def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    return preprocess_inputs(**vars(args))


class _CallablePreprocessModule(types.ModuleType):
    def __call__(self, **kwargs):
        return preprocess_inputs(**kwargs)


sys.modules[__name__].__class__ = _CallablePreprocessModule


if __name__ == "__main__":
    main()
