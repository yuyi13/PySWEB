#!/usr/bin/env python3
"""
Preprocess spatial forcing and soil datasets into aligned NetCDF products.

This script harmonises precipitation (monthly NetCDF), evapotranspiration (daily GeoTIFF),
and soil hydraulic property rasters (GeoTIFF) onto a common grid, extent, and time range.
All processed outputs are written as NetCDF files that can be consumed by the spatial
soil water balance driver.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from pyproj import Transformer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

_DEFAULT_SOIL_DEPTHS = ("000", "005", "015", "030", "060", "100")
_RAIN_WORKER_STATE: Dict[str, object] = {}
_ET_WORKER_STATE: Dict[str, object] = {}
_SMAP_WORKER_STATE: Dict[str, object] = {}


@dataclass(frozen=True)
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


def _subset_to_extent(
    da: xr.DataArray, lat_dim: str, lon_dim: str, extent: Tuple[float, float, float, float]
) -> xr.DataArray:
    min_lon, min_lat, max_lon, max_lat = extent
    lat_values = da.coords[lat_dim].values
    lon_values = da.coords[lon_dim].values

    lat_slice = slice(max_lat, min_lat) if lat_values[0] > lat_values[-1] else slice(min_lat, max_lat)
    lon_slice = slice(min_lon, max_lon)

    subset = da.sel({lat_dim: lat_slice, lon_dim: lon_slice})
    if subset.sizes[lat_dim] == 0 or subset.sizes[lon_dim] == 0:
        raise ValueError("Extent selection removed all grid cells; adjust --extent bounds.")
    return subset


def _transform_extent(
    extent: Tuple[float, float, float, float], src_crs: str, dst_crs: str
) -> Tuple[float, float, float, float]:
    if src_crs == dst_crs:
        return extent
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
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
        minx=minx - buffer_x,
        miny=miny - buffer_y,
        maxx=maxx + buffer_x,
        maxy=maxy + buffer_y,
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
    lon_max_adj = min_lon + lon_steps * res
    if not np.isclose(lon_max_adj, max_lon, atol=res * 1e-4):
        if lon_max_adj > max_lon + res * 1e-4:
            raise ValueError("Longitude extent is not aligned with requested resolution.")
        max_lon = lon_max_adj
    target_lon = min_lon + np.arange(lon_steps + 1, dtype=float) * res

    lat_steps = int(round((max_lat - min_lat) / res))
    lat_max_adj = min_lat + lat_steps * res
    if not np.isclose(lat_max_adj, max_lat, atol=res * 1e-4):
        if lat_max_adj > max_lat + res * 1e-4:
            raise ValueError("Latitude extent is not aligned with requested resolution.")
        max_lat = lat_max_adj
    target_lat = (min_lat + np.arange(lat_steps + 1, dtype=float) * res)[::-1]

    return np.round(target_lat, 6), np.round(target_lon, 6)


def _compute_transform(latitudes: np.ndarray, longitudes: np.ndarray):
    lat_arr = np.asarray(latitudes, dtype=float)
    lon_arr = np.asarray(longitudes, dtype=float)
    if lat_arr.size < 2 or lon_arr.size < 2:
        raise ValueError("At least two coordinate values are required to infer transform.")

    lat_sorted = np.sort(lat_arr)
    lon_sorted = np.sort(lon_arr)

    res_lat = float(np.abs(np.diff(lat_sorted).mean()))
    res_lon = float(np.abs(np.diff(lon_sorted).mean()))
    if res_lat == 0.0 or res_lon == 0.0:
        raise ValueError("Detected zero spatial resolution in forcing data.")

    north = float(lat_arr.max())
    west = float(lon_arr.min())
    return from_origin(west, north, res_lon, res_lat)


def _grid_resolution(latitudes: np.ndarray, longitudes: np.ndarray) -> Tuple[float, float]:
    lat_arr = np.asarray(latitudes, dtype=float)
    lon_arr = np.asarray(longitudes, dtype=float)
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
    transform = from_origin(float(longitudes[0]), float(latitudes[0]), res_lon, res_lat)
    template = xr.DataArray(
        np.zeros((latitudes.size, longitudes.size), dtype=np.uint8),
        coords={lat_dim: latitudes, lon_dim: longitudes},
        dims=(lat_dim, lon_dim),
        name="template_mask",
    )
    template = template.rio.write_crs(crs)
    template = template.rio.write_transform(transform)
    template = template.rio.set_spatial_dims(x_dim=lon_dim, y_dim=lat_dim)
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
        minx=minx - buffer_x,
        miny=miny - buffer_y,
        maxx=maxx + buffer_x,
        maxy=maxy + buffer_y,
    )


def _reproject_to_template(data: xr.DataArray, grid: TargetGrid, resampling: Resampling = Resampling.bilinear) -> xr.DataArray:
    if grid.lat_dim in data.dims and grid.lon_dim in data.dims:
        lat_name = grid.lat_dim
        lon_name = grid.lon_dim
    elif "y" in data.dims and "x" in data.dims:
        lat_name = "y"
        lon_name = "x"
    else:
        raise ValueError("DataArray must have latitude/longitude or y/x dimensions for reprojection.")

    data = data.sortby(lat_name, ascending=False)
    lat_vals = data.coords[lat_name].values
    lon_vals = data.coords[lon_name].values
    if lat_vals.size < 2 or lon_vals.size < 2:
        res_lon, res_lat = _grid_resolution(grid.latitudes, grid.longitudes)
        transform = from_origin(float(lon_vals.min()), float(lat_vals.max()), res_lon, res_lat)
    else:
        transform = _compute_transform(lat_vals, lon_vals)
    if data.rio.crs is None:
        data = data.rio.write_crs(grid.crs)
    data = data.rio.write_transform(transform)
    data = data.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name)
    data = data.rio.reproject_match(grid.template, resampling=resampling)
    rename_map = {}
    if "y" in data.dims:
        rename_map["y"] = grid.lat_dim
    if "x" in data.dims:
        rename_map["x"] = grid.lon_dim
    if rename_map:
        data = data.rename(rename_map)
    data = data.assign_coords(
        {grid.lat_dim: grid.template.coords[grid.lat_dim].values, grid.lon_dim: grid.template.coords[grid.lon_dim].values}
    )
    return data


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
        values = np.asarray(single.values, dtype=float)
        out = xr.DataArray(
            np.broadcast_to(
                values[:, None, None],
                (values.shape[0], grid.latitudes.size, grid.longitudes.size),
            ),
            dims=("time", grid.lat_dim, grid.lon_dim),
            coords={
                "time": single.coords["time"].values,
                grid.lat_dim: grid.latitudes,
                grid.lon_dim: grid.longitudes,
            },
            name=data.name,
            attrs=data.attrs,
        )
    else:
        value = float(single.values)
        out = xr.DataArray(
            np.full((grid.latitudes.size, grid.longitudes.size), value, dtype=float),
            dims=(grid.lat_dim, grid.lon_dim),
            coords={grid.lat_dim: grid.latitudes, grid.lon_dim: grid.longitudes},
            name=data.name,
            attrs=data.attrs,
        )
    return out


def _generate_month_paths(root: Path, pattern: str, start: pd.Timestamp, end: pd.Timestamp) -> List[Path]:
    months = pd.period_range(start=start, end=end, freq="M")
    paths = []
    for period in months:
        year = period.year
        month = period.month
        path = root / str(year) / pattern.format(year=year, month=month)
        if not path.exists():
            raise FileNotFoundError(f"Missing precipitation file: {path}")
        paths.append(path)
    return paths


def _generate_year_paths(root: Path, pattern: str, start: pd.Timestamp, end: pd.Timestamp) -> List[Path]:
    years = range(start.year, end.year + 1)
    paths = []
    for year in years:
        path = root / pattern.format(year=year)
        if not path.exists():
            raise FileNotFoundError(f"Missing precipitation file: {path}")
        paths.append(path)
    return paths


def _generate_daily_paths(root: Path, pattern: str, dates: Sequence[pd.Timestamp]) -> List[Path]:
    paths = []
    missing = []
    for date in dates:
        path = root / pattern.format(year=date.year, month=date.month, day=date.day, date=date)
        if not path.exists():
            missing.append(str(path))
        else:
            paths.append(path)
    if missing:
        raise FileNotFoundError(f"Missing ET raster(s): {', '.join(missing[:5])}" + (" ..." if len(missing) > 5 else ""))
    return paths


def _ensure_date_inputs(args: argparse.Namespace) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if args.date_range:
        if args.start_date or args.end_date:
            raise ValueError("--date-range cannot be combined with --start-date or --end-date.")
        start_str, end_str = args.date_range
        args.start_date = start_str
        args.end_date = end_str
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
    if res <= 0:
        raise ValueError("--sm-res must be positive.")

    target_lat, target_lon = _build_target_coordinates(min_lon, max_lon, min_lat, max_lat, res)
    template = _prepare_template(target_lat, target_lon, res, res, args.lat_dim, args.lon_dim, args.crs)
    return TargetGrid(
        template=template,
        latitudes=template.coords[args.lat_dim].values,
        longitudes=template.coords[args.lon_dim].values,
        lat_dim=args.lat_dim,
        lon_dim=args.lon_dim,
        crs=args.crs,
    )


def _standard_encoding(dtype: str) -> Dict[str, Dict[str, object]]:
    storage_dtype = np.dtype(dtype)
    fill_value = storage_dtype.type(np.nan)
    return {
        "zlib": True,
        "complevel": 4,
        "dtype": storage_dtype.name,
        "_FillValue": fill_value,
    }


def _print_progress(label: str, index: int, total: int) -> None:
    if total <= 0:
        return
    bar_len = 20
    filled = int(bar_len * index / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    end_char = "\n" if index == total else "\r"
    print(f"{label} [{bar}] {index}/{total}", end=end_char, flush=True)


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
    with xr.open_dataset(path) as ds:
        if rain_var not in ds:
            raise KeyError(f"Variable '{rain_var}' not found in precipitation dataset: {path}")
        da = ds[rain_var]
        da = da.sel(time=slice(start, end))
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
        path=Path(path_str),
        rain_var=str(state["rain_var"]),
        start=pd.Timestamp(state["start"]),
        end=pd.Timestamp(state["end"]),
        extent=state["extent"],
        grid=state["grid"],
        lat_dim=str(state["lat_dim"]),
        lon_dim=str(state["lon_dim"]),
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
    with xr.open_dataset(path) as ds:
        if var_name not in ds:
            raise KeyError(f"Variable '{var_name}' not found in ET dataset: {path}")
        da = ds[var_name]
        if "time" in da.coords:
            time_min = pd.to_datetime(da.coords["time"].values).min()
            time_max = pd.to_datetime(da.coords["time"].values).max()
            if start_date < time_min or end_date > time_max:
                raise ValueError(
                    f"ET file {path} does not cover requested date range "
                    f"({start_date.date()} to {end_date.date()}); "
                    f"available range is {time_min.date()} to {time_max.date()}."
                )
            da = da.sel(time=slice(start_date, end_date))
        if extent:
            if da.rio.crs is not None:
                da = _clip_to_extent(da, grid, extent)
            else:
                da = _subset_to_extent(da, lat_dim, lon_dim, extent)
        da = da.load()
    da = _reproject_to_template(da, grid, resampling=Resampling.bilinear)
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
    da_full = rioxarray.open_rasterio(path, masked=True)
    if da_full.rio.crs is None:
        da_full = da_full.rio.write_crs(grid.crs)

    band_names = getattr(da_full.rio, "band_names", None)
    descriptions = getattr(da_full.rio, "descriptions", None)
    band_index = None
    if band_names and band_name in band_names:
        band_index = band_names.index(band_name)
    elif descriptions and band_name in descriptions:
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
                stacklevel=2,
            )
            band_index = 0
        else:
            raise ValueError(f"{band_name} band not found in {path}")

    da = da_full.isel(band=band_index)
    if "band" in da.dims:
        da = da.squeeze("band", drop=True)
    da = _crop_raster_to_grid(da, grid, buffer_pixels=2)
    da = da.astype("float64") * 86400.0  # convert kg m-2 s-1 to mm day-1
    da.attrs["units"] = "mm day-1"
    da = _reproject_to_template(da, grid, resampling=Resampling.bilinear)
    da = da.astype(dtype)
    da = da.assign_coords(time=date).expand_dims("time")
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
        path=Path(path_str),
        date=pd.Timestamp(date_str),
        grid=state["grid"],
        band_name=str(state["band_name"]),
        dtype=str(state["dtype"]),
    )
    return idx, da


def _init_smap_worker(
    grid: TargetGrid,
    smap_dir: Path,
    smap_pattern: str,
    dtype: str,
    smap_band: str,
) -> None:
    global _SMAP_WORKER_STATE
    _SMAP_WORKER_STATE = {
        "grid": grid,
        "smap_dir": smap_dir,
        "smap_pattern": smap_pattern,
        "dtype": dtype,
        "smap_band": smap_band,
    }


def _load_smap_day_task(task: Tuple[int, str]) -> Tuple[int, xr.DataArray, Optional[str]]:
    idx, date_str = task
    state = _SMAP_WORKER_STATE
    date = pd.Timestamp(date_str)
    grid = state["grid"]
    path = Path(state["smap_dir"]) / str(state["smap_pattern"]).format(date=date)
    missing_path = None
    if path.exists():
        da = _load_smap_raster(path, grid, str(state["dtype"]), str(state["smap_band"]))
    else:
        da = _blank_template_array(grid, str(state["dtype"]))
        missing_path = str(path)
    da = da.assign_coords(time=date).expand_dims("time")
    return idx, da, missing_path


def process_precipitation(args: argparse.Namespace, grid: TargetGrid, start: pd.Timestamp, end: pd.Timestamp) -> xr.DataArray:
    if args.rain_file:
        path = Path(args.rain_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Precipitation file not found: {path}")
        with xr.open_dataset(path) as ds:
            if args.rain_var not in ds:
                raise KeyError(f"Variable '{args.rain_var}' not found in precipitation dataset: {path}")
            da = ds[args.rain_var]
            if "time" in da.coords:
                da = da.sel(time=slice(start, end))
            if args.extent:
                if da.rio.crs is not None:
                    da = _clip_to_extent(da, grid, tuple(args.extent))
                else:
                    da = _subset_to_extent(da, grid.lat_dim, grid.lon_dim, tuple(args.extent))
            rain = da.load()
        if rain.sizes.get("time", 0) == 0:
            raise ValueError("No precipitation data found in the requested date range.")
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
                flush=True,
            )
            with ProcessPoolExecutor(
                max_workers=effective_workers,
                mp_context=_pool_context(),
                initializer=_init_rain_worker,
                initargs=(
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
                    for idx, path in enumerate(month_paths, start=1)
                ]
                completed = 0
                for future in as_completed(futures):
                    _, da = future.result()
                    if da is not None:
                        data_arrays.append(da)
                    completed += 1
                    _print_progress("Rain files", completed, total_paths)
        else:
            for idx, path in enumerate(month_paths, start=1):
                _print_progress("Rain files", idx, total_paths)
                da = _load_rain_file_for_window(
                    path=path,
                    rain_var=args.rain_var,
                    start=start,
                    end=end,
                    extent=extent,
                    grid=grid,
                    lat_dim=args.lat_dim,
                    lon_dim=args.lon_dim,
                )
                if da is not None:
                    data_arrays.append(da)

        if not data_arrays:
            raise ValueError("No precipitation data found in the requested date range.")

        rain = xr.concat(data_arrays, dim="time").sortby("time")

    broadcast = _broadcast_single_pixel(rain, grid)
    if broadcast is not None:
        rain = broadcast
    else:
        rain = _reproject_to_template(rain, grid, resampling=Resampling.bilinear)
    rain = rain.astype(args.dtype)
    rain.name = "precipitation"
    rain.attrs.update({"long_name": "Daily precipitation", "units": "mm day-1"})
    return rain


def process_et(args: argparse.Namespace, grid: TargetGrid, dates: Sequence[pd.Timestamp]) -> Dict[str, xr.DataArray]:
    if args.et_file:
        path = Path(args.et_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"ET file not found: {path}")
        print(f"ET file: {path}", flush=True)
        has_ndvi = False
        with xr.open_dataset(path) as ds:
            if args.t_var not in ds:
                raise KeyError(
                    f"Variable '{args.t_var}' not found in ET dataset: {path}. "
                    "Provide a transpiration variable generated from SSEBop (e.g., T)."
                )
            if args.e_var:
                if args.e_var not in ds:
                    raise KeyError(f"Variable '{args.e_var}' not found in ET dataset: {path}")
            elif args.et_var not in ds:
                raise KeyError(f"Variable '{args.et_var}' not found in ET dataset: {path}")
            has_ndvi = args.ndvi_var in ds
        if not has_ndvi:
            print(
                f"Warning: NDVI variable '{args.ndvi_var}' not found in ET dataset: {path}; "
                "skipping NDVI output (optional for downstream NDVI root-depth capping).",
                flush=True,
            )

        extent = tuple(args.extent) if args.extent else None
        tasks = [
            (args.t_var, "t", "Daily plant transpiration", "mm day-1"),
        ]
        if args.e_var:
            tasks.append((args.e_var, "e", "Daily soil evaporation", "mm day-1"))
        else:
            tasks.append((args.et_var, "et", "Daily evapotranspiration", "mm day-1"))
        if has_ndvi:
            tasks.append((args.ndvi_var, "ndvi", "Daily NDVI", "1"))

        components: Dict[str, xr.DataArray] = {}
        if args.workers > 1 and len(tasks) > 1:
            effective_workers = min(args.workers, len(tasks))
            print(
                f"ET file processing with process workers: requested={args.workers}, "
                f"effective={effective_workers}, components={len(tasks)}",
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=effective_workers, mp_context=_pool_context()) as executor:
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
                    for var_name, out_name, long_name, units in tasks
                }
                for future in as_completed(future_to_name):
                    out_name = future_to_name[future]
                    components[out_name] = future.result()
        else:
            for var_name, out_name, long_name, units in tasks:
                components[out_name] = _prepare_et_file_component(
                    path=path,
                    var_name=var_name,
                    out_name=out_name,
                    long_name=long_name,
                    units=units,
                    start_date=dates[0],
                    end_date=dates[-1],
                    grid=grid,
                    extent=extent,
                    lat_dim=args.lat_dim,
                    lon_dim=args.lon_dim,
                    dtype=args.dtype,
                )

        t = components["t"]
        if "e" in components:
            et = (components["e"] + t).astype(args.dtype)
            et.name = "et"
            et.attrs.update(
                {
                    "long_name": "Daily evapotranspiration",
                    "units": "mm day-1",
                    "formula": "et = e + t",
                }
            )
            components["et"] = et
        return components

    et_root = Path(args.et_root).expanduser().resolve()
    et_pattern = args.et_filename_pattern or "GLDAS_2.2_ET_SM_{year:04d}-{month:02d}-{day:02d}.tif"
    et_paths = _generate_daily_paths(et_root, et_pattern, dates)

    layers: List[xr.DataArray] = []
    total_days = len(dates)
    band_name = args.et_var or "Evap_tavg"
    if args.workers > 1 and total_days > 1:
        effective_workers = min(args.workers, total_days)
        print(
            f"ET daily processing with process workers: requested={args.workers}, "
            f"effective={effective_workers}, days={total_days}",
            flush=True,
        )
        ordered_layers: List[Optional[xr.DataArray]] = [None] * total_days
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=_pool_context(),
            initializer=_init_et_worker,
            initargs=(grid, band_name, args.dtype),
        ) as executor:
            futures = [
                executor.submit(_load_et_daily_task, (idx, str(path), date.strftime("%Y-%m-%d")))
                for idx, (path, date) in enumerate(zip(et_paths, dates), start=1)
            ]
            completed = 0
            for future in as_completed(futures):
                idx, da = future.result()
                ordered_layers[idx - 1] = da
                completed += 1
                _print_progress("ET days", completed, total_days)
        layers = [layer for layer in ordered_layers if layer is not None]
    else:
        for idx, (path, date) in enumerate(zip(et_paths, dates), start=1):
            _print_progress("ET days", idx, total_days)
            da = _prepare_et_daily_layer(
                path=path,
                date=date,
                grid=grid,
                band_name=band_name,
                dtype=args.dtype,
            )
            layers.append(da)

    et = xr.concat(layers, dim="time")
    et.name = "et"
    et.attrs.update({"long_name": "Daily evapotranspiration", "units": "mm day-1"})
    return {"et": et}


def _load_soil_raster(path: Path, grid: TargetGrid) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Soil raster not found: {path}")
    da = rioxarray.open_rasterio(path, masked=True).squeeze(drop=True)
    da = _crop_raster_to_grid(da, grid, buffer_pixels=2)
    da = _reproject_to_template(da, grid, resampling=Resampling.bilinear)
    return np.asarray(da.values, dtype=float)


def process_soil_properties(args: argparse.Namespace, grid: TargetGrid) -> Dict[str, xr.DataArray]:
    texture_dir = Path(args.soil_texture_dir).expanduser().resolve()
    soc_dir = Path(args.soil_soc_dir).expanduser().resolve()
    if not texture_dir.exists():
        raise FileNotFoundError(f"Soil texture directory does not exist: {texture_dir}")
    if not soc_dir.exists():
        raise FileNotFoundError(f"Soil organic carbon directory does not exist: {soc_dir}")

    if args.soil_depths:
        depth_list = [f"{int(depth):03d}" for depth in args.soil_depths]
    else:
        depth_list = list(_DEFAULT_SOIL_DEPTHS)
    if len(depth_list) < 2:
        raise ValueError("At least two soil depth boundaries are required.")

    porosity_layers = []
    wilting_layers = []
    awc_layers = []
    b_layers = []
    ksat_layers = []
    layer_depth_mm = []

    total_layers = len(depth_list) - 1
    for idx, (top, bottom) in enumerate(zip(depth_list[:-1], depth_list[1:]), start=1):
        _print_progress("Soil layers", idx, total_layers)
        clay_path = texture_dir / f"CLY_{top}_{bottom}_EV_N_P_AU_TRN_N_20210902.tif"
        sand_path = texture_dir / f"SND_{top}_{bottom}_EV_N_P_AU_TRN_N_20210902.tif"
        soc_path = soc_dir / f"{top}-{bottom}cm" / f"SOC_{top}_{bottom}_EV_N_P_AU_NAT_N_20220727.tif"

        clay = _load_soil_raster(clay_path, grid)
        sand = _load_soil_raster(sand_path, grid)
        soc = _load_soil_raster(soc_path, grid)

        om = 1.72 * soc * 0.01

        theta_33t = (
            -0.251 * sand
            + 0.195 * clay
            + 0.011 * om
            + 0.006 * sand * om
            - 0.027 * clay * om
            + 0.452 * sand * clay
            + 0.299
        )
        theta_33 = theta_33t + (1.283 * theta_33t**2 - 0.374 * theta_33t - 0.015)

        theta_s_33t = (
            0.278 * sand
            + 0.034 * clay
            + 0.022 * om
            - 0.018 * sand * om
            - 0.027 * clay * om
            - 0.584 * sand * clay
            + 0.078
        )
        theta_s_33 = theta_s_33t + (0.636 * theta_s_33t - 0.107)

        theta_1500t = (
            -0.024 * sand
            + 0.487 * clay
            + 0.006 * om
            + 0.005 * sand * om
            - 0.013 * clay * om
            + 0.068 * sand * clay
            + 0.031
        )
        theta_1500 = theta_1500t + (0.14 * theta_1500t - 0.02)

        theta_s = theta_33 + theta_s_33 - 0.097 * sand + 0.043

        with np.errstate(divide="ignore", invalid="ignore"):
            b_coeff = (np.log(1500.0) - np.log(33.0)) / (np.log(theta_33) - np.log(theta_1500))
            lambda_coeff = 1.0 / b_coeff
            ksat = 1930.0 * np.power(theta_s - theta_33, 3.0 - lambda_coeff)

        wilting_point = theta_1500
        porosity = theta_s
        available_water = np.clip(theta_33 - wilting_point, a_min=0.0, a_max=None)

        porosity_layers.append(porosity)
        wilting_layers.append(wilting_point)
        awc_layers.append(available_water)
        b_layers.append(b_coeff)
        ksat_layers.append(ksat * 24.0)

        layer_depth_mm.append(float(int(bottom) * 10))

    porosity = np.stack(porosity_layers, axis=0)
    wilting = np.stack(wilting_layers, axis=0)
    awc = np.stack(awc_layers, axis=0)
    b_coeff = np.stack(b_layers, axis=0)
    ksat = np.stack(ksat_layers, axis=0)

    layer_depth_mm = np.asarray(layer_depth_mm, dtype=float)
    layer_ids = np.arange(1, layer_depth_mm.size + 1, dtype=int)

    def _to_da(values: np.ndarray, name: str, attrs: Dict[str, str]) -> xr.DataArray:
        da = xr.DataArray(
            values.astype(args.dtype, copy=False),
            dims=("layer", grid.lat_dim, grid.lon_dim),
            coords={
                "layer": layer_ids,
                grid.lat_dim: grid.latitudes,
                grid.lon_dim: grid.longitudes,
            },
            name=name,
            attrs=attrs,
        )
        return da

    soil_arrays: Dict[str, xr.DataArray] = {
        "porosity": _to_da(
            porosity,
            "porosity",
            {"long_name": "Soil porosity", "units": "m3 m-3"},
        ),
        "wilting_point": _to_da(
            wilting,
            "wilting_point",
            {"long_name": "Wilting point volumetric water content", "units": "m3 m-3"},
        ),
        "available_water_capacity": _to_da(
            awc,
            "available_water_capacity",
            {"long_name": "Available water capacity", "units": "m3 m-3"},
        ),
        "b_coefficient": _to_da(
            b_coeff,
            "b_coefficient",
            {"long_name": "Campbell b coefficient", "units": "dimensionless"},
        ),
        "conductivity_sat": _to_da(
            ksat,
            "conductivity_sat",
            {"long_name": "Saturated hydraulic conductivity", "units": "mm day-1"},
        ),
    }

    return soil_arrays


def _blank_template_array(grid: TargetGrid, dtype: str) -> xr.DataArray:
    blank = xr.DataArray(
        np.full((grid.latitudes.size, grid.longitudes.size), np.nan, dtype=dtype),
        coords={grid.lat_dim: grid.latitudes, grid.lon_dim: grid.longitudes},
        dims=(grid.lat_dim, grid.lon_dim),
        name="blank",
    )
    blank = blank.rio.write_crs(grid.crs)
    blank = blank.rio.write_transform(grid.template.rio.transform())
    blank = blank.rio.set_spatial_dims(x_dim=grid.lon_dim, y_dim=grid.lat_dim)
    return blank


def _select_smap_band(da: xr.DataArray, band_choice: str) -> xr.DataArray:
    if "band" not in da.dims:
        return da
    if band_choice == "mean":
        return da.mean(dim="band", skipna=True)
    try:
        band_index = int(band_choice)
    except ValueError as exc:
        raise ValueError(f"Invalid --smap-band value: {band_choice}") from exc
    if "band" in da.coords:
        if band_index in da.coords["band"]:
            return da.sel(band=band_index)
    return da.isel(band=band_index - 1)


def _load_smap_raster(path: Path, grid: TargetGrid, dtype: str, band_choice: str) -> xr.DataArray:
    if not path.exists():
        raise FileNotFoundError(f"SMAP raster not found: {path}")
    da = rioxarray.open_rasterio(path, masked=True).squeeze(drop=True)
    if "band" not in da.dims:
        da = _crop_raster_to_grid(da, grid, buffer_pixels=2)
        da = _reproject_to_template(da, grid, resampling=Resampling.bilinear)
        return da.astype(dtype)

    band_values = da.coords["band"].values.tolist()
    band_layers = []
    for band_value in band_values:
        band_da = da.sel(band=band_value)
        band_da = _crop_raster_to_grid(band_da, grid, buffer_pixels=2)
        band_da = _reproject_to_template(band_da, grid, resampling=Resampling.bilinear)
        band_layers.append(band_da)

    if band_choice == "mean":
        merged = xr.concat(band_layers, dim="band").mean(dim="band", skipna=True)
    else:
        try:
            band_index = int(band_choice)
        except ValueError as exc:
            raise ValueError(f"Invalid --smap-band value: {band_choice}") from exc
        if band_index in band_values:
            select_idx = band_values.index(band_index)
        else:
            select_idx = band_index - 1
        merged = band_layers[select_idx]

    return merged.astype(dtype)


def process_smap_ssm(args: argparse.Namespace, grid: TargetGrid, dates: Sequence[pd.Timestamp]) -> xr.DataArray:
    smap_dir = Path(args.smap_dir).expanduser().resolve()
    if not smap_dir.exists():
        raise FileNotFoundError(f"SMAP directory does not exist: {smap_dir}")

    missing: List[str] = []
    layers: List[xr.DataArray] = []

    total_days = len(dates)
    if args.workers > 1 and total_days > 1:
        effective_workers = min(args.workers, total_days)
        print(
            f"SMAP processing with process workers: requested={args.workers}, "
            f"effective={effective_workers}, days={total_days}",
            flush=True,
        )
        ordered_layers: List[Optional[xr.DataArray]] = [None] * total_days
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=_pool_context(),
            initializer=_init_smap_worker,
            initargs=(grid, smap_dir, args.smap_pattern, args.dtype, args.smap_band),
        ) as executor:
            futures = [
                executor.submit(_load_smap_day_task, (idx, date.strftime("%Y-%m-%d")))
                for idx, date in enumerate(dates, start=1)
            ]
            completed = 0
            for future in as_completed(futures):
                idx, da, missing_path = future.result()
                ordered_layers[idx - 1] = da
                if missing_path is not None:
                    missing.append(missing_path)
                completed += 1
                _print_progress("SMAP days", completed, total_days)
        layers = [layer for layer in ordered_layers if layer is not None]
    else:
        blank = _blank_template_array(grid, args.dtype)
        for idx, date in enumerate(dates, start=1):
            _print_progress("SMAP days", idx, total_days)
            path = smap_dir / args.smap_pattern.format(date=date)
            if not path.exists():
                missing.append(str(path))
                da = blank.copy()
            else:
                da = _load_smap_raster(path, grid, args.dtype, args.smap_band)
            da = da.assign_coords(time=date).expand_dims("time")
            layers.append(da)

    if missing:
        print(f"Warning: {len(missing)} SMAP files missing; missing dates filled with NaN.", flush=True)

    smap = xr.concat(layers, dim="time")
    smap.name = "smap_ssm"
    smap.attrs.update({"long_name": "SMAP-DS surface soil moisture", "units": "m3 m-3"})
    return smap


def write_dataarray(da: xr.DataArray, output_path: Path):
    da = da.copy()
    da.attrs.pop("_FillValue", None)
    encoding = {da.name: _standard_encoding(str(da.dtype))}
    da.to_dataset(name=da.name).to_netcdf(output_path, encoding=encoding)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess spatial datasets into aligned NetCDF files.")
    parser.add_argument("--start-date", type=str, help="Simulation start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, help="Simulation end date (YYYY-MM-DD).")
    parser.add_argument("--date-range", nargs=2, metavar=("START", "END"), help="Shortcut for start/end date.")
    parser.add_argument("--extent", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"), help="Spatial extent.")
    parser.add_argument("--sm-res", type=float, required=True, help="Target spatial resolution (square grid, degrees).")
    parser.add_argument("--crs", default="EPSG:4326", help="Target CRS.")
    parser.add_argument("--lat-dim", default="lat", help="Latitude dimension name.")
    parser.add_argument("--lon-dim", default="lon", help="Longitude dimension name.")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32", help="Storage dtype.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for rain/ET/SMAP preprocessing.")

    parser.add_argument("--rain-file", help="Single NetCDF file containing daily precipitation.")
    parser.add_argument("--rain-root", default="/g/data/gh70/ANUClimate/v2-0/stable/day/rain", help="Base directory for precipitation NetCDFs.")
    parser.add_argument("--rain-var", default="rain", help="Variable name in precipitation files.")
    parser.add_argument("--rain-filename-pattern", help="Filename pattern with {year} and {month} placeholders.")

    parser.add_argument("--et-file", help="Single NetCDF file containing daily ET and transpiration inputs.")
    parser.add_argument("--et-root", default="/g/data/yx97/GEE_collections/NASA/GLDAS/daily", help="Directory containing daily ET GeoTIFFs.")
    parser.add_argument("--e-var", help="Variable name for daily soil evaporation in --et-file NetCDF (if set, et=e+t).")
    parser.add_argument("--et-var", default="Evap_tavg", help="Band name to extract from ET GeoTIFFs.")
    parser.add_argument("--t-var", default="T", help="Variable name for daily transpiration in --et-file NetCDF.")
    parser.add_argument(
        "--ndvi-var",
        default="ndvi_interp",
        help=(
            "Variable name for daily NDVI in --et-file NetCDF "
            "(used only when downstream workflows enable NDVI root-depth capping)."
        ),
    )
    parser.add_argument("--tc-var", default="Tc", help="Deprecated compatibility argument; transpiration coefficient is no longer generated.")
    parser.add_argument("--et-filename-pattern", help="ET filename pattern with {year}, {month}, {day} placeholders.")

    parser.add_argument(
        "--soil-texture-dir",
        default="/g/data/yx97/EO_collections/TERN/SLGA/v2/SoilTexture",
        help="Directory holding SLGA soil texture rasters (CLY/SND).",
    )
    parser.add_argument(
        "--soil-soc-dir",
        default="/g/data/yx97/EO_collections/TERN/SLGA/v2/SoilOrganicCarbon",
        help="Directory holding SLGA soil organic carbon rasters (SOC).",
    )
    parser.add_argument(
        "--soil-depths",
        nargs="+",
        help="Ordered soil depth boundaries in cm (e.g. 000 005 015 030 060 100).",
    )

    parser.add_argument("--output-dir", default="/g/data/yx97/users_unikey/yiyu0116/sweb_model/2_spatial_preprocess", help="Directory for processed outputs.")
    parser.add_argument(
        "--smap-dir",
        default="/g/data/yx97/EO_collections/NASA/SMAP/SMAP-DS",
        help="Directory holding SMAP-DS GeoTIFFs.",
    )
    parser.add_argument(
        "--smap-pattern",
        default="NSIDC-0779_EASE2_G1km_SMAP_SM_DS_{date:%Y%m%d}.tif",
        help="SMAP-DS filename pattern.",
    )
    parser.add_argument(
        "--smap-band",
        default="mean",
        help="SMAP band selection: 1, 2, or 'mean' for dual-layer files.",
    )
    parser.add_argument("--skip-smap", action="store_true", help="Skip SMAP-DS preprocessing.")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")
    start, end = _ensure_date_inputs(args)
    dates = pd.date_range(start=start, end=end, freq="D")
    grid = _build_target_grid(args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Processing precipitation…", flush=True)
    rain = process_precipitation(args, grid, start, end)
    rain_output = output_dir / f"rain_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc"
    write_dataarray(rain, rain_output)
    print(f"Wrote {rain_output}", flush=True)

    print("Processing evapotranspiration…", flush=True)
    et_components = process_et(args, grid, dates)
    et = et_components["et"]
    et_output = output_dir / f"et_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc"
    write_dataarray(et, et_output)
    print(f"Wrote {et_output}", flush=True)
    if "e" in et_components:
        e_output = output_dir / f"e_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc"
        write_dataarray(et_components["e"], e_output)
        print(f"Wrote {e_output}", flush=True)
    if "t" in et_components:
        t_output = output_dir / f"t_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc"
        write_dataarray(et_components["t"], t_output)
        print(f"Wrote {t_output}", flush=True)
    if "ndvi" in et_components:
        ndvi_output = output_dir / f"ndvi_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc"
        write_dataarray(et_components["ndvi"], ndvi_output)
        print(f"Wrote {ndvi_output}", flush=True)

    print("Processing soil properties…", flush=True)
    soil_arrays = process_soil_properties(args, grid)
    for name, da in soil_arrays.items():
        soil_output = output_dir / f"soil_{name}.nc"
        write_dataarray(da, soil_output)
        print(f"Wrote {soil_output}", flush=True)

    if not args.skip_smap:
        print("Processing SMAP-DS surface soil moisture…", flush=True)
        smap = process_smap_ssm(args, grid, dates)
        smap_output = output_dir / f"smap_ssm_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc"
        write_dataarray(smap, smap_output)
        print(f"Wrote {smap_output}", flush=True)

    print("Preprocessing complete.", flush=True)


if __name__ == "__main__":
    main()
