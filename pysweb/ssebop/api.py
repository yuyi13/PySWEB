"""
Script: api.py
Objective: Provide package-owned SSEBop input-preparation and model-run APIs.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: API parameters, optional YAML config, local Landsat GeoTIFFs, meteorology NetCDFs, and DEM rasters.
Outputs: Prepared inputs plus SSEBop ET GeoTIFF and NetCDF products in the requested output directory.
Usage: Imported as `pysweb.ssebop.api`
Dependencies: numpy, pandas, xarray, rasterio, rioxarray, pyproj, scipy, pyyaml
"""
from __future__ import annotations

import glob
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import repeat
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import CRS, Transformer

from pysweb.met.era5land import download as era5land_download
from pysweb.met.era5land import stack as era5land_stack
from pysweb.met.paths import infer_met_var_from_path, resolve_met_input_paths
from pysweb.ssebop.core import (
    build_doy_climatology,
    compute_dt_daily,
    et_fraction_xr,
    tcold_fano_simple_xr,
)
from pysweb.ssebop.grid import reproject_match, reproject_match_crop_first
from pysweb.ssebop.inputs import landsat
from pysweb.ssebop.landcover import (
    load_worldcover_landcover,
    worldcover_masks,
)

_SCENE_WORKER_CONTEXT: Optional[Dict[str, object]] = None


def prepare_inputs(
    *,
    date_range: str,
    extent: list[float],
    met_source: str,
    landsat_dir: str,
    met_raw_dir: str,
    met_stack_dir: str,
    dem: str,
    gee_config: str,
) -> None:
    start_date, end_date = landsat.parse_date_range(date_range)

    if met_source != "era5land":
        raise NotImplementedError(f"Unsupported met_source: {met_source}")

    landsat.prepare_landsat_inputs(
        date_range = date_range,
        extent = extent,
        gee_config = gee_config,
        out_dir = landsat_dir,
    )

    era5land_download.download_era5land_daily(
        start_date = start_date,
        end_date = end_date,
        extent = extent,
        output_dir = met_raw_dir,
    )
    era5land_stack.stack_era5land_daily_inputs(
        raw_dir = met_raw_dir,
        dem = dem,
        start_date = start_date,
        end_date = end_date,
        output_dir = met_stack_dir,
    )


def list_landsat_files(landsat_dir: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(landsat_dir, pattern)))


def parse_date_from_filename(path: str) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(path))
    if not match:
        raise ValueError(f"Could not parse date from filename: {path}")
    return match.group(0)


def parse_date_range(date_range: str) -> Tuple[str, str]:
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", date_range)
    if len(dates) != 2:
        raise ValueError("date_range must include two dates in YYYY-MM-DD format.")
    return dates[0], dates[1]


def filter_landsat_files_by_date(
    landsat_files: Sequence[str],
    date_range: Optional[str],
) -> List[str]:
    if not date_range:
        return list(landsat_files)

    start_date, end_date = parse_date_range(date_range)
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    kept: List[str] = []

    for path in landsat_files:
        date_str = parse_date_from_filename(path)
        dt = datetime.fromisoformat(date_str)
        if start_dt <= dt <= end_dt:
            kept.append(path)

    return kept


def read_geotiff_bands(path: str) -> Dict[str, xr.DataArray]:
    da = rioxarray.open_rasterio(path, masked=True)
    with rasterio.open(path) as src:
        descriptions = list(src.descriptions)

    if not any(descriptions):
        descriptions = [f"band{i}" for i in range(1, da.sizes["band"] + 1)]

    bands = {}
    for index, name in enumerate(descriptions, start=1):
        bands[name] = da.sel(band=index).rename(name)

    return bands


def ensure_spatial_dims(da: xr.DataArray) -> xr.DataArray:
    if {"x", "y"}.issubset(set(da.dims)):
        return da
    if {"lon", "lat"}.issubset(set(da.dims)):
        da = da.rename({"lon": "x", "lat": "y"})
    return da


def _coerce_paths(path: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(path, (list, tuple)):
        return list(path)
    return [path]


def open_meteorology_da(
    path: Union[str, Sequence[str]],
    var: Optional[str],
    default_var: Optional[str] = None,
) -> xr.DataArray:
    paths = _coerce_paths(path)
    if len(paths) == 1:
        ds = xr.open_dataset(paths[0])
    else:
        try:
            ds = xr.open_mfdataset(paths, combine="by_coords")
        except ValueError:
            datasets = []
            for one_path in paths:
                ds_part = xr.open_dataset(one_path)
                ds_part = ds_part.copy()
                ds_part.attrs = {}
                datasets.append(ds_part)
            ds = xr.combine_by_coords(datasets, combine_attrs="override")

    if var is None:
        var = infer_met_var_from_path(
            paths[0],
            default_var = default_var if len(paths) == 1 else None,
        )

    if var:
        if var not in ds:
            raise ValueError(f"Variable '{var}' not found in {path}")
        da = ds[var]
    else:
        da = ds.to_array().squeeze("variable", drop=True)

    da = ensure_spatial_dims(da)
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326")
    return da


def scale_landsat_sr(band: xr.DataArray) -> xr.DataArray:
    return band * 0.0000275 - 0.2


def scale_landsat_st(band: xr.DataArray) -> xr.DataArray:
    return band * 0.00341802 + 149.0


def _slice_to_date_range(da: xr.DataArray, date_range: Optional[str]) -> xr.DataArray:
    if not date_range or "time" not in da.coords:
        return da

    start_date, end_date = parse_date_range(date_range)
    return da.sel(time=slice(start_date, end_date))


def build_lat_lon(template: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    transform = template.rio.transform()
    height, width = template.shape
    xs = transform.c + transform.a * (np.arange(width) + 0.5)
    ys = transform.f + transform.e * (np.arange(height) + 0.5)
    xv, yv = np.meshgrid(xs, ys)

    crs = template.rio.crs
    if crs is None:
        raise ValueError("Template raster has no CRS")

    if CRS.from_user_input(crs).to_epsg() == 4326:
        lon = xr.DataArray(xv, dims=("y", "x"))
        lat = xr.DataArray(yv, dims=("y", "x"))
        return lat, lon

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_vals, lat_vals = transformer.transform(xv, yv)
    lat = xr.DataArray(lat_vals, dims=("y", "x"))
    lon = xr.DataArray(lon_vals, dims=("y", "x"))
    return lat, lon


def assign_dim_coords_from_reference(
    data: xr.DataArray,
    reference: xr.DataArray,
    exclude_dims: Sequence[str] = (),
) -> xr.DataArray:
    excluded = set(exclude_dims)
    coords = {}

    for dim in data.dims:
        if dim in excluded or dim not in reference.dims or dim not in reference.coords:
            continue
        if data.sizes[dim] != reference.sizes[dim]:
            raise ValueError(
                f"Cannot assign coordinate '{dim}': size {data.sizes[dim]} does not match reference size {reference.sizes[dim]}"
            )
        coords[dim] = reference.coords[dim].copy()

    if not coords:
        return data

    return data.assign_coords(coords)


def load_output_stack(
    records: Sequence[Tuple[np.datetime64, str]],
    var_name: str,
    reference_grid: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    arrays: List[xr.DataArray] = []
    times: List[np.datetime64] = []

    for ts, path in sorted(records, key=lambda item: item[0]):
        da = rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True).rename(var_name)
        if reference_grid is not None:
            da = assign_dim_coords_from_reference(da, reference_grid)
        da = da.load()
        arrays.append(da.assign_coords(time=ts).expand_dims("time"))
        times.append(ts)

    return xr.concat(arrays, dim="time").assign_coords(time=times)


def interpolate_etf_daily(
    etf_stack: xr.DataArray,
    daily_time: xr.DataArray,
    max_gap_days: int,
) -> xr.DataArray:
    daily_time_ns = pd.to_datetime(daily_time.values, errors="coerce").values.astype("datetime64[ns]")
    if pd.isna(daily_time_ns).any():
        raise ValueError("daily_time contains non-convertible values")

    daily_time_dt = xr.DataArray(
        daily_time_ns,
        coords={"time": daily_time_ns},
        dims=("time",),
    )

    etf_time_ns = pd.to_datetime(etf_stack.time.values, errors="coerce").values.astype("datetime64[ns]")
    if pd.isna(etf_time_ns).any():
        raise ValueError("etf_stack.time contains non-convertible values")
    etf_stack = etf_stack.assign_coords(time=etf_time_ns)

    etf_daily = etf_stack.interp(time=daily_time_dt, method="linear")

    obs = np.array(etf_time_ns, dtype="datetime64[ns]")
    daily = np.array(daily_time_ns, dtype="datetime64[ns]")
    idx = np.searchsorted(obs, daily, side="right")
    prev = np.where(idx > 0, obs[idx - 1], np.datetime64("NaT"))
    idx_next = np.clip(idx, 0, len(obs) - 1)
    next_ = np.where(idx < len(obs), obs[idx_next], np.datetime64("NaT"))
    gap = (next_ - prev).astype("timedelta64[D]").astype("int64")
    gap[np.isnat(prev) | np.isnat(next_)] = max_gap_days + 1

    mask = xr.DataArray(
        gap <= max_gap_days,
        coords={"time": daily_time_dt.values},
        dims=("time",),
    )
    etf_daily = etf_daily.where(mask)
    etf_daily = etf_daily.transpose(*etf_stack.dims)
    etf_daily = assign_dim_coords_from_reference(etf_daily, etf_stack, exclude_dims=("time",))
    return etf_daily


def _gapfill_flat_chunk(
    flat_chunk: np.ndarray,
    times: np.ndarray,
    window_len: int,
    polyorder: int,
    min_samples: int,
) -> np.ndarray:
    from scipy.signal import savgol_filter

    out = np.array(flat_chunk, copy=True)

    for idx in range(out.shape[1]):
        series = out[:, idx]
        valid = np.isfinite(series)
        if valid.sum() < min_samples or valid.all():
            continue

        series_interp = pd.Series(series, index=times).interpolate(method="time", limit_direction="both")
        interp_values = series_interp.to_numpy()
        if not np.isfinite(interp_values).all():
            continue

        smooth = savgol_filter(interp_values, window_length=window_len, polyorder=polyorder, mode="interp")
        fill_mask = ~valid
        if fill_mask.any():
            series_filled = series.copy()
            series_filled[fill_mask] = smooth[fill_mask]
            out[:, idx] = series_filled

    return out


def gapfill_etf_savgol(
    etf_stack: xr.DataArray,
    window_days: int,
    min_samples: int,
    workers: int = 1,
) -> xr.DataArray:
    times = pd.to_datetime(etf_stack.time.values).to_numpy(dtype="datetime64[ns]")
    data = np.array(etf_stack.values, copy=True)

    if data.shape[0] < 3:
        return etf_stack

    deltas = np.diff(times).astype("timedelta64[D]").astype("float64")
    median_dt = np.nanmedian(deltas[deltas > 0]) if np.isfinite(deltas).any() else 1.0
    if not np.isfinite(median_dt) or median_dt <= 0:
        median_dt = 1.0

    window_len = int(np.floor(window_days / median_dt))
    if window_len < min_samples:
        window_len = min_samples
    if window_len % 2 == 0:
        window_len += 1
    if window_len > data.shape[0]:
        window_len = data.shape[0] if data.shape[0] % 2 == 1 else data.shape[0] - 1
    if window_len < 3:
        return etf_stack

    polyorder = min(2, window_len - 1)

    flat = data.reshape(data.shape[0], -1)
    missing_cols = np.any(~np.isfinite(flat), axis=0)
    if not missing_cols.any():
        return etf_stack

    target_cols = np.where(missing_cols)[0]
    target = flat[:, target_cols]

    if workers > 1 and target.shape[1] > 1:
        n_workers = min(workers, target.shape[1])
        chunk_size = int(np.ceil(target.shape[1] / n_workers))
        chunks = [target[:, i : i + chunk_size] for i in range(0, target.shape[1], chunk_size)]
        mp_ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as executor:
            filled_chunks = list(
                executor.map(
                    _gapfill_flat_chunk,
                    chunks,
                    repeat(times, len(chunks)),
                    repeat(window_len, len(chunks)),
                    repeat(polyorder, len(chunks)),
                    repeat(min_samples, len(chunks)),
                )
            )
        target_filled = np.concatenate(filled_chunks, axis=1)
    else:
        target_filled = _gapfill_flat_chunk(target, times, window_len, polyorder, min_samples)

    flat_out = np.array(flat, copy=True)
    flat_out[:, target_cols] = target_filled
    filled = flat_out.reshape(data.shape).astype(data.dtype, copy=False)
    return etf_stack.copy(data=filled)


def _process_landsat_scene_worker(tif_path: str) -> Tuple[np.datetime64, str, str]:
    if _SCENE_WORKER_CONTEXT is None:
        raise RuntimeError("Scene worker context is not initialised")

    return process_landsat_scene(
        tif_path = tif_path,
        lst_band = str(_SCENE_WORKER_CONTEXT["lst_band"]),
        ndvi_band = str(_SCENE_WORKER_CONTEXT["ndvi_band"]),
        red_band = str(_SCENE_WORKER_CONTEXT["red_band"]),
        nir_band = str(_SCENE_WORKER_CONTEXT["nir_band"]),
        apply_water_mask = bool(_SCENE_WORKER_CONTEXT["apply_water_mask"]),
        water_mask = _SCENE_WORKER_CONTEXT["water_mask"],  # type: ignore[arg-type]
        dt_clim = _SCENE_WORKER_CONTEXT["dt_clim"],  # type: ignore[arg-type]
        template_crs = _SCENE_WORKER_CONTEXT["template_crs"],  # type: ignore[arg-type]
        etf_dir = str(_SCENE_WORKER_CONTEXT["etf_dir"]),
        ndvi_dir = str(_SCENE_WORKER_CONTEXT["ndvi_dir"]),
    )


def process_landsat_scene(
    tif_path: str,
    lst_band: str,
    ndvi_band: str,
    red_band: str,
    nir_band: str,
    apply_water_mask: bool,
    water_mask: Optional[xr.DataArray],
    dt_clim: xr.DataArray,
    template_crs: Optional[CRS],
    etf_dir: str,
    ndvi_dir: str,
) -> Tuple[np.datetime64, str, str]:
    date_str = parse_date_from_filename(tif_path)
    doy = datetime.fromisoformat(date_str).timetuple().tm_yday
    bands = read_geotiff_bands(tif_path)

    if lst_band not in bands:
        raise ValueError(f"LST band '{lst_band}' not found in {tif_path}")
    lst = scale_landsat_st(bands[lst_band])

    if ndvi_band in bands:
        ndvi = bands[ndvi_band]
    else:
        if red_band not in bands or nir_band not in bands:
            raise ValueError(f"NDVI or red/nir bands missing in {tif_path}")
        red = scale_landsat_sr(bands[red_band])
        nir = scale_landsat_sr(bands[nir_band])
        ndvi = (nir - red) / (nir + red)

    if apply_water_mask and water_mask is not None:
        lst = lst.where(water_mask == 0)
        ndvi = ndvi.where(water_mask == 0)

    dt = reproject_match(dt_clim.sel(dayofyear=doy), lst, resampling="bilinear")
    tcold = tcold_fano_simple_xr(lst, ndvi, dt)
    etf = et_fraction_xr(lst, tcold, dt).rename("etf")
    ts = np.datetime64(date_str, "ns")
    etf = etf.assign_coords(time=ts).expand_dims("time")
    ndvi_out = ndvi.rename("ndvi").assign_coords(time=ts).expand_dims("time")

    for coord in ("dayofyear", "band"):
        if coord in etf.coords and coord not in etf.dims:
            etf = etf.drop_vars(coord)
        if coord in ndvi_out.coords and coord not in ndvi_out.dims:
            ndvi_out = ndvi_out.drop_vars(coord)

    out_etf = os.path.join(etf_dir, f"etf_{date_str}.tif")
    etf_single = etf.squeeze("time", drop=True)
    etf_single.rio.write_crs(template_crs, inplace=True)
    etf_single.rio.to_raster(out_etf)

    out_ndvi = os.path.join(ndvi_dir, f"ndvi_{date_str}.tif")
    ndvi_single = ndvi_out.squeeze("time", drop=True)
    ndvi_single.rio.write_crs(template_crs, inplace=True)
    ndvi_single.rio.to_raster(out_ndvi)

    return ts, out_etf, out_ndvi


def _load_run_config(config_path: Optional[str]) -> dict:
    if not config_path:
        return {}

    import yaml

    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _cfg_value(
    cfg: dict,
    args: dict,
    key: str,
    default = None,
    aliases: Sequence[str] = (),
):
    if cfg:
        for candidate in (key, *aliases):
            if candidate in cfg:
                return cfg[candidate]
    return args.get(key, default)


def run_ssebop_workflow(
    date_range: Optional[str] = None,
    landsat_dir: Optional[str] = None,
    met_dir: Optional[str] = None,
    dem: Optional[str] = None,
    output_dir: Optional[str] = None,
    **kwargs,
) -> None:
    config_path = (
        kwargs.pop("config", None)
        or kwargs.pop("config_pos", None)
        or kwargs.pop("config_path", None)
    )
    cfg = _load_run_config(config_path)
    args = {
        "date_range": date_range,
        "landsat_dir": landsat_dir,
        "met_dir": met_dir,
        "dem": dem,
        "output_dir": output_dir,
        **kwargs,
    }

    date_range = _cfg_value(cfg, args, "date_range")
    silo_dir = _cfg_value(cfg, args, "silo_dir")
    met_dir = _cfg_value(cfg, args, "met_dir")
    landsat_dir = _cfg_value(cfg, args, "landsat_dir")
    landsat_pattern = _cfg_value(cfg, args, "landsat_pattern", default="*.tif")
    lst_band = _cfg_value(cfg, args, "lst_band", default="lst")
    ndvi_band = _cfg_value(cfg, args, "ndvi_band", default="ndvi")
    red_band = _cfg_value(cfg, args, "red_band", default="red")
    nir_band = _cfg_value(cfg, args, "nir_band", default="nir")
    et_short_crop = _cfg_value(cfg, args, "et_short_crop")
    et_short_crop_var = _cfg_value(cfg, args, "et_short_crop_var")
    tmax_path = _cfg_value(cfg, args, "tmax")
    tmax_var = _cfg_value(cfg, args, "tmax_var")
    tmin_path = _cfg_value(cfg, args, "tmin")
    tmin_var = _cfg_value(cfg, args, "tmin_var")
    rs_path = _cfg_value(cfg, args, "rs")
    rs_var = _cfg_value(cfg, args, "rs_var")
    ea_path = _cfg_value(cfg, args, "ea")
    ea_var = _cfg_value(cfg, args, "ea_var")
    dem_path = _cfg_value(cfg, args, "dem")
    landcover_path = _cfg_value(cfg, args, "landcover")
    output_dir = _cfg_value(cfg, args, "output_dir")
    max_gap_days = int(_cfg_value(cfg, args, "max_gap_days", default=32))
    apply_water_mask = bool(_cfg_value(cfg, args, "apply_water_mask", default=False))
    met_temp_units = _cfg_value(
        cfg,
        args,
        "met_temp_units",
        default="celsius",
        aliases=("silo_temp_units",),
    )
    gapfill_etf = bool(_cfg_value(cfg, args, "gapfill_etf", default=False))
    gapfill_window_days = _cfg_value(cfg, args, "gapfill_window_days")
    gapfill_min_samples = int(_cfg_value(cfg, args, "gapfill_min_samples", default=5))
    workers = int(_cfg_value(cfg, args, "workers", default=1))

    if workers < 1:
        raise ValueError("workers must be >= 1")

    missing = [
        key
        for key, value in {
            "landsat_dir": landsat_dir,
            "dem": dem_path,
            "output_dir": output_dir,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")

    os.makedirs(output_dir, exist_ok=True)
    etf_dir = os.path.join(output_dir, "etf")
    ndvi_dir = os.path.join(output_dir, "ndvi")
    os.makedirs(etf_dir, exist_ok=True)
    os.makedirs(ndvi_dir, exist_ok=True)

    landsat_files = list_landsat_files(landsat_dir, landsat_pattern)
    landsat_files = filter_landsat_files_by_date(landsat_files, date_range)
    if not landsat_files:
        raise RuntimeError("No Landsat GeoTIFFs found.")

    sample_bands = read_geotiff_bands(landsat_files[0])
    template = next(iter(sample_bands.values()))
    buffer = 2 * float(template.rio.resolution()[0])

    if landcover_path:
        landcover = load_worldcover_landcover(landcover_path)
        landcover = reproject_match_crop_first(landcover, template, resampling="nearest", buffer=buffer)
        _, _, water_mask = worldcover_masks(landcover)
    else:
        water_mask = None

    dem_da = rioxarray.open_rasterio(dem_path, masked=True).squeeze("band", drop=True)
    dem_da = reproject_match_crop_first(dem_da, template, resampling="bilinear", buffer=buffer).load()

    met_paths = {
        "et_short_crop": resolve_met_input_paths("et_short_crop", et_short_crop, met_dir, silo_dir, date_range),
        "tmax": resolve_met_input_paths("tmax", tmax_path, met_dir, silo_dir, date_range),
        "tmin": resolve_met_input_paths("tmin", tmin_path, met_dir, silo_dir, date_range),
        "rs": resolve_met_input_paths("rs", rs_path, met_dir, silo_dir, date_range),
        "ea": resolve_met_input_paths("ea", ea_path, met_dir, silo_dir, date_range),
    }
    et_short_crop = met_paths["et_short_crop"]
    tmax_path = met_paths["tmax"]
    tmin_path = met_paths["tmin"]
    rs_path = met_paths["rs"]
    ea_path = met_paths["ea"]

    missing_paths = [key for key, value in met_paths.items() if not value]
    if missing_paths:
        raise ValueError(
            "Missing meteorology paths: "
            + ", ".join(missing_paths)
            + ". Provide explicit NetCDF paths, or use --met-dir with --date-range for ERA5-Land stacks, "
            + "or --silo-dir with --date-range for legacy SILO files."
        )

    tmax = reproject_match_crop_first(
        _slice_to_date_range(open_meteorology_da(tmax_path, tmax_var, default_var="tmax"), date_range),
        template,
        resampling="bilinear",
        buffer=buffer,
    ).load()
    tmin = reproject_match_crop_first(
        _slice_to_date_range(open_meteorology_da(tmin_path, tmin_var, default_var="tmin"), date_range),
        template,
        resampling="bilinear",
        buffer=buffer,
    ).load()
    rs = reproject_match_crop_first(
        _slice_to_date_range(open_meteorology_da(rs_path, rs_var, default_var="rs"), date_range),
        template,
        resampling="bilinear",
        buffer=buffer,
    ).load()
    ea = reproject_match_crop_first(
        _slice_to_date_range(open_meteorology_da(ea_path, ea_var, default_var="ea"), date_range),
        template,
        resampling="bilinear",
        buffer=buffer,
    ).load()

    if met_temp_units == "celsius":
        tmax = tmax + 273.15
        tmin = tmin + 273.15

    lat_deg, _ = build_lat_lon(template)
    dt_daily = compute_dt_daily(tmax, tmin, dem_da, lat_deg, rs_mj_m2_day=rs, ea_kpa=ea).load()
    dt_clim = build_doy_climatology(dt_daily).load()

    del dt_daily
    del tmax
    del tmin
    del rs
    del ea
    del dem_da
    del lat_deg

    eto = reproject_match_crop_first(
        _slice_to_date_range(
            open_meteorology_da(et_short_crop, et_short_crop_var, default_var="et_short_crop"),
            date_range,
        ),
        template,
        resampling="bilinear",
        buffer=buffer,
    ).load()

    scene_outputs: List[Tuple[np.datetime64, str, str]] = []
    template_crs = template.rio.crs
    if workers == 1:
        for tif_path in landsat_files:
            scene_outputs.append(
                process_landsat_scene(
                    tif_path = tif_path,
                    lst_band = lst_band,
                    ndvi_band = ndvi_band,
                    red_band = red_band,
                    nir_band = nir_band,
                    apply_water_mask = apply_water_mask,
                    water_mask = water_mask,
                    dt_clim = dt_clim,
                    template_crs = template_crs,
                    etf_dir = etf_dir,
                    ndvi_dir = ndvi_dir,
                )
            )
    else:
        scene_context: Dict[str, object] = {
            "lst_band": lst_band,
            "ndvi_band": ndvi_band,
            "red_band": red_band,
            "nir_band": nir_band,
            "apply_water_mask": apply_water_mask,
            "water_mask": water_mask,
            "dt_clim": dt_clim,
            "template_crs": template_crs,
            "etf_dir": etf_dir,
            "ndvi_dir": ndvi_dir,
        }

        global _SCENE_WORKER_CONTEXT
        _SCENE_WORKER_CONTEXT = scene_context
        try:
            mp_ctx = mp.get_context("fork")
            scene_workers = min(workers, len(landsat_files))
            scene_chunksize = max(1, len(landsat_files) // (scene_workers * 4))
            with ProcessPoolExecutor(max_workers=scene_workers, mp_context=mp_ctx) as executor:
                for output in executor.map(
                    _process_landsat_scene_worker,
                    landsat_files,
                    chunksize=scene_chunksize,
                ):
                    scene_outputs.append(output)
        finally:
            _SCENE_WORKER_CONTEXT = None

    etf_stack = load_output_stack(
        [(ts, etf_path_out) for ts, etf_path_out, _ in scene_outputs],
        "etf",
        reference_grid=template,
    )
    ndvi_stack = load_output_stack(
        [(ts, ndvi_path_out) for ts, _, ndvi_path_out in scene_outputs],
        "ndvi",
        reference_grid=template,
    )
    if gapfill_etf:
        window_days = gapfill_window_days if gapfill_window_days is not None else max_gap_days
        etf_stack = gapfill_etf_savgol(etf_stack, window_days, gapfill_min_samples, workers=workers)
        ndvi_stack = gapfill_etf_savgol(ndvi_stack, window_days, gapfill_min_samples, workers=workers)

    etf_start = pd.to_datetime(etf_stack.time.values).min()
    etf_end = pd.to_datetime(etf_stack.time.values).max()
    daily_index = pd.date_range(etf_start, etf_end, freq="D")
    daily_time = xr.DataArray(daily_index.values, coords={"time": daily_index.values}, dims=("time",))

    etf_daily = interpolate_etf_daily(etf_stack, daily_time, max_gap_days).rename("etf_interp")
    ndvi_daily = interpolate_etf_daily(ndvi_stack, daily_time, max_gap_days).rename("ndvi_interp")

    suffix = ""
    if date_range:
        start_date, end_date = parse_date_range(date_range)
        suffix = f"_{start_date}_{end_date}"

    etf_stack.to_netcdf(os.path.join(output_dir, f"etf_stack{suffix}.nc"))
    ndvi_stack.to_netcdf(os.path.join(output_dir, f"ndvi_stack{suffix}.nc"))
    del etf_stack
    del ndvi_stack

    eto = eto.sel(time=daily_index.values)
    et_daily = (etf_daily * eto).rename("ET")
    del eto
    tc_daily = (1.26 * ndvi_daily - 0.18).clip(0.0, 1.0).rename("Tc")
    t_daily = (et_daily * tc_daily).rename("T")
    e_daily = (et_daily - t_daily).rename("E")

    tc_daily.attrs.update(
        {
            "long_name": "Transpiration coefficient",
            "description": "Tc equals fractional green cover (fc) derived from NDVI",
            "formula": "fc = 1.26 * NDVI - 0.18; Tc = clip(fc, 0, 1)",
            "units": "1",
            "source_variable": "ndvi_interp",
        }
    )
    et_daily.attrs.update({"long_name": "Actual evapotranspiration", "units": "mm/day"})
    t_daily.attrs.update({"long_name": "Plant transpiration", "units": "mm/day", "formula": "T = ET * Tc"})
    e_daily.attrs.update({"long_name": "Soil evaporation", "units": "mm/day", "formula": "E = ET - T"})
    etf_daily.attrs.update({"long_name": "Interpolated SSEBop ET fraction", "units": "1"})
    ndvi_daily.attrs.update({"long_name": "Interpolated NDVI", "units": "1"})

    xr.Dataset(
        {
            "ET": et_daily,
            "E": e_daily,
            "T": t_daily,
            "etf_interp": etf_daily,
            "ndvi_interp": ndvi_daily,
            "Tc": tc_daily,
        }
    ).to_netcdf(os.path.join(output_dir, f"et_daily_ssebop{suffix}.nc"))


def run(
    *,
    date_range: Optional[str] = None,
    landsat_dir: Optional[str] = None,
    met_dir: Optional[str] = None,
    dem: Optional[str] = None,
    output_dir: Optional[str] = None,
    **kwargs,
) -> None:
    workflow_kwargs = dict(kwargs)
    if date_range is not None:
        workflow_kwargs["date_range"] = date_range
    if landsat_dir is not None:
        workflow_kwargs["landsat_dir"] = landsat_dir
    if met_dir is not None:
        workflow_kwargs["met_dir"] = met_dir
    if dem is not None:
        workflow_kwargs["dem"] = dem
    if output_dir is not None:
        workflow_kwargs["output_dir"] = output_dir

    meaningful_inputs = {
        key: value
        for key, value in workflow_kwargs.items()
        if key != "date_range" and value not in (None, "")
    }
    if not meaningful_inputs:
        raise ValueError("Missing required inputs for SSEBop run")

    run_ssebop_workflow(**workflow_kwargs)
