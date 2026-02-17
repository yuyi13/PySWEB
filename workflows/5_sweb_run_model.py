#!/usr/bin/env python3
"""
Run the 1-D soil water balance model on preprocessed gridded forcings.

Example
-------
python 5_sweb_run_model.py \\
    --precip /g/data/ym05/sweb_model/2_spatial_preprocess/rain_daily_20210101_20210131.nc \\
    --et /g/data/ym05/sweb_model/2_spatial_preprocess/et_daily_20210101_20210131.nc \\
    --t /g/data/ym05/sweb_model/2_spatial_preprocess/t_daily_20210101_20210131.nc \\
    --soil-dir /g/data/ym05/sweb_model/2_spatial_preprocess \\
    --output-dir /g/data/ym05/sweb_model/3_model_output
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.swb_model_1d import soil_water_balance_1d

SOIL_FILE_STEMS = {
    "porosity": "soil_porosity.nc",
    "wilting_point": "soil_wilting_point.nc",
    "available_water_capacity": "soil_available_water_capacity.nc",
    "b_coefficient": "soil_b_coefficient.nc",
    "conductivity_sat": "soil_conductivity_sat.nc",
}

DEFAULT_LAYER_BOTTOMS_MM: Sequence[float] = (50.0, 150.0, 300.0, 600.0, 1000.0)
_LAYER_ATTR_KEYS: Sequence[str] = ("layer_bottoms_mm", "layer_depth_mm", "layer_depths_mm")
_LAYER_COORD_KEYS: Sequence[str] = ("layer_depth", "depth_mm", "depth")
_PROCESS_RUN_STATE: Dict[str, object] = {}


def _load_single_variable(path: Path) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        data_vars = list(ds.data_vars)
        if not data_vars:
            raise ValueError(f"No data variables found in {path}")
        da = ds[data_vars[0]].load()
    return da


def _resolve_soil_paths(args: argparse.Namespace) -> Dict[str, Path]:
    soil_dir = Path(args.soil_dir).expanduser().resolve() if args.soil_dir else None
    paths: Dict[str, Path] = {}
    for key, filename in SOIL_FILE_STEMS.items():
        attr_name = f"soil_{key}"
        override = getattr(args, attr_name, None)
        if override:
            candidate = Path(override).expanduser().resolve()
        elif soil_dir:
            candidate = soil_dir / filename
        else:
            raise ValueError(
                f"Provide --soil-dir or explicit path via --{attr_name.replace('_', '-')}"
            )
        if not candidate.exists():
            raise FileNotFoundError(f"Soil input not found for '{key}': {candidate}")
        paths[key] = candidate
    return paths


def load_soil_arrays(args: argparse.Namespace) -> Tuple[Dict[str, xr.DataArray], Dict[str, Path]]:
    paths = _resolve_soil_paths(args)
    arrays: Dict[str, xr.DataArray] = {}
    for key, path in paths.items():
        arrays[key] = _load_single_variable(path)
    return arrays, paths


def infer_layer_bottoms(
    soil_arrays: Dict[str, xr.DataArray],
    user_bottoms: Optional[Sequence[float]],
) -> np.ndarray:
    for da in soil_arrays.values():
        for attr_key in _LAYER_ATTR_KEYS:
            if attr_key in da.attrs:
                values = np.asarray(da.attrs[attr_key], dtype=float)
                if values.ndim == 1:
                    return values
        for coord_key in _LAYER_COORD_KEYS:
            if coord_key in da.coords:
                values = np.asarray(da.coords[coord_key].values, dtype=float)
                if values.ndim == 1:
                    return values
    if user_bottoms:
        values = np.asarray(list(user_bottoms), dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("--layer-bottoms-mm must contain at least one value.")
        return values
    return np.asarray(DEFAULT_LAYER_BOTTOMS_MM, dtype=float)


def ensure_matching_grid(
    reference: xr.DataArray,
    soil_arrays: Dict[str, xr.DataArray],
    lat_dim: str,
    lon_dim: str,
) -> None:
    lat_ref = reference.coords[lat_dim].values
    lon_ref = reference.coords[lon_dim].values
    for name, array in soil_arrays.items():
        if lat_dim not in array.dims or lon_dim not in array.dims:
            raise ValueError(f"Soil array '{name}' is missing '{lat_dim}'/'{lon_dim}' dimensions.")
        lat_vals = array.coords[lat_dim].values
        lon_vals = array.coords[lon_dim].values
        if not (np.array_equal(lat_vals, lat_ref) and np.array_equal(lon_vals, lon_ref)):
            raise ValueError(
                f"Soil array '{name}' does not share the same grid as the forcing data."
            )


def prepare_soil_property_grids(
    soil_arrays: Dict[str, xr.DataArray],
    layer_bottoms_mm: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    sample = soil_arrays["porosity"]
    lat_dim = args.lat_dim
    lon_dim = args.lon_dim
    if "layer" not in sample.dims:
        raise ValueError("Soil arrays must include a 'layer' dimension.")
    if layer_bottoms_mm.size != sample.sizes["layer"]:
        raise ValueError(
            "Number of layer bottoms does not match number of soil layers "
            f"({layer_bottoms_mm.size} vs {sample.sizes['layer']})."
        )

    layer_thickness = np.empty_like(layer_bottoms_mm, dtype=float)
    layer_thickness[0] = layer_bottoms_mm[0]
    if layer_bottoms_mm.size > 1:
        layer_thickness[1:] = np.diff(layer_bottoms_mm)

    lat_count = sample.sizes[lat_dim]
    lon_count = sample.sizes[lon_dim]

    def _to_numpy(name: str) -> np.ndarray:
        da = soil_arrays[name].transpose("layer", lat_dim, lon_dim)
        return np.asarray(da.values, dtype=float)

    soil_grids: Dict[str, np.ndarray] = {
        "layer_depth": layer_bottoms_mm.astype(float),
        "layer_thickness": layer_thickness.astype(float),
        "porosity": _to_numpy("porosity"),
        "wilting_point": _to_numpy("wilting_point"),
        "available_water_capacity": _to_numpy("available_water_capacity"),
        "b_coefficient": _to_numpy("b_coefficient"),
        "conductivity_sat": _to_numpy("conductivity_sat"),
        "root_beta": np.full((lat_count, lon_count), args.root_beta, dtype=float),
        "max_root_depth": np.full((lat_count, lon_count), float(layer_bottoms_mm[-1]), dtype=float),
        "drainage_slope": np.full((lat_count, lon_count), args.drainage_slope, dtype=float),
        "drainage_upper_limit": np.full((lat_count, lon_count), args.drainage_upper_limit, dtype=float),
        "drainage_lower_limit": np.full((lat_count, lon_count), args.drainage_lower_limit, dtype=float),
        "sm_max_factor": np.full((lat_count, lon_count), args.sm_max_factor, dtype=float),
        "sm_min_factor": np.full((lat_count, lon_count), args.sm_min_factor, dtype=float),
    }
    return soil_grids


def extract_soil_properties_for_cell(
    soil_grids: Dict[str, np.ndarray],
    lat_idx: int,
    lon_idx: int,
) -> Dict[str, np.ndarray]:
    props: Dict[str, np.ndarray] = {}
    for key, value in soil_grids.items():
        arr = np.asarray(value)
        if arr.ndim == 1:
            props[key] = arr.copy()
        elif arr.ndim == 2:
            props[key] = float(arr[lat_idx, lon_idx])
        elif arr.ndim == 3:
            props[key] = arr[:, lat_idx, lon_idx].copy()
        else:
            raise ValueError(f"Unsupported dimensionality for soil property '{key}'")
    return props


def has_invalid_soil_values(props: Dict[str, np.ndarray]) -> bool:
    return any(not np.all(np.isfinite(np.asarray(value, dtype=float))) for value in props.values())


def load_forcing(
    path: Path,
    variable: str,
    start: Optional[str],
    end: Optional[str],
) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise KeyError(f"Variable '{variable}' not found in {path}")
        da = ds[variable]
        if start or end:
            da = da.sel(time=slice(start, end))
        return da.load()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatial soil water balance driver.")
    parser.add_argument("--precip", required=True, help="NetCDF file with precipitation (mm day-1).")
    parser.add_argument("--precip-var", default="precipitation", help="Variable name for precipitation.")
    parser.add_argument("--et", required=True, help="NetCDF file with evapotranspiration (mm day-1).")
    parser.add_argument("--et-var", default="et", help="Variable name for evapotranspiration.")
    parser.add_argument("--t", required=True, help="NetCDF file with transpiration (mm day-1).")
    parser.add_argument("--t-var", default="t", help="Variable name for transpiration.")
    parser.add_argument("--lat-dim", default="lat", help="Latitude dimension name.")
    parser.add_argument("--lon-dim", default="lon", help="Longitude dimension name.")
    parser.add_argument("--start-date", type=str, help="Optional simulation start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, help="Optional simulation end date (YYYY-MM-DD).")
    parser.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START", "END"),
        help="Shortcut for supplying start and end date.",
    )
    parser.add_argument("--time-step", type=float, default=1.0, help="Model time step in days.")
    parser.add_argument("--infil-coeff", type=float, default=0.3, help="Infiltration coefficient for the top layer.")
    parser.add_argument("--diff-factor", type=float, default=2e5, help="Diffusivity scaling factor.")
    parser.add_argument(
        "--sm-max-factor",
        type=float,
        default=1.0,
        help="Multiplier for porosity to set the upper soil moisture bound.",
    )
    parser.add_argument(
        "--sm-min-factor",
        type=float,
        default=1.0,
        help="Multiplier for wilting point to set the lower soil moisture bound.",
    )
    parser.add_argument("--param-grid", help="NetCDF with spatially varying parameters.")
    parser.add_argument("--param-infil-var", default="infil_coeff", help="Variable name for infil_coeff grid.")
    parser.add_argument("--param-diff-var", default="diff_factor", help="Variable name for diff_factor grid.")
    parser.add_argument("--output-dir", required=True, help="Directory for consolidated RZSM NetCDF output.")
    parser.add_argument(
        "--output-file",
        help="Optional filename/path for consolidated RZSM NetCDF. Default: SWEB_RZSM_<start>_<end>.nc in output-dir.",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32", help="Storage dtype for outputs.")
    parser.add_argument("--nan-to-zero", action="store_true", help="Replace NaNs in forcing data with zeros.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip run if output NetCDF already exists.")
    parser.add_argument(
        "--sm-res",
        type=float,
        help="Optional output resolution for RZSM/products (square grid, degrees). Defaults to forcing grid.",
    )

    default_soil_dir = REPO_ROOT / "2_spatial_preprocess"
    parser.add_argument("--soil-dir", default=str(default_soil_dir), help="Directory containing preprocessed soil NetCDFs.")
    parser.add_argument("--soil-porosity", help="Override path to soil_porosity.nc.")
    parser.add_argument("--soil-wilting-point", help="Override path to soil_wilting_point.nc.")
    parser.add_argument("--soil-available-water-capacity", help="Override path to soil_available_water_capacity.nc.")
    parser.add_argument("--soil-b-coefficient", help="Override path to soil_b_coefficient.nc.")
    parser.add_argument("--soil-conductivity-sat", help="Override path to soil_conductivity_sat.nc.")
    parser.add_argument(
        "--layer-bottoms-mm",
        nargs="+",
        type=float,
        help="Depth to layer bottoms in mm (overrides defaults/inferred metadata).",
    )

    parser.add_argument("--root-beta", type=float, default=0.96, help="Root distribution beta parameter.")
    parser.add_argument("--drainage-slope", type=float, default=0.5, help="Drainage slope parameter.")
    parser.add_argument("--drainage-upper-limit", type=float, default=25.0, help="Upper limit for drainage (mm day-1).")
    parser.add_argument("--drainage-lower-limit", type=float, default=0.0, help="Lower limit for drainage (mm day-1).")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for spatial SWEB simulation.")

    args = parser.parse_args()
    if args.date_range:
        if args.start_date or args.end_date:
            parser.error("--date-range cannot be combined with --start-date or --end-date.")
        args.start_date, args.end_date = args.date_range
    return args


def _infer_resolution(coords: np.ndarray) -> Optional[float]:
    coords = np.asarray(coords, dtype=float)
    if coords.size < 2:
        return None
    diffs = np.diff(coords)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return None
    return float(np.median(np.abs(diffs)))


def _build_target_coords(coords: np.ndarray, res: float) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.size < 2:
        return coords
    step = np.sign(coords[1] - coords[0]) * abs(res)
    if step == 0:
        return coords
    stop = coords[-1] + step * 0.1
    return np.arange(coords[0], stop, step, dtype=float)


def _maybe_resample(
    da: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    target_lat: Optional[np.ndarray],
    target_lon: Optional[np.ndarray],
) -> xr.DataArray:
    if target_lat is None and target_lon is None:
        return da
    coords = {}
    if target_lat is not None:
        coords[lat_dim] = target_lat
    if target_lon is not None:
        coords[lon_dim] = target_lon
    return da.interp(coords, method="linear")


def _compute_row_soil_moisture(
    lat_idx: int,
    n_lon: int,
    precip_values: np.ndarray,
    et_values: np.ndarray,
    t_values: np.ndarray,
    soil_grids: Dict[str, np.ndarray],
    param_values: Optional[Dict[str, np.ndarray]],
    time_index: pd.DatetimeIndex,
    time_step: float,
    default_infil_coeff: float,
    default_diff_factor: float,
    emit_warnings: bool = True,
) -> np.ndarray:
    n_time = precip_values.shape[0]
    n_layers = int(np.asarray(soil_grids["layer_depth"]).size)
    row_soil_moisture = np.full((n_time, n_layers, n_lon), np.nan, dtype=float)

    for lon_idx in range(n_lon):
        precip_series = precip_values[:, lat_idx, lon_idx]
        et_series = et_values[:, lat_idx, lon_idx]
        t_series = t_values[:, lat_idx, lon_idx]

        if np.all(np.isnan(precip_series)) and np.all(np.isnan(et_series)) and np.all(np.isnan(t_series)):
            continue

        soil_props = extract_soil_properties_for_cell(soil_grids, lat_idx, lon_idx)
        if has_invalid_soil_values(soil_props):
            continue

        if param_values is not None:
            infil_coeff = float(param_values["infil_coeff"][lat_idx, lon_idx])
            diff_factor = float(param_values["diff_factor"][lat_idx, lon_idx])
            if not np.isfinite(infil_coeff + diff_factor):
                continue
        else:
            infil_coeff = default_infil_coeff
            diff_factor = default_diff_factor

        try:
            result = soil_water_balance_1d(
                precip_series,
                et_series,
                soil_props,
                time_index,
                time_step=time_step,
                initial_soil_moisture=None,
                infil_coeff=infil_coeff,
                diff_factor=diff_factor,
                transpiration_data=t_series,
            )
        except Exception as exc:  # pragma: no cover - diagnostic output
            if emit_warnings:
                print(
                    f"Warning: failed to solve soil moisture at cell ({lat_idx}, {lon_idx}): {exc}",
                    file=sys.stderr,
                )
            continue

        row_soil_moisture[:, :, lon_idx] = result.to_numpy(dtype=float, copy=False)

    return row_soil_moisture


def _init_process_run_worker(
    precip_values: np.ndarray,
    et_values: np.ndarray,
    t_values: np.ndarray,
    soil_grids: Dict[str, np.ndarray],
    param_values: Optional[Dict[str, np.ndarray]],
    time_index: pd.DatetimeIndex,
    time_step: float,
    default_infil_coeff: float,
    default_diff_factor: float,
) -> None:
    global _PROCESS_RUN_STATE
    _PROCESS_RUN_STATE = {
        "precip_values": precip_values,
        "et_values": et_values,
        "t_values": t_values,
        "soil_grids": soil_grids,
        "param_values": param_values,
        "time_index": time_index,
        "time_step": time_step,
        "default_infil_coeff": default_infil_coeff,
        "default_diff_factor": default_diff_factor,
    }


def _run_model_for_lat_process(lat_idx: int) -> Tuple[int, np.ndarray]:
    state = _PROCESS_RUN_STATE
    precip_values = state["precip_values"]
    et_values = state["et_values"]
    t_values = state["t_values"]
    soil_grids = state["soil_grids"]
    param_values = state["param_values"]
    time_index = state["time_index"]
    time_step = float(state["time_step"])
    default_infil_coeff = float(state["default_infil_coeff"])
    default_diff_factor = float(state["default_diff_factor"])
    n_lon = int(np.asarray(precip_values).shape[2])

    row_soil_moisture = _compute_row_soil_moisture(
        lat_idx=lat_idx,
        n_lon=n_lon,
        precip_values=precip_values,
        et_values=et_values,
        t_values=t_values,
        soil_grids=soil_grids,
        param_values=param_values,
        time_index=time_index,
        time_step=time_step,
        default_infil_coeff=default_infil_coeff,
        default_diff_factor=default_diff_factor,
        emit_warnings=False,
    )
    return lat_idx, row_soil_moisture


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be >= 1")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading forcing data…", flush=True)
    precip = load_forcing(Path(args.precip).expanduser(), args.precip_var, args.start_date, args.end_date)
    et = load_forcing(Path(args.et).expanduser(), args.et_var, args.start_date, args.end_date)
    t = load_forcing(Path(args.t).expanduser(), args.t_var, args.start_date, args.end_date)

    precip, et, t = xr.align(precip, et, t, join="inner")

    soil_arrays, soil_paths = load_soil_arrays(args)
    ensure_matching_grid(precip, soil_arrays, args.lat_dim, args.lon_dim)

    param_grids = None
    if args.param_grid:
        param_path = Path(args.param_grid).expanduser().resolve()
        if not param_path.exists():
            raise FileNotFoundError(f"Parameter grid not found: {param_path}")
        with xr.open_dataset(param_path) as ds:
            param_grids = {
                "infil_coeff": ds[args.param_infil_var].load(),
                "diff_factor": ds[args.param_diff_var].load(),
            }
        ensure_matching_grid(param_grids["infil_coeff"], soil_arrays, args.lat_dim, args.lon_dim)
        ensure_matching_grid(param_grids["diff_factor"], soil_arrays, args.lat_dim, args.lon_dim)

    layer_bottoms = infer_layer_bottoms(soil_arrays, args.layer_bottoms_mm)
    soil_grids = prepare_soil_property_grids(soil_arrays, layer_bottoms, args)
    param_values = None
    if param_grids is not None:
        param_values = {
            "infil_coeff": np.asarray(param_grids["infil_coeff"].values, dtype=float),
            "diff_factor": np.asarray(param_grids["diff_factor"].values, dtype=float),
        }

    time_index = pd.to_datetime(precip.coords["time"].values)
    latitudes = precip.coords[args.lat_dim].values
    longitudes = precip.coords[args.lon_dim].values

    precip_values = precip.values.astype(float, copy=False)
    et_values = et.values.astype(float, copy=False)
    t_values = t.values.astype(float, copy=False)

    if args.nan_to_zero:
        precip_values = np.nan_to_num(precip_values, nan=0.0)
        et_values = np.nan_to_num(et_values, nan=0.0)
        t_values = np.nan_to_num(t_values, nan=0.0)

    n_time, n_lat, n_lon = precip_values.shape
    n_layers = soil_grids["layer_depth"].size

    total_cells = n_lat * n_lon
    print(
        f"Running model across {total_cells} grid cells with {n_layers} layers "
        f"(workers={args.workers})...",
        flush=True,
    )
    soil_moisture = np.full((n_time, n_layers, n_lat, n_lon), np.nan, dtype=float)
    progress_interval = max(1, total_cells // 20)
    processed_cells = 0
    reported_cells = 0

    if args.workers == 1:
        for lat_idx in range(n_lat):
            row_soil_moisture = _compute_row_soil_moisture(
                lat_idx=lat_idx,
                n_lon=n_lon,
                precip_values=precip_values,
                et_values=et_values,
                t_values=t_values,
                soil_grids=soil_grids,
                param_values=param_values,
                time_index=time_index,
                time_step=args.time_step,
                default_infil_coeff=args.infil_coeff,
                default_diff_factor=args.diff_factor,
            )
            soil_moisture[:, :, lat_idx, :] = row_soil_moisture
            processed_cells += n_lon
            if (processed_cells - reported_cells) >= progress_interval or processed_cells == total_cells:
                print(f"Processed {processed_cells}/{total_cells} cells", flush=True)
                reported_cells = processed_cells
    else:
        effective_workers = max(1, min(args.workers, n_lat))
        print(
            f"Using process workers: requested={args.workers}, effective={effective_workers}, rows={n_lat}",
            flush=True,
        )
        mp_context = None
        try:
            mp_context = mp.get_context("fork")
        except ValueError:
            mp_context = None

        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=mp_context,
            initializer=_init_process_run_worker,
            initargs=(
                precip_values,
                et_values,
                t_values,
                soil_grids,
                param_values,
                time_index,
                args.time_step,
                args.infil_coeff,
                args.diff_factor,
            ),
        ) as executor:
            futures = [
                executor.submit(
                    _run_model_for_lat_process,
                    lat_idx,
                )
                for lat_idx in range(n_lat)
            ]
            for future in as_completed(futures):
                lat_idx, row_soil_moisture = future.result()
                soil_moisture[:, :, lat_idx, :] = row_soil_moisture
                processed_cells += n_lon
                if (processed_cells - reported_cells) >= progress_interval or processed_cells == total_cells:
                    print(f"Processed {processed_cells}/{total_cells} cells", flush=True)
                    reported_cells = processed_cells

    layer_ids = np.arange(1, n_layers + 1, dtype=int)
    storage_dtype = np.float32 if args.dtype == "float32" else np.float64
    fill_value = storage_dtype(np.nan)
    layer_thickness = soil_grids["layer_thickness"].astype(float, copy=False)

    if args.soil_dir:
        soil_source_dir = Path(args.soil_dir).expanduser().resolve()
    else:
        soil_source_dir = Path(next(iter(soil_paths.values()))).parent

    if args.output_file:
        output_path = Path(args.output_file).expanduser()
        if not output_path.is_absolute():
            output_path = out_dir / output_path
    else:
        output_path = out_dir / f"SWEB_RZSM_{time_index[0]:%Y-%m-%d}_{time_index[-1]:%Y-%m-%d}.nc"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and output_path.exists():
        print(f"Skipping existing output: {output_path}", flush=True)
        return

    sm_target_lat = None
    sm_target_lon = None
    if args.sm_res:
        sm_res = float(args.sm_res)
        lon_res = _infer_resolution(longitudes)
        lat_res = _infer_resolution(latitudes)
        if lon_res is None or not np.isclose(lon_res, abs(sm_res)):
            sm_target_lon = _build_target_coords(longitudes, sm_res)
        if lat_res is None or not np.isclose(lat_res, abs(sm_res)):
            sm_target_lat = _build_target_coords(latitudes, sm_res)
    print("Writing consolidated RZSM NetCDF…", flush=True)
    rzsm_all_da = xr.DataArray(
        soil_moisture.astype(storage_dtype, copy=False),
        dims=("time", "layer", args.lat_dim, args.lon_dim),
        coords={
            "time": time_index,
            "layer": layer_ids,
            args.lat_dim: latitudes,
            args.lon_dim: longitudes,
        },
        name="rzsm",
        attrs={
            "long_name": "Root zone soil moisture by layer",
            "units": "m3 m-3",
        },
    )
    profile_values = np.sum(
        soil_moisture.astype(storage_dtype, copy=False) * layer_thickness[None, :, None, None],
        axis=1,
    )
    profile_da = xr.DataArray(
        profile_values.astype(storage_dtype, copy=False),
        dims=("time", args.lat_dim, args.lon_dim),
        coords={
            "time": time_index,
            args.lat_dim: latitudes,
            args.lon_dim: longitudes,
        },
        name="profile_sm",
        attrs={
            "long_name": "Profile soil moisture",
            "units": "mm",
            "comment": "Sum of all rzsm_layer_* variables multiplied by layer thickness.",
        },
    )

    rzsm_all_da = _maybe_resample(rzsm_all_da, args.lat_dim, args.lon_dim, sm_target_lat, sm_target_lon)
    profile_da = _maybe_resample(profile_da, args.lat_dim, args.lon_dim, sm_target_lat, sm_target_lon)

    ds_rzsm = xr.Dataset()
    for layer_idx in range(n_layers):
        var_name = f"rzsm_layer_{layer_idx + 1}"
        layer_da = rzsm_all_da.isel(layer=layer_idx, drop=True).rename(var_name)
        layer_da.attrs.update(
            {
                "long_name": f"Root zone soil moisture layer {layer_idx + 1}",
                "units": "m3 m-3",
                "depth_bottom_mm": float(soil_grids["layer_depth"][layer_idx]),
                "layer_thickness_mm": float(layer_thickness[layer_idx]),
            }
        )
        ds_rzsm[var_name] = layer_da
    ds_rzsm["profile_sm"] = profile_da

    attrs = {
        "title": "Soil Water Balance Model Output (Consolidated RZSM)",
        "source": "core/swb_model_1d soil_water_balance_1d",
        "soil_property_source": str(soil_source_dir),
        "layer_bottoms_mm": np.asarray(soil_grids["layer_depth"], dtype=float).tolist(),
        "layer_thickness_mm": np.asarray(layer_thickness, dtype=float).tolist(),
    }
    if args.param_grid:
        attrs["parameter_grid"] = str(param_path)
    else:
        attrs["infiltration_coefficient"] = args.infil_coeff
        attrs["diffusivity_factor"] = args.diff_factor
    attrs["sm_max_factor"] = args.sm_max_factor
    attrs["sm_min_factor"] = args.sm_min_factor
    ds_rzsm.attrs.update(attrs)

    encoding: Dict[str, Dict[str, object]] = {}
    for var_name in ds_rzsm.data_vars:
        encoding[var_name] = {
            "dtype": storage_dtype,
            "_FillValue": fill_value,
            "zlib": True,
            "complevel": 4,
        }
    ds_rzsm.to_netcdf(output_path, encoding=encoding)
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
