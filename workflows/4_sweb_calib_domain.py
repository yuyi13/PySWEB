#!/usr/bin/env python3
"""
Calibrate SWEB parameters against domain-wide SMAP-DS surface soil moisture.

Example
-------
python 4_sweb_calib_domain.py \
    --precip /g/data/ym05/sweb_model/3_spatial_preprocess/rain_daily_20210101_20210131.nc \
    --et /g/data/ym05/sweb_model/3_spatial_preprocess/et_daily_20210101_20210131.nc \
    --t /g/data/ym05/sweb_model/3_spatial_preprocess/t_daily_20210101_20210131.nc \
    --soil-dir /g/data/ym05/sweb_model/3_spatial_preprocess \
    --smap-ssm /g/data/ym05/sweb_model/3_spatial_preprocess/smap_ssm_daily_20210101_20210131.nc \
    --date-range 2021-01-01 2021-01-31 \
    --output /g/data/ym05/sweb_model/3_spatial_preprocess/domain_calibration.csv
"""

from __future__ import annotations

import argparse
import csv
import inspect
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import differential_evolution

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


def _load_single_variable(path: Path, var: Optional[str] = None) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        if var is not None:
            if var not in ds:
                raise KeyError(f"Variable '{var}' not found in {path}")
            da = ds[var]
        else:
            data_vars = list(ds.data_vars)
            if not data_vars:
                raise ValueError(f"No data variables found in {path}")
            da = ds[data_vars[0]]
        return da.load()


def _resolve_soil_paths(soil_dir: Path) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for key, filename in SOIL_FILE_STEMS.items():
        candidate = soil_dir / filename
        if not candidate.exists():
            raise FileNotFoundError(f"Soil input not found for '{key}': {candidate}")
        paths[key] = candidate
    return paths


def _load_soil_arrays(soil_dir: Path) -> Dict[str, xr.DataArray]:
    paths = _resolve_soil_paths(soil_dir)
    arrays: Dict[str, xr.DataArray] = {}
    for key, path in paths.items():
        arrays[key] = _load_single_variable(path)
    return arrays


def _build_layer_thickness(layer_bottoms_mm: Sequence[float]) -> np.ndarray:
    layer_bottoms = np.asarray(layer_bottoms_mm, dtype=float)
    thickness = np.empty_like(layer_bottoms)
    thickness[0] = layer_bottoms[0]
    if layer_bottoms.size > 1:
        thickness[1:] = np.diff(layer_bottoms)
    return thickness


def _surface_layer_index(layer_depths: Sequence[float], surface_depth: float) -> int:
    for idx, depth in enumerate(layer_depths):
        if depth >= surface_depth:
            return idx
    return len(layer_depths) - 1


def _soil_valid_mask(soil_arrays: Dict[str, xr.DataArray]) -> np.ndarray:
    mask = None
    for da in soil_arrays.values():
        valid = np.isfinite(da.values).all(axis=0)
        mask = valid if mask is None else (mask & valid)
    if mask is None:
        raise ValueError("No soil arrays available for validation.")
    return mask


def _extract_props_from_arrays(
    soil_arrays: Dict[str, np.ndarray],
    layer_bottoms_mm: Sequence[float],
    lat_idx: int,
    lon_idx: int,
    root_beta: float,
    ndvi_value: float,
    drainage_slope: float,
    drainage_upper_limit: float,
    drainage_lower_limit: float,
    sm_max_factor: float,
    sm_min_factor: float,
    use_ndvi_root_depth: bool,
) -> Dict[str, np.ndarray]:
    layer_bottoms = np.asarray(layer_bottoms_mm, dtype=float)
    thickness = _build_layer_thickness(layer_bottoms)
    props = {
        "layer_depth": layer_bottoms,
        "layer_thickness": thickness,
        "porosity": soil_arrays["porosity"][:, lat_idx, lon_idx].astype(float, copy=False),
        "wilting_point": soil_arrays["wilting_point"][:, lat_idx, lon_idx].astype(float, copy=False),
        "available_water_capacity": soil_arrays["available_water_capacity"][:, lat_idx, lon_idx].astype(float, copy=False),
        "b_coefficient": soil_arrays["b_coefficient"][:, lat_idx, lon_idx].astype(float, copy=False),
        "conductivity_sat": soil_arrays["conductivity_sat"][:, lat_idx, lon_idx].astype(float, copy=False),
        "root_beta": float(root_beta),
        "drainage_slope": float(drainage_slope),
        "drainage_upper_limit": float(drainage_upper_limit),
        "drainage_lower_limit": float(drainage_lower_limit),
        "sm_max_factor": float(sm_max_factor),
        "sm_min_factor": float(sm_min_factor),
        "use_ndvi_root_depth": bool(use_ndvi_root_depth),
    }
    if bool(use_ndvi_root_depth) and np.isfinite(ndvi_value):
        props["ndvi"] = float(ndvi_value)
    return props


def _parse_dates(args: argparse.Namespace) -> Tuple[pd.Timestamp, pd.Timestamp]:
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


def _compute_rmse(
    params: Sequence[float],
    precip_vals: np.ndarray,
    et_vals: np.ndarray,
    t_vals: np.ndarray,
    smap_vals: np.ndarray,
    time_index: pd.DatetimeIndex,
    soil_values: Dict[str, np.ndarray],
    soil_valid: np.ndarray,
    ndvi_mean_vals: Optional[np.ndarray],
    layer_bottoms_mm: Sequence[float],
    surface_layer_idx: int,
    drainage_slope: float,
    drainage_upper_limit: float,
    drainage_lower_limit: float,
    use_ndvi_root_depth: bool,
) -> Tuple[float, int]:
    infil_coeff, diff_factor, sm_max_factor, sm_min_factor, root_beta = params
    model_col = f"layer_{surface_layer_idx + 1}"

    obs_mask = np.isfinite(smap_vals) & soil_valid[None, :, :]
    obs_sum = np.where(obs_mask, smap_vals, 0.0).sum(axis=(1, 2))
    obs_count = obs_mask.sum(axis=(1, 2))
    obs_mean = np.full_like(obs_sum, np.nan, dtype=float)
    valid_obs = obs_count > 0
    obs_mean[valid_obs] = obs_sum[valid_obs] / obs_count[valid_obs]

    sim_sum = np.zeros_like(obs_sum, dtype=float)
    sim_count = np.zeros_like(obs_count, dtype=int)

    n_lat, n_lon = soil_valid.shape
    for lat_idx in range(n_lat):
        for lon_idx in range(n_lon):
            if not soil_valid[lat_idx, lon_idx]:
                continue

            if not np.any(obs_mask[:, lat_idx, lon_idx]):
                continue

            ndvi_value = np.nan
            if ndvi_mean_vals is not None:
                ndvi_value = float(ndvi_mean_vals[lat_idx, lon_idx])

            soil_props = _extract_props_from_arrays(
                soil_values,
                layer_bottoms_mm,
                lat_idx,
                lon_idx,
                root_beta,
                ndvi_value,
                drainage_slope,
                drainage_upper_limit,
                drainage_lower_limit,
                sm_max_factor,
                sm_min_factor,
                use_ndvi_root_depth,
            )

            try:
                simulated = soil_water_balance_1d(
                    precip_vals[:, lat_idx, lon_idx],
                    et_vals[:, lat_idx, lon_idx],
                    soil_props,
                    time_index,
                    time_step=1.0,
                    initial_soil_moisture=None,
                    infil_coeff=infil_coeff,
                    diff_factor=diff_factor,
                    transpiration_data=t_vals[:, lat_idx, lon_idx],
                )
            except Exception:
                continue

            if model_col not in simulated.columns:
                continue

            sim_vals = simulated[model_col].values
            valid_time = obs_mask[:, lat_idx, lon_idx]
            sim_sum[valid_time] += sim_vals[valid_time]
            sim_count[valid_time] += 1

    valid_time = (obs_count > 0) & (sim_count > 0)
    if not np.any(valid_time):
        return float("inf"), 0

    sim_mean = np.full_like(sim_sum, np.nan, dtype=float)
    sim_mean[valid_time] = sim_sum[valid_time] / sim_count[valid_time]

    diff = sim_mean[valid_time] - obs_mean[valid_time]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return rmse, int(obs_count[valid_time].sum())


def _objective_function(
    params: Sequence[float],
    precip_vals: np.ndarray,
    et_vals: np.ndarray,
    t_vals: np.ndarray,
    smap_vals: np.ndarray,
    time_index: pd.DatetimeIndex,
    soil_values: Dict[str, np.ndarray],
    soil_valid: np.ndarray,
    ndvi_mean_vals: Optional[np.ndarray],
    layer_bottoms_mm: Sequence[float],
    surface_layer_idx: int,
    drainage_slope: float,
    drainage_upper_limit: float,
    drainage_lower_limit: float,
    use_ndvi_root_depth: bool,
) -> float:
    rmse, _ = _compute_rmse(
        params,
        precip_vals,
        et_vals,
        t_vals,
        smap_vals,
        time_index,
        soil_values,
        soil_valid,
        ndvi_mean_vals,
        layer_bottoms_mm,
        surface_layer_idx,
        drainage_slope,
        drainage_upper_limit,
        drainage_lower_limit,
        use_ndvi_root_depth,
    )
    return rmse


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Domain-wide calibration using SMAP-DS SSM.")
    parser.add_argument("--precip", required=True, help="NetCDF file with precipitation (mm day-1).")
    parser.add_argument("--precip-var", default="precipitation", help="Variable name for precipitation.")
    parser.add_argument("--et", required=True, help="NetCDF file with evapotranspiration (mm day-1).")
    parser.add_argument("--et-var", default="et", help="Variable name for evapotranspiration.")
    parser.add_argument("--t", required=True, help="NetCDF file with transpiration (mm day-1).")
    parser.add_argument("--t-var", default="t", help="Variable name for transpiration.")
    parser.add_argument("--ndvi", help="Optional NetCDF file with NDVI on model grid.")
    parser.add_argument("--ndvi-var", default="ndvi", help="Variable name for NDVI.")
    parser.add_argument(
        "--use-ndvi-root-depth",
        action="store_true",
        help=(
            "Enable NDVI-constrained root depth capping. Disabled by default, so "
            "Jackson beta uptake spans all layers unless max_root_depth is set."
        ),
    )
    parser.add_argument("--soil-dir", required=True, help="Directory containing soil NetCDFs.")
    parser.add_argument("--smap-ssm", required=True, help="NetCDF file with SMAP-DS SSM on model grid.")
    parser.add_argument("--smap-var", default="smap_ssm", help="Variable name for SMAP-DS SSM.")
    parser.add_argument("--start-date", type=str, help="Calibration start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, help="Calibration end date (YYYY-MM-DD).")
    parser.add_argument("--date-range", nargs=2, metavar=("START", "END"), help="Shortcut for start/end date.")
    parser.add_argument("--lat-dim", default="lat", help="Latitude dimension name.")
    parser.add_argument("--lon-dim", default="lon", help="Longitude dimension name.")
    parser.add_argument(
        "--sm-res",
        type=float,
        default=0.01,
        help="Target resolution in degrees for domain-mean calibration (square grid).",
    )
    parser.add_argument("--max-iter", type=int, default=30, help="Maximum iterations for optimization.")
    parser.add_argument("--surface-depth", type=float, default=50.0, help="Depth of surface observation (mm).")
    parser.add_argument("--root-beta", type=float, default=0.961, help="Initial root_beta value for optimizer seeding.")
    parser.add_argument("--drainage-slope", type=float, default=0.5, help="Drainage slope parameter.")
    parser.add_argument("--drainage-upper-limit", type=float, default=25.0, help="Upper limit for drainage (mm day-1).")
    parser.add_argument("--drainage-lower-limit", type=float, default=0.0, help="Lower limit for drainage (mm day-1).")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for differential evolution objective evaluation.")
    parser.add_argument("--infil-bounds", nargs=2, type=float, default=(0.3, 1.0), help="Bounds for infil_coeff.")
    parser.add_argument(
        "--diff-bounds",
        nargs=2,
        type=float,
        default=(10, 1e4),
        help="Bounds for diff_factor (mm).",
    )
    parser.add_argument(
        "--sm-max-bounds",
        nargs=2,
        type=float,
        default=(1.0, 1.5),
        help="Bounds for sm_max_factor (multiplier on porosity).",
    )
    parser.add_argument(
        "--sm-min-bounds",
        nargs=2,
        type=float,
        default=(0.1, 1.0),
        help="Bounds for sm_min_factor (multiplier on wilting point).",
    )
    parser.add_argument(
        "--beta-bounds",
        nargs=2,
        type=float,
        default=(0.90, 0.995),
        help="Bounds for root_beta.",
    )
    parser.add_argument(
        "--layer-bottoms-mm",
        nargs="+",
        type=float,
        help="Layer bottoms in mm (defaults to 50 150 300 600 1000).",
    )
    parser.add_argument("--output", required=True, help="CSV output path for calibrated parameters.")
    return parser.parse_args(argv)


def _coarsen_to_target(
    da: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    target_res: float,
) -> xr.DataArray:
    lat_vals = da.coords[lat_dim].values
    lon_vals = da.coords[lon_dim].values
    if lat_vals.size < 2 or lon_vals.size < 2:
        return da

    lat_res = float(np.nanmean(np.abs(np.diff(lat_vals))))
    lon_res = float(np.nanmean(np.abs(np.diff(lon_vals))))
    if lat_res == 0 or lon_res == 0:
        return da

    lat_factor = max(1, int(round(target_res / lat_res)))
    lon_factor = max(1, int(round(target_res / lon_res)))
    if not np.isclose(lat_res * lat_factor, target_res, rtol=1e-3, atol=1e-6):
        raise ValueError(
            f"Target resolution {target_res} does not align with latitude spacing {lat_res}."
        )
    if not np.isclose(lon_res * lon_factor, target_res, rtol=1e-3, atol=1e-6):
        raise ValueError(
            f"Target resolution {target_res} does not align with longitude spacing {lon_res}."
        )

    if lat_factor == 1 and lon_factor == 1:
        return da

    return da.coarsen({lat_dim: lat_factor, lon_dim: lon_factor}, boundary="trim").mean()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be >= 1")

    start, end = _parse_dates(args)
    print("Loading forcing data...", flush=True)
    precip = _load_single_variable(Path(args.precip), args.precip_var)
    et = _load_single_variable(Path(args.et), args.et_var)
    t = _load_single_variable(Path(args.t), args.t_var)
    smap = _load_single_variable(Path(args.smap_ssm), args.smap_var)
    ndvi = None
    if args.use_ndvi_root_depth and args.ndvi:
        ndvi = _load_single_variable(Path(args.ndvi), args.ndvi_var)

    if "time" not in precip.coords:
        raise ValueError("Precipitation data must include a 'time' coordinate.")
    if "time" not in et.coords:
        raise ValueError("ET data must include a 'time' coordinate.")
    if "time" not in t.coords:
        raise ValueError("Transpiration data must include a 'time' coordinate.")
    if ndvi is not None and "time" not in ndvi.coords:
        raise ValueError("NDVI data must include a 'time' coordinate.")

    precip = precip.sel(time=slice(start, end))
    et = et.sel(time=precip.coords["time"])
    t = t.sel(time=precip.coords["time"])
    smap = smap.sel(time=precip.coords["time"])
    if ndvi is not None:
        ndvi = ndvi.sel(time=precip.coords["time"])

    lat_dim = args.lat_dim
    lon_dim = args.lon_dim
    time_index = pd.to_datetime(precip.coords["time"].values).normalize()

    if time_index.size != precip.sizes["time"]:
        raise ValueError("Time coordinate length mismatch in precipitation data.")
    if precip.sizes.get("time") != et.sizes.get("time"):
        raise ValueError("Precipitation and ET time dimensions do not align.")
    if precip.sizes.get("time") != t.sizes.get("time"):
        raise ValueError("Precipitation and transpiration time dimensions do not align.")
    if precip.sizes.get("time") != smap.sizes.get("time"):
        raise ValueError("SMAP-DS and forcing time dimensions do not align.")

    precip = precip.transpose("time", lat_dim, lon_dim)
    et = et.transpose("time", lat_dim, lon_dim)
    t = t.transpose("time", lat_dim, lon_dim)
    smap = smap.transpose("time", lat_dim, lon_dim)
    if ndvi is not None:
        ndvi = ndvi.transpose("time", lat_dim, lon_dim)

    if args.sm_res:
        precip = _coarsen_to_target(precip, lat_dim, lon_dim, args.sm_res)
        et = _coarsen_to_target(et, lat_dim, lon_dim, args.sm_res)
        t = _coarsen_to_target(t, lat_dim, lon_dim, args.sm_res)
        smap = _coarsen_to_target(smap, lat_dim, lon_dim, args.sm_res)
        if ndvi is not None:
            ndvi = _coarsen_to_target(ndvi, lat_dim, lon_dim, args.sm_res)
            precip, et, t, smap, ndvi = xr.align(precip, et, t, smap, ndvi, join="inner")
        else:
            precip, et, t, smap = xr.align(precip, et, t, smap, join="inner")

    soil_dir = Path(args.soil_dir).expanduser().resolve()
    print("Loading soil inputs...", flush=True)
    soil_arrays = _load_soil_arrays(soil_dir)
    if args.sm_res:
        soil_arrays = {
            key: _coarsen_to_target(da.transpose("layer", lat_dim, lon_dim), lat_dim, lon_dim, args.sm_res)
            for key, da in soil_arrays.items()
        }

    layer_bottoms_mm = args.layer_bottoms_mm or list(DEFAULT_LAYER_BOTTOMS_MM)
    surface_layer_idx = _surface_layer_index(layer_bottoms_mm, args.surface_depth)

    soil_values = {key: da.values for key, da in soil_arrays.items()}
    soil_valid = _soil_valid_mask(soil_arrays)
    precip_vals = precip.values.astype(float, copy=False)
    et_vals = et.values.astype(float, copy=False)
    t_vals = t.values.astype(float, copy=False)
    smap_vals = smap.values.astype(float, copy=False)
    ndvi_mean_vals: Optional[np.ndarray] = None
    if ndvi is not None:
        ndvi_vals = ndvi.values.astype(float, copy=False)
        valid_counts = np.sum(np.isfinite(ndvi_vals), axis=0)
        ndvi_sums = np.nansum(ndvi_vals, axis=0)
        ndvi_mean_vals = np.full(valid_counts.shape, np.nan, dtype=float)
        np.divide(ndvi_sums, valid_counts, out=ndvi_mean_vals, where=valid_counts > 0)

    n_time, n_lat, n_lon = precip_vals.shape
    print("Calibration setup:", flush=True)
    print(f"  precip: {Path(args.precip).expanduser().resolve()} (var={args.precip_var})", flush=True)
    print(f"  et: {Path(args.et).expanduser().resolve()} (var={args.et_var})", flush=True)
    print(f"  t: {Path(args.t).expanduser().resolve()} (var={args.t_var})", flush=True)
    if args.use_ndvi_root_depth and args.ndvi:
        print(
            f"  ndvi: {Path(args.ndvi).expanduser().resolve()} (var={args.ndvi_var}; "
            "used for root-depth capping)",
            flush=True,
        )
    elif args.use_ndvi_root_depth:
        print(
            "  ndvi: not provided (NDVI root-depth capping requested but unavailable; "
            "running without NDVI cap)",
            flush=True,
        )
    elif args.ndvi:
        print(
            f"  ndvi: {Path(args.ndvi).expanduser().resolve()} (var={args.ndvi_var}; "
            "ignored because --use-ndvi-root-depth is not set)",
            flush=True,
        )
    else:
        print("  ndvi root-depth capping: disabled (default)", flush=True)
    print(f"  smap_ssm: {Path(args.smap_ssm).expanduser().resolve()} (var={args.smap_var})", flush=True)
    print(f"  soil_dir: {soil_dir}", flush=True)
    print(
        f"  time range: {start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')} "
        f"({n_time} steps)",
        flush=True,
    )
    print(f"  dims: time={n_time}, {lat_dim}={n_lat}, {lon_dim}={n_lon}", flush=True)
    if args.sm_res:
        print(f"  target resolution: {args.sm_res} deg", flush=True)
    print(f"  layer bottoms (mm): {list(layer_bottoms_mm)}", flush=True)
    print(
        f"  surface depth (mm): {args.surface_depth} -> surface layer index={surface_layer_idx}",
        flush=True,
    )
    print(
        "  bounds: "
        f"infil_coeff={tuple(args.infil_bounds)}, "
        f"diff_factor={tuple(args.diff_bounds)}, "
        f"sm_max_factor={tuple(args.sm_max_bounds)}, "
        f"sm_min_factor={tuple(args.sm_min_bounds)}, "
        f"root_beta={tuple(args.beta_bounds)}",
        flush=True,
    )
    print(
        "  optimizer: scipy.optimize.differential_evolution "
        f"(maxiter={args.max_iter}, popsize=10, tol=1e-3, strategy=best1bin)",
        flush=True,
    )
    bounds = [
        tuple(args.infil_bounds),
        tuple(args.diff_bounds),
        tuple(args.sm_max_bounds),
        tuple(args.sm_min_bounds),
        tuple(args.beta_bounds),
    ]
    de_x0 = np.array([0.3, 1e3, 1.0, 1.0, float(args.root_beta)], dtype=float)
    for idx, (low, high) in enumerate(bounds):
        de_x0[idx] = float(np.clip(de_x0[idx], low, high))
    de_popsize = 10
    de_population_size = de_popsize * len(bounds)
    de_workers = max(1, min(args.workers, de_population_size))
    print(
        f"  optimizer workers (processes): requested={args.workers}, "
        f"effective={de_workers}, population={de_population_size}",
        flush=True,
    )
    print("  objective: RMSE between domain-mean simulated and observed SMAP-DS SSM", flush=True)
    print(
        f"  optimizer seed x0: infil_coeff={de_x0[0]:.6g}, diff_factor={de_x0[1]:.6g}, "
        f"sm_max_factor={de_x0[2]:.6g}, sm_min_factor={de_x0[3]:.6g}, "
        f"root_beta={de_x0[4]:.6g}",
        flush=True,
    )
    de_extra = {}
    if "x0" in inspect.signature(differential_evolution).parameters:
        de_extra["x0"] = de_x0
    result = differential_evolution(
        _objective_function,
        bounds,
        args=(
            precip_vals,
            et_vals,
            t_vals,
            smap_vals,
            time_index,
            soil_values,
            soil_valid,
            ndvi_mean_vals,
            layer_bottoms_mm,
            surface_layer_idx,
            args.drainage_slope,
            args.drainage_upper_limit,
            args.drainage_lower_limit,
            args.use_ndvi_root_depth,
        ),
        maxiter=args.max_iter,
        popsize=de_popsize,
        mutation=(0.5, 1.0),
        recombination=0.7,
        strategy="best1bin",
        tol=1e-3,
        disp=False,
        workers=de_workers,
        updating="deferred" if de_workers > 1 else "immediate",
        **de_extra,
    )

    best_params = result.x
    final_rmse, n_obs = _compute_rmse(
        best_params,
        precip_vals,
        et_vals,
        t_vals,
        smap_vals,
        time_index,
        soil_values,
        soil_valid,
        ndvi_mean_vals,
        layer_bottoms_mm,
        surface_layer_idx,
        args.drainage_slope,
        args.drainage_upper_limit,
        args.drainage_lower_limit,
        args.use_ndvi_root_depth,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["infil_coeff", "diff_factor", "sm_max_factor", "sm_min_factor", "root_beta", "rmse", "n_obs"]
        )
        writer.writerow(
            [
                f"{best_params[0]:.6g}",
                f"{best_params[1]:.6g}",
                f"{best_params[2]:.6g}",
                f"{best_params[3]:.6g}",
                f"{best_params[4]:.6g}",
                f"{final_rmse:.6g}",
                str(int(n_obs)),
            ]
        )

    print(f"Wrote domain calibration CSV: {output_path}", flush=True)


if __name__ == "__main__":
    main()
