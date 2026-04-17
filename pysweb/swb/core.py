"""
Script: core.py
Objective: Provide reusable SWB forcing and soil-input helpers for package-owned run and calibration workflows.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: NetCDF forcing paths, soil-property NetCDF paths, and layer metadata or user-supplied layer depths.
Outputs: Loaded xarray DataArrays, resolved soil-path mappings, numpy soil-property grids, and validation errors.
Usage: Imported as `pysweb.swb.core`
Dependencies: numpy, xarray
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import xarray as xr

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


def load_single_variable(path: str | Path) -> xr.DataArray:
    dataset_path = Path(path).expanduser().resolve()
    with xr.open_dataset(dataset_path) as ds:
        data_vars = list(ds.data_vars)
        if not data_vars:
            raise ValueError(f"No data variables found in {dataset_path}")
        da = ds[data_vars[0]].load()
    return da


def resolve_soil_paths(
    *,
    soil_dir: str | Path | None = None,
    soil_porosity: str | Path | None = None,
    soil_wilting_point: str | Path | None = None,
    soil_available_water_capacity: str | Path | None = None,
    soil_b_coefficient: str | Path | None = None,
    soil_conductivity_sat: str | Path | None = None,
) -> Dict[str, Path]:
    soil_dir_path = Path(soil_dir).expanduser().resolve() if soil_dir else None
    overrides = {
        "porosity": soil_porosity,
        "wilting_point": soil_wilting_point,
        "available_water_capacity": soil_available_water_capacity,
        "b_coefficient": soil_b_coefficient,
        "conductivity_sat": soil_conductivity_sat,
    }

    paths: Dict[str, Path] = {}
    for key, filename in SOIL_FILE_STEMS.items():
        override = overrides[key]
        if override:
            candidate = Path(override).expanduser().resolve()
        elif soil_dir_path is not None:
            candidate = soil_dir_path / filename
        else:
            raise ValueError(
                f"Provide soil_dir or an explicit path for soil property '{key}'."
            )
        if not candidate.exists():
            raise FileNotFoundError(f"Soil input not found for '{key}': {candidate}")
        paths[key] = candidate

    return paths


def load_soil_arrays(
    *,
    soil_dir: str | Path | None = None,
    soil_porosity: str | Path | None = None,
    soil_wilting_point: str | Path | None = None,
    soil_available_water_capacity: str | Path | None = None,
    soil_b_coefficient: str | Path | None = None,
    soil_conductivity_sat: str | Path | None = None,
) -> tuple[Dict[str, xr.DataArray], Dict[str, Path]]:
    paths = resolve_soil_paths(
        soil_dir = soil_dir,
        soil_porosity = soil_porosity,
        soil_wilting_point = soil_wilting_point,
        soil_available_water_capacity = soil_available_water_capacity,
        soil_b_coefficient = soil_b_coefficient,
        soil_conductivity_sat = soil_conductivity_sat,
    )
    arrays = {key: load_single_variable(path) for key, path in paths.items()}
    return arrays, paths


def infer_layer_bottoms(
    soil_arrays: Mapping[str, xr.DataArray],
    user_bottoms: Optional[Sequence[float]] = None,
) -> np.ndarray:
    if user_bottoms is not None:
        values = np.asarray(list(user_bottoms), dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("layer_bottoms_mm must contain at least one value.")
        return values

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

    return np.asarray(DEFAULT_LAYER_BOTTOMS_MM, dtype=float)


def ensure_matching_grid(
    reference: xr.DataArray,
    soil_arrays: Mapping[str, xr.DataArray],
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
    soil_arrays: Mapping[str, xr.DataArray],
    layer_bottoms_mm: Sequence[float],
    *,
    lat_dim: str,
    lon_dim: str,
    root_beta: float,
    drainage_slope: float,
    drainage_upper_limit: float,
    drainage_lower_limit: float,
    sm_max_factor: float,
    sm_min_factor: float,
) -> Dict[str, np.ndarray]:
    sample = soil_arrays["porosity"]
    if "layer" not in sample.dims:
        raise ValueError("Soil arrays must include a 'layer' dimension.")

    layer_bottoms = np.asarray(layer_bottoms_mm, dtype=float)
    if layer_bottoms.ndim != 1 or layer_bottoms.size == 0:
        raise ValueError("layer_bottoms_mm must contain at least one depth value.")
    if layer_bottoms.size != sample.sizes["layer"]:
        raise ValueError(
            "Number of layer bottoms does not match number of soil layers "
            f"({layer_bottoms.size} vs {sample.sizes['layer']})."
        )

    layer_thickness = np.empty_like(layer_bottoms, dtype=float)
    layer_thickness[0] = layer_bottoms[0]
    if layer_bottoms.size > 1:
        layer_thickness[1:] = np.diff(layer_bottoms)

    lat_count = sample.sizes[lat_dim]
    lon_count = sample.sizes[lon_dim]

    def to_numpy(name: str) -> np.ndarray:
        da = soil_arrays[name].transpose("layer", lat_dim, lon_dim)
        return np.asarray(da.values, dtype=float)

    soil_grids: Dict[str, np.ndarray] = {
        "layer_depth": layer_bottoms.astype(float),
        "layer_thickness": layer_thickness.astype(float),
        "porosity": to_numpy("porosity"),
        "wilting_point": to_numpy("wilting_point"),
        "available_water_capacity": to_numpy("available_water_capacity"),
        "b_coefficient": to_numpy("b_coefficient"),
        "conductivity_sat": to_numpy("conductivity_sat"),
        "root_beta": np.full((lat_count, lon_count), root_beta, dtype=float),
        "drainage_slope": np.full((lat_count, lon_count), drainage_slope, dtype=float),
        "drainage_upper_limit": np.full((lat_count, lon_count), drainage_upper_limit, dtype=float),
        "drainage_lower_limit": np.full((lat_count, lon_count), drainage_lower_limit, dtype=float),
        "sm_max_factor": np.full((lat_count, lon_count), sm_max_factor, dtype=float),
        "sm_min_factor": np.full((lat_count, lon_count), sm_min_factor, dtype=float),
    }
    return soil_grids


def extract_soil_properties_for_cell(
    soil_grids: Mapping[str, np.ndarray],
    lat_idx: int,
    lon_idx: int,
) -> Dict[str, np.ndarray | float]:
    props: Dict[str, np.ndarray | float] = {}
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


def has_invalid_soil_values(props: Mapping[str, np.ndarray | float]) -> bool:
    return any(not np.all(np.isfinite(np.asarray(value, dtype=float))) for value in props.values())


def load_forcing(
    path: str | Path,
    variable: str,
    start: str | None = None,
    end: str | None = None,
) -> xr.DataArray:
    dataset_path = Path(path).expanduser().resolve()
    with xr.open_dataset(dataset_path) as ds:
        if variable not in ds:
            raise KeyError(f"Variable '{variable}' not found in {dataset_path}")
        da = ds[variable]
        if start or end:
            da = da.sel(time=slice(start, end))
        return da.load()
