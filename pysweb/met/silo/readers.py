"""SILO meteorology readers for pysweb."""
from __future__ import annotations

import xarray as xr


def _ensure_spatial_dims(data_array: xr.DataArray) -> xr.DataArray:
    """Ensure spatial dims are named consistently for package helpers."""
    if {"x", "y"}.issubset(set(data_array.dims)):
        return data_array
    if {"lon", "lat"}.issubset(set(data_array.dims)):
        data_array = data_array.rename({"lon": "x", "lat": "y"})
    return data_array


def open_silo_da(path: str, variable: str) -> xr.DataArray:
    with xr.open_dataset(path) as dataset:
        if variable not in dataset:
            raise ValueError(f"Variable '{variable}' not found in {path}")
        data_array = dataset[variable].load()
    return data_array


def open_silo_et_short_crop(file_path: str, variable: str = "et_short_crop") -> xr.DataArray:
    """Open SILO FAO56 ETo NetCDF and return the ETo DataArray."""
    return open_silo_variable(file_path, variable)


def open_silo_variable(file_path: str, variable: str) -> xr.DataArray:
    """Open a SILO NetCDF and return the requested variable as a DataArray."""
    return _ensure_spatial_dims(open_silo_da(file_path, variable))
