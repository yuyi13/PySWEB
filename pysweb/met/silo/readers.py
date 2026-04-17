"""SILO meteorology readers for pysweb."""
from __future__ import annotations

import xarray as xr


def open_silo_da(path: str, variable: str) -> xr.DataArray:
    dataset = xr.open_dataset(path)
    if variable not in dataset:
        raise ValueError(f"Variable '{variable}' not found in {path}")
    return dataset[variable]
