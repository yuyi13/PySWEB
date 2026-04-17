#!/usr/bin/env python3
"""
Script: landcover.py
Objective: Provide package-native SSEBop landcover loaders and masks for ESA WorldCover inputs.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: WorldCover raster paths and xarray landcover grids.
Outputs: Landcover DataArray values and class masks.
Usage: Imported by pysweb.ssebop consumers.
Dependencies: rioxarray, xarray
"""
from __future__ import annotations

from typing import Optional

import rioxarray  # noqa: F401
import xarray as xr


__all__ = ["load_worldcover_landcover", "worldcover_masks"]


def _ensure_spatial_dims(data_array: xr.DataArray) -> xr.DataArray:
    """Ensure spatial dims are named consistently for package helpers."""
    if {"x", "y"}.issubset(set(data_array.dims)):
        return data_array
    if {"lon", "lat"}.issubset(set(data_array.dims)):
        data_array = data_array.rename({"lon": "x", "lat": "y"})
    return data_array


def load_worldcover_landcover(
    path: Optional[str] = None,
    masked: bool = True,
) -> xr.DataArray:
    """Load ESA WorldCover (v200) GeoTIFF and return landcover classes."""
    if path is None:
        raise ValueError("path is required for package-level WorldCover loading")
    lc = rioxarray.open_rasterio(path, masked=masked).squeeze("band", drop=True)
    return _ensure_spatial_dims(lc)


def worldcover_masks(
    landcover: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Build ag, anomalous, and water masks from WorldCover classes."""
    ag_mask = landcover.isin([30, 40, 90]).astype("uint8")
    anomalous_mask = landcover.isin([20, 50, 60]).astype("uint8")
    water_mask = landcover.isin([80]).astype("uint8")
    return ag_mask, anomalous_mask, water_mask
