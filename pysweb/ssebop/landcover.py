#!/usr/bin/env python3
"""
Script: landcover.py
Objective: Provide package-native SSEBop landcover loaders and masks for ESA WorldCover inputs.
Author: Yi Yu (with assistance from Codex)
Created: 2026-04-17
Last updated: 2026-10-22
Inputs: WorldCover raster paths and xarray landcover grids.
Outputs: Landcover DataArray values and class masks.
Usage: import pysweb.ssebop.landcover
Dependencies: rioxarray, xarray
"""
from __future__ import annotations

from typing import Optional

import rioxarray  # noqa: F401
import xarray as xr


__all__ = ["load_worldcover_landcover", "worldcover_masks"]

WORLDCOVER_CLASSES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
WORLDCOVER_ASSET = 'ESA/WorldCover/v200'
WORLDCOVER_FILENAME = 'worldcover_v200_2021.tif'


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
    template: xr.DataArray | None = None,
) -> xr.DataArray:
    """Load ESA WorldCover (v200) GeoTIFF and return landcover classes."""
    if path is None:
        raise ValueError("path is required for package-level WorldCover loading")
    with rioxarray.open_rasterio(path, masked=masked) as source:
        if template is not None:
            from rasterio.warp import transform_bounds
            bounds = transform_bounds(template.rio.crs, source.rio.crs,
                                      *template.rio.bounds(), densify_pts=21)
            rx, ry = map(abs, source.rio.resolution())
            source = source.rio.clip_box(bounds[0]-rx, bounds[1]-ry,
                                        bounds[2]+rx, bounds[3]+ry)
        lc = source.squeeze("band", drop=True).load()
    return _ensure_spatial_dims(lc)


def worldcover_masks(
    landcover: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Build reference, anomalous and water masks (not an exact NLCD crosswalk).

    Tree cover and mangroves are excluded from agricultural references.
    WorldCover does not distinguish developed open space from dense built-up
    land, so all built-up pixels use the anomalous/non-agricultural branch.
    """
    ag_mask = landcover.isin([30, 40, 90]).astype("uint8")
    anomalous_mask = landcover.isin([20, 50, 60]).astype("uint8")
    water_mask = landcover.isin([80]).astype("uint8")
    return ag_mask, anomalous_mask, water_mask
