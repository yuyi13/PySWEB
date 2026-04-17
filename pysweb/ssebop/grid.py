#!/usr/bin/env python3
"""
Script: grid.py
Objective: Provide package-native SSEBop raster reprojection helpers.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: xarray DataArray rasters with CRS metadata.
Outputs: Reprojected DataArray rasters aligned to a target grid.
Usage: Imported by pysweb.ssebop consumers.
Dependencies: rasterio, xarray
"""
from __future__ import annotations

from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
import xarray as xr


__all__ = ["reproject_match", "reproject_match_crop_first"]


def reproject_match(
    source: xr.DataArray,
    match: xr.DataArray,
    resampling: str = "nearest",
) -> xr.DataArray:
    """Reproject a DataArray to match another grid."""
    resampling_enum = Resampling[resampling]
    return source.rio.reproject_match(match, resampling=resampling_enum)


def reproject_match_crop_first(
    source: xr.DataArray,
    match: xr.DataArray,
    resampling: str = "nearest",
    buffer: float = 0.0,
) -> xr.DataArray:
    """Crop to the match bounds (plus buffer), then reproject to match."""
    if source.rio.crs is None:
        raise ValueError("source must have a CRS")
    if match.rio.crs is None:
        raise ValueError("match must have a CRS")

    left, bottom, right, top = match.rio.bounds()
    if buffer:
        left -= buffer
        bottom -= buffer
        right += buffer
        top += buffer

    src_bounds = transform_bounds(
        match.rio.crs,
        source.rio.crs,
        left,
        bottom,
        right,
        top,
        densify_pts=21,
    )
    try:
        clipped = source.rio.clip_box(*src_bounds, allow_one_dimensional_raster=True)
    except Exception:
        clipped = source
    return reproject_match(clipped, match, resampling=resampling)
