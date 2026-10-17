#!/usr/bin/env python3
"""
Script: raster_grid.py
Objective: Select exact integer-aligned raster windows without resampling cells.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Open source and target raster datasets with CRS and affine grids.
Outputs: A validated source window covering the target grid exactly.
Usage: aligned_grid_window(source, target)
Dependencies: rasterio, math.
"""
import math

from rasterio.windows import Window


def aligned_grid_window(source, target, *, tolerance_pixels=1e-7):
    """Return a contained target window; reject reprojection or resampling.

    The tolerance covers floating-point affine serialization only. It is in
    source pixels and does not allow a fractional-cell grid displacement.
    Both objects must expose crs, transform, height and width attributes.
    """
    if not math.isfinite(tolerance_pixels) or not 0 < tolerance_pixels <= 1e-5:
        raise ValueError("Grid tolerance must be positive and at most 1e-5 pixels")
    if source.crs is None or target.crs is None or source.crs != target.crs:
        raise ValueError("Source and target CRS must be defined and equal")
    relative = ~source.transform * target.transform
    if any(abs(actual - expected) > tolerance_pixels for actual, expected in
           zip((relative.a, relative.b, relative.d, relative.e), (1, 0, 0, 1))):
        raise ValueError("Target grid requires resampling")
    col, row = round(relative.c), round(relative.f)
    if max(abs(relative.c - col), abs(relative.f - row)) > tolerance_pixels:
        raise ValueError("Target grid has a fractional-cell displacement")
    if (target.width <= 0 or target.height <= 0 or col < 0 or row < 0
            or col + target.width > source.width
            or row + target.height > source.height):
        raise ValueError("Target grid is not contained in source raster")
    return Window(col, row, target.width, target.height)
