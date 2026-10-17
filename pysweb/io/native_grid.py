"""
Script: native_grid.py
Objective: Preserve and verify Earth Engine source grids in raster downloads.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Earth Engine images, selected bands and downloaded raster metadata.
Outputs: Native CRS/affine export parameters and verified grid provenance.
Usage: from pysweb.io.native_grid import native_projection, verify_native_grid
Dependencies: earthengine-api, numpy, rasterio
"""
from __future__ import annotations

import math
import json
import numpy as np
from affine import Affine
from rasterio.crs import CRS


def native_projection(image, bands=None):
    """Resolve every selected band's grid; never silently resample mixed grids.

    A multiband GeoTIFF has one grid. Bands with different native grids must
    therefore be requested separately, rather than coerced onto the first band.
    """
    import ee

    if bands is None:
        bands = image.bandNames().getInfo()
    if not bands:
        raise ValueError("Native export requires at least one band")
    grids = ee.Dictionary({band: image.select([band]).projection() for band in bands}).getInfo()
    reference = grids[bands[0]]
    for band in bands:
        grid = grids[band]
        if (grid["crs"] != reference["crs"]
                or not np.allclose(grid["transform"], reference["transform"], rtol=0, atol=1e-12)):
            raise ValueError(f"Band {band} has a different native grid; download the bands separately")
    transform = reference["transform"]
    if len(transform) != 6 or not np.isfinite(transform).all() or abs(Affine(*transform).determinant) == 0:
        raise ValueError("Invalid native image transform")
    return {"crs": reference["crs"], "transform": list(transform), "bands": list(bands)}


def region_extent(region):
    """Resolve a small geographic export envelope, never image pixel data."""
    if hasattr(region, 'bounds'):
        region = region.bounds(maxError=1,proj='EPSG:4326').getInfo()
    if isinstance(region,str):
        region = json.loads(region)
    if isinstance(region,(list,tuple)) and len(region) == 4 and all(np.isscalar(v) for v in region):
        return tuple(float(v) for v in region)
    coordinates = np.asarray(region['coordinates'],dtype=float).reshape(-1,2)
    return (float(coordinates[:,0].min()),float(coordinates[:,1].min()),
            float(coordinates[:,0].max()),float(coordinates[:,1].max()))


def export_parameters(projection, extent=None):
    """Use an explicit integer pixel window to avoid global-origin EE clipping.

    The crop translates the affine origin by whole source pixels. It changes
    neither resolution nor alignment. No region+dimensions resize is requested.
    """
    if extent is None:
        return {"crs": projection["crs"], "crs_transform": projection["transform"]}
    from rasterio.warp import transform_bounds
    bounds = transform_bounds('EPSG:4326',projection['crs'],*extent,densify_pts=21)
    source = Affine(*projection['transform'])
    inverse = ~source
    corners = [inverse*(x,y) for x in (bounds[0],bounds[2]) for y in (bounds[1],bounds[3])]
    first_col = math.floor(min(p[0] for p in corners)+1e-7)
    first_row = math.floor(min(p[1] for p in corners)+1e-7)
    last_col = math.ceil(max(p[0] for p in corners)-1e-7)
    last_row = math.ceil(max(p[1] for p in corners)-1e-7)
    if last_col <= first_col or last_row <= first_row:
        raise ValueError('Native export envelope is empty')
    crop = source*Affine.translation(first_col,first_row)
    return {'crs':projection['crs'],'crs_transform':list(crop)[:6],
            'dimensions':[last_col-first_col,last_row-first_row]}


def verify_native_grid(dataset, projection, tolerance_pixels=1e-6):
    """Accept an integer-window subset of the source grid, including tile merges."""
    if dataset.crs != CRS.from_user_input(projection["crs"]):
        raise ValueError("Downloaded CRS differs from the native source CRS")
    relative = ~Affine(*projection["transform"]) * dataset.transform
    if not np.allclose([relative.a, relative.b, relative.d, relative.e], [1, 0, 0, 1],
                       rtol=0, atol=tolerance_pixels):
        raise ValueError("Downloaded resolution or rotation differs from the native source grid")
    if max(abs(relative.c-round(relative.c)), abs(relative.f-round(relative.f))) > tolerance_pixels:
        raise ValueError("Downloaded raster is shifted off the native source grid")
    return {"crs": str(dataset.crs), "transform": list(dataset.transform)[:6],
            "shape": [dataset.height, dataset.width], "native_grid_verified": True}
