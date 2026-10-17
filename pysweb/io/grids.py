"""
Script: grids.py
Objective: Select exact raster templates and explicitly align continuous inputs.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Georeferenced rasters, geographic AOI bounds and target templates.
Outputs: Native-grid subsets and aligned fields with explicit spatial coordinates.
Usage: from pysweb.io.grids import open_template, subset_template, align_to_grid
Dependencies: numpy, xarray, rioxarray, rasterio
"""
from pathlib import Path
import math
import numpy as np
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from pyproj import Transformer
from rasterio.windows import from_bounds


def spatial(data):
    """Standardise internal raster dimensions without inventing a projected CRS."""
    rename = {a: b for a, b in (("lat", "y"), ("latitude", "y"),
                               ("lon", "x"), ("longitude", "x")) if a in data.dims}
    geographic = bool(rename)
    data = data.rename(rename) if rename else data
    if not {"x", "y"}.issubset(data.dims):
        raise ValueError("Raster requires x/y or latitude/longitude dimensions")
    data = data.rio.set_spatial_dims(x_dim="x", y_dim="y")
    if data.rio.crs is None:
        if not geographic:
            raise ValueError("Projected x/y inputs require explicit CRS metadata")
        data = data.rio.write_crs("EPSG:4326")
    return data


def open_template(path, variable=None):
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rioxarray.open_rasterio(path, masked=True) as source:
            source = spatial(source)
            coords = {k: source[k].values for k in ("x", "y")}
            transform, crs = source.rio.transform(), source.rio.crs
    else:
        with xr.open_dataset(path, decode_coords="all") as dataset:
            if variable is None:
                variable = next((k for k, da in dataset.data_vars.items()
                                 if any(d in da.dims for d in ("x", "lon", "longitude"))), None)
            if variable is None:
                raise ValueError("Template file has no spatial data variable")
            source = spatial(dataset[variable])
            coords = {k: source[k].values for k in ("x", "y")}
            transform, crs = source.rio.transform(), source.rio.crs
    if transform.b or transform.d:
        raise ValueError("Rotated rasters require an explicit north-up processing template")
    template = xr.DataArray(np.zeros((len(coords['y']), len(coords['x'])), dtype='uint8'),
                            dims=("y", "x"), coords=coords)
    return template.rio.write_crs(crs).rio.write_transform(transform)


def subset_template(template, extent, *, extent_crs="EPSG:4326"):
    """Cover an explicitly referenced AOI with complete source pixels.

    Passing projected bounds directly avoids enlarging a metric rectangle by
    converting its geographic bounding box back to the native projected grid.
    """
    template = spatial(template)
    bounds = transform_bounds(extent_crs, template.rio.crs, *extent, densify_pts=21)
    window = from_bounds(*bounds, transform=template.rio.transform())
    x0, y0 = math.floor(window.col_off+1e-7), math.floor(window.row_off+1e-7)
    x1 = math.ceil(window.col_off+window.width-1e-7)
    y1 = math.ceil(window.row_off+window.height-1e-7)
    if x0 < 0 or y0 < 0 or x1 > template.sizes['x'] or y1 > template.sizes['y']:
        raise ValueError("Template does not cover complete AOI pixels; prepare/download a padded extent")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("AOI does not select any template pixels")
    return template.isel(x=slice(x0, x1), y=slice(y0, y1))


def intersecting_cells(template, bounds, bounds_crs, *, edge_points=65):
    """Select complete cells touched by a densified, transformed AOI rectangle.

    The returned mask admits whole cells, with their original areas. It does
    not weight observations by the fraction of a cell inside the AOI. A
    bounding raster window may contain corner cells outside a projected AOI;
    those cells are excluded here before soil and forcing support selection.
    """
    from rasterio.features import geometry_mask
    if not isinstance(edge_points, int) or edge_points < 2:
        raise ValueError('At least two points per AOI edge required')
    values = np.asarray(bounds, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError('AOI requires four finite ordered bounds')
    target = spatial(template)
    left, bottom, right, top = values
    corners = [(left,bottom),(right,bottom),(right,top),(left,top),(left,bottom)]
    points = np.concatenate([np.linspace(first,last,edge_points,endpoint=False)
                             for first,last in zip(corners[:-1],corners[1:])])
    transformer = Transformer.from_crs(bounds_crs,target.rio.crs,always_xy=True)
    x,y = transformer.transform(points[:,0],points[:,1])
    ring = list(zip(x,y))
    ring.append(ring[0])
    if not np.isfinite(np.asarray(ring)).all():
        raise ValueError('AOI transformation produced nonfinite coordinates')
    mask = geometry_mask([dict(type='Polygon',coordinates=[ring])],out_shape=target.shape,
        transform=target.rio.transform(),all_touched=True,invert=True)
    if not mask.any():
        raise ValueError('AOI does not intersect any target cells')
    output = target.copy(data=mask).rename('aoi_support')
    output.attrs.update(selection='Whole cells intersecting the configured AOI; boundary touches included',
                        aoi_crs=str(bounds_crs),aoi_bounds=values.tolist(),edge_points=edge_points)
    return output


def align_to_grid(data, template, resampling="bilinear", minimum_coverage=1.0):
    """Align values with a separate finite-support warp; retain missing coverage."""
    source, target = spatial(data), spatial(template)
    original_attrs = dict(data.attrs)
    # An integer-window crop is not a reprojection. Preserve reference values
    # and categorical flags exactly, including when the source is padded.
    if source.rio.crs == target.rio.crs:
        indices = {k: source.indexes[k].get_indexer(target[k].values) for k in ('x', 'y')}
        if all(np.all(v >= 0) for v in indices.values()):
            source = source.isel(**indices)
    same = source.rio.crs == target.rio.crs and all(
        np.array_equal(source[k].values, target[k].values) for k in ('x', 'y'))
    if same:
        result = source.copy(deep=False)
    else:
        method = Resampling[resampling] if isinstance(resampling, str) else resampling
        # Time/soil layers remain independent bands. No interpolation across time.
        result = source.astype('float64').rio.write_nodata(np.nan).rio.reproject_match(target, resampling=method)
        valid = np.isfinite(source).astype('float32').rio.write_nodata(None)
        support = valid.rio.reproject_match(target, resampling=method, nodata=0)
        result = result.where(support >= minimum_coverage-1e-6)
        if method == Resampling.average and minimum_coverage == 1.0:
            # GDAL normalises averages over available source pixels, even if a
            # target cell extends outside the source raster. Exclude these edge
            # cells instead of reporting a partial-footprint mean as a full cell.
            x, y = np.meshgrid(target.x.values, target.y.values)
            rx, ry = target.rio.resolution()
            transform = Transformer.from_crs(target.rio.crs,source.rio.crs,always_xy=True)
            inverse = ~source.rio.transform()
            complete = np.ones(x.shape,dtype=bool)
            for dx,dy in ((-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5),(0,-.5),(0,.5),(-.5,0),(.5,0)):
                sx,sy = transform.transform(x+dx*rx,y+dy*ry)
                columns,rows = inverse*(sx,sy)
                complete &= ((columns >= -1e-6) & (columns <= source.sizes['x']+1e-6)
                             & (rows >= -1e-6) & (rows <= source.sizes['y']+1e-6))
            result = result.where(xr.DataArray(complete,dims=('y','x'),coords={'y':target.y,'x':target.x}))
    result = result.assign_coords(x=target.x, y=target.y)
    result.attrs.update(original_attrs)
    result.attrs['spatial_resampling'] = 'none; exact grid' if same else str(resampling)
    return result.rio.write_crs(target.rio.crs).rio.write_transform(target.rio.transform())
