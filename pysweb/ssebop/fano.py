"""
Script: fano.py
Objective: Estimate land-cover-aware FANO cold boundaries on fixed projected grids.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-22
Last updated: 2026-10-22
Inputs: Aligned Landsat temperature, NDVI, dT and ESA WorldCover classifications.
Outputs: Cold temperature, scene-fallback contribution and reference diagnostics.
Usage: from pysweb.ssebop.fano import worldcover_fano
Dependencies: numpy, scipy, xarray, rasterio, rioxarray, pyproj, pysweb
"""
import math

import numpy as np
import xarray as xr
from pyproj import CRS
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import uniform_filter

from pysweb.ssebop.core import LocalFanoConfig
from pysweb.ssebop.landcover import WORLDCOVER_CLASSES, worldcover_masks


FANO_VERSION = "worldcover-fano-v1"
TCOLD_METHODS = ("worldcover_fano", "legacy")


def validate_method(method, landcover):
    if method not in TCOLD_METHODS:
        raise ValueError(f"tcold_method must be one of {TCOLD_METHODS}")
    if method == "worldcover_fano" and landcover is None:
        raise ValueError("worldcover_fano requires ESA WorldCover landcover; run ssebop-prepare "
                         "or provide landcover/forcing.retrieval.landcover_file. "
                         "Use tcold_method='legacy' only to reproduce historical retrievals.")


def config_from_options(options):
    """Use the same tuning contract in direct and configured scene workflows."""
    from dataclasses import fields
    return LocalFanoConfig(**{field.name: options['tcold_'+field.name]
                              for field in fields(LocalFanoConfig)
                              if 'tcold_'+field.name in options})


def _grid(template, scale):
    """Anchor grids globally, so cropping a scene cannot shift the coarse cells."""
    left, bottom, right, top = template.rio.bounds()
    left = 15 + math.floor((left-15)/scale)*scale
    top = 15 + math.ceil((top-15)/scale)*scale
    width, height = math.ceil((right-left)/scale), math.ceil((top-bottom)/scale)
    transform = Affine(scale, 0, left, 0, -scale, top)
    return xr.DataArray(np.full((height, width), np.nan), dims=('y', 'x'),
        coords={'x': left+(np.arange(width)+.5)*scale,
                'y': top-(np.arange(height)+.5)*scale}).rio.write_crs(
                    template.rio.crs).rio.write_transform(transform)


def _warp(source, target, method='nearest'):
    output = np.full(target.shape, np.nan, dtype='float64')
    reproject(np.asarray(source, dtype='float64'), output,
        src_transform=source.rio.transform(), src_crs=source.rio.crs, src_nodata=np.nan,
        dst_transform=target.rio.transform(), dst_crs=target.rio.crs, dst_nodata=np.nan,
        resampling=Resampling[method], num_threads=1)
    return target.copy(data=output)


def _mean(data, radius):
    """Square focal mean, ignoring missing cells without extending edge values."""
    values = np.asarray(data, dtype='float64')
    finite = np.isfinite(values)
    size = 2*radius+1
    numerator = uniform_filter(np.where(finite, values, 0), size=size, mode='constant')
    denominator = uniform_filter(finite.astype(float), size=size, mode='constant')
    result = np.full(values.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 1e-12)
    return data.copy(data=result)


def worldcover_fano(lst, ndvi, dt, landcover, *, config=None):
    """OpenET-style FANO with a documented WorldCover crosswalk.

    Fine/coarse area means use all available raster pixels, rather than Earth
    Engine's bestEffort pyramid reductions. Categorical masks use nearest
    sampling. See docs/methods.md for crosswalk, water and scene-extent limits.
    """
    validate_method('worldcover_fano', landcover)
    config = config or LocalFanoConfig()
    scales = (config.fine_scale_m, config.coarse_scale_m, config.smooth_resolution_m)
    if (not all(np.isfinite(s) and s > 0 for s in scales)
            or config.coarse_scale_m < config.fine_scale_m):
        raise ValueError('FANO scales must be positive with coarse >= fine')
    radius = config.smooth_radius_pixels
    if not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError('smooth_radius_pixels must be a nonnegative integer')
    if (not np.isfinite(config.dt_coeff) or config.dt_coeff <= 0
            or not 0 <= config.anchor_ndvi_threshold <= config.high_ndvi_threshold <= 1):
        raise ValueError('Invalid FANO coefficient or NDVI thresholds')
    for field in (lst, ndvi, dt):
        if field.dims != ('y', 'x') or field.rio.crs != lst.rio.crs:
            raise ValueError('FANO requires aligned two-dimensional y/x inputs')
        xr.align(lst, field, join='exact')
    crs = CRS(lst.rio.crs)
    transform = lst.rio.transform()
    if (not crs.is_projected or any(axis.unit_conversion_factor != 1 for axis in crs.axis_info)
            or transform.b or transform.d or transform.a <= 0 or transform.e >= 0):
        raise ValueError('FANO requires a north-up projected grid in metres')
    if landcover.dims != ('y', 'x') or landcover.rio.crs is None:
        raise ValueError('WorldCover requires a georeferenced two-dimensional raster')
    cover = _warp(landcover, lst)
    valid = np.isfinite(lst) & np.isfinite(ndvi) & np.isfinite(dt) & (dt > 0)
    valid = valid & (ndvi >= -1) & (ndvi <= 1)
    # Unknown or absent cover is a missing input, never an implicit non-ag class.
    if bool((valid & ~cover.isin(WORLDCOVER_CLASSES)).any()):
        raise ValueError('WorldCover does not cover all valid scene pixels with recognised classes')
    ag, anomalous, water = worldcover_masks(cover)
    land = valid & ~water.astype(bool)
    fine, coarse, smooth = (_grid(lst, scale) for scale in scales)
    fine_ndvi = _warp(ndvi.where(land), fine, 'average')
    fine_lst = _warp(lst.where(land), fine, 'average')
    fine_dt = _warp(dt.where(land), fine, 'average')
    fine_tc = fine_lst-config.dt_coeff*fine_dt*(config.high_ndvi_threshold-fine_ndvi)*10

    # OpenET masks fine-resolution fields with native green agricultural
    # pixels before the coarse reduction. Nearest sampling retains class IDs.
    anchor = land & ag.astype(bool) & (ndvi >= config.anchor_ndvi_threshold)
    fine_anchor = _warp(anchor.astype(float), fine) == 1
    coarse_ndvi = _warp(fine_ndvi.where(fine_anchor), coarse, 'average')
    coarse_lst = _warp(fine_lst.where(fine_anchor), coarse, 'average')
    coarse_dt = _warp(dt.where(land), coarse, 'average')
    coarse_tc = coarse_lst-config.dt_coeff*coarse_dt*(config.high_ndvi_threshold-coarse_ndvi)*10
    regional = _mean(coarse_tc, 1)
    available = np.asarray(fine_tc)[np.isfinite(fine_tc)]
    scene_tc = float(available.mean()) if available.size else np.nan
    missing = regional.isnull()
    regional = _mean(regional.fillna(scene_tc), 1)
    # Propagate the fraction of the temperature estimate supplied by the
    # scene scalar through exactly the same second and final means.
    regional_fallback = _mean(missing.astype(float).where(np.isfinite(regional)), 1)
    ag_smooth = _warp(ag.astype(float), smooth) == 1
    green_smooth = _warp(fine_ndvi, smooth) >= config.anchor_ndvi_threshold
    use_regional = ag_smooth & ~green_smooth
    local = _warp(fine_tc, smooth)
    layered = xr.where(use_regional, _warp(regional, smooth), local)
    weight = xr.where(use_regional, _warp(regional_fallback, smooth), 0).where(np.isfinite(layered))
    cold = _warp(_mean(layered, radius), lst)
    fallback = _warp(_mean(weight, radius), lst).clip(0, 1)
    # Mirror FANO's negative-NDVI special case for shrub/bare/built classes.
    # Water is excluded from terrestrial ET, including when optional masking
    # is disabled by the caller; it cannot enter reference statistics.
    cold = xr.where(ndvi < 0, xr.where(anomalous.astype(bool), 250., lst), cold).where(land)
    fallback = xr.where(ndvi < 0, 0., fallback).where(land)
    result = xr.Dataset(dict(tcold_K=cold, cold_boundary_fallback_fraction=fallback,
        fano_reference_eligible=anchor.astype('uint8'), worldcover_class=cover))
    result.attrs.update(tcold_method='worldcover_fano', tcold_algorithm=FANO_VERSION,
        landcover_crosswalk='reference=30,40,90; anomalous=20,50,60; water=80',
        landcover_epoch='2021 (ESA WorldCover v200); static for all scene dates',
        tcold_scene_fallback_K=scene_tc)
    result.tcold_K.attrs.update(units='K', long_name='FANO cold boundary temperature')
    result.cold_boundary_fallback_fraction.attrs.update(units='1',
        long_name='Scene-scalar contribution to cold boundary after branch selection and smoothing')
    result.fano_reference_eligible.attrs.update(units='1',
        long_name='Valid native pixel with eligible WorldCover class and NDVI above anchor threshold')
    result.worldcover_class.attrs.update(source='ESA/WorldCover/v200', epoch=2021,
        spatial_resampling='nearest', long_name='ESA WorldCover class')
    return result.rio.write_crs(lst.rio.crs).rio.write_transform(transform)
