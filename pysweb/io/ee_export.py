#!/usr/bin/env python3
"""
Script: ee_export.py
Objective: Download one explicit Earth Engine image onto a specified grid with auditable bands and atomic output.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Earth Engine image, download parameters, band names, output path and provenance.
Outputs: Validated GeoTIFF and returned checksum/grid/source metadata.
Usage: from pysweb.io.ee_export import download_geotiff
Dependencies: rasterio, existing PySWEB downloader, authenticated Earth Engine session.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import rasterio
from pysweb.io.native_grid import native_projection, export_parameters, verify_native_grid, region_extent
from pysweb.io.gee_downloader import _retry_download


def download_geotiff(image, path, *, parameters, bands, source, nodata=None, preserve_native_grid=True):
    """Download a bounded image without compositing or changing QA values.

    ``parameters`` must supply an explicit region, CRS and either scale or
    affine grid. Requests exceeding the EE direct-download limit must be
    tiled by the caller. Native-grid validation is the default; only explicit
    legacy reproductions may set preserve_native_grid=False. Integer QA zero remains valid unless explicitly
    overridden. Never persist the signed download URL.
    """
    if not bands or len(set(bands)) != len(bands):
        raise ValueError('bands must be nonempty and unique')
    if 'region' not in parameters or 'crs' not in parameters:
        raise ValueError('Explicit region and CRS are required')
    transforms = [key for key in ('crs_transform','crsTransform') if key in parameters]
    if len(transforms)>1 or bool(transforms)==('scale' in parameters):
        raise ValueError('Specify exactly one of scale or affine transform')
    if nodata == 0 and any(b in ('QA_PIXEL','QA_RADSAT') for b in bands):
        raise ValueError('Zero is valid QA; do not declare QA zero nodata')
    projection = native_projection(image,bands) if preserve_native_grid else None
    if projection is not None:
        if 'scale' in parameters:
            raise ValueError('Native export requires the source affine transform, not scale')
        supplied = parameters[transforms[0]]
        import numpy as np
        if rasterio.crs.CRS.from_user_input(parameters['crs']) != rasterio.crs.CRS.from_user_input(projection['crs']) or not np.allclose(supplied,projection['transform'],rtol=0,atol=1e-10):
            raise ValueError('Requested grid differs from native source grid')
        extent = region_extent(parameters['region'])
        parameters = {key:value for key,value in parameters.items() if key not in ('crs','crs_transform','crsTransform','region','dimensions')}
        parameters.update(export_parameters(projection,extent))
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+'.partial')
    url=image.select(bands).getDownloadURL({**parameters,'format':'GEO_TIFF','filePerBand':False})
    _retry_download(url,str(partial),max_retries=2)
    with rasterio.open(partial,'r+') as ds:
        if ds.count != len(bands):
            raise ValueError(f'Expected {len(bands)} bands, received {ds.count}')
        if str(ds.crs) != str(parameters['crs']):
            raise ValueError('Downloaded CRS differs from requested CRS')
        if projection is not None:
            verify_native_grid(ds,projection)
        ds.descriptions=tuple(bands)
        ds.nodata=nodata
        ds.update_tags(source_json=json.dumps(source),download_contract='exact image; no implicit daily composite')
        grid={'crs':str(ds.crs),'transform':list(ds.transform),'shape':[ds.height,ds.width],'dtype':ds.dtypes[0]}
    h=hashlib.sha256()
    with partial.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            h.update(chunk)
    partial.replace(path)
    return {'path':str(path),'native_projection':projection,'source':source,'bands':list(bands),'sha256':h.hexdigest(),'bytes':path.stat().st_size,'grid':grid,'retrieved_utc':datetime.now(timezone.utc).isoformat()}
