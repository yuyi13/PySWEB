#!/usr/bin/env python3
"""
Script: worldcover_tiles.py
Objective: Recover wide WorldCover acquisition using lossless native-grid tiles.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-22
Last updated: 2026-10-23
Inputs: Prepared WorldCover extent, existing Earth Engine project and frozen PySWEB APIs.
Outputs: Native categorical GeoTIFF, resumable tile receipts and exact assembly validation.
Usage: from pysweb.io.worldcover_tiles import acquire_worldcover_tiled
Dependencies: numpy, rasterio, affine, earthengine-api, pysweb.
"""
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import rasterio
from affine import Affine

from pysweb.io.acquisition import acquire_worldcover
from pysweb.io.native_grid import export_parameters, verify_native_grid
from pysweb.io.provenance import atomic_json, sha256
from pysweb.ssebop.landcover import WORLDCOVER_ASSET, WORLDCOVER_CLASSES


def assemble_tiles(paths, projection, extent, destination):
    """Copy integer pixel windows, rejecting holes, shifts or conflicting overlaps."""
    parameters = export_parameters(projection, extent)
    transform = Affine(*parameters['crs_transform'])
    width, height = parameters['dimensions']
    values = np.zeros((height, width), dtype='uint8')
    covered = np.zeros((height, width), dtype=bool)
    overlap_pixels = 0
    for path in paths:
        with rasterio.open(path) as source:
            verify_native_grid(source, projection)
            if source.count != 1 or source.dtypes != ('uint8',):
                raise ValueError('WorldCover tile must have one uint8 band')
            offset = ~transform * source.transform
            col, row = round(offset.c), round(offset.f)
            if row < 0 or col < 0 or row+source.height > height or col+source.width > width:
                raise ValueError('Tile extends outside the full native window')
            data = source.read(1)
            if not np.isin(data, (0, *WORLDCOVER_CLASSES)).all():
                raise ValueError('WorldCover contains unknown class codes')
            target = values[row:row+source.height, col:col+source.width]
            seen = covered[row:row+source.height, col:col+source.width]
            if not np.array_equal(target[seen], data[seen]):
                raise ValueError('Overlapping WorldCover source pixels disagree')
            overlap_pixels += int(seen.sum())
            target[:] = data
            seen[:] = True
    if not covered.all():
        raise ValueError('WorldCover assembly has missing pixels')
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix+'.partial')
    with rasterio.open(temporary, 'w', driver='GTiff', width=width, height=height,
                       count=1, dtype='uint8', crs=projection['crs'], transform=transform,
                       nodata=None, compress='deflate') as target:
        target.write(values, 1)
        target.descriptions = ('Map',)
        target.update_tags(collection=WORLDCOVER_ASSET, year=2021,
                           assembly='exact native pixels; no resampling')
    with rasterio.open(temporary) as target:
        grid = verify_native_grid(target, projection)
        if not np.array_equal(target.read(1), values):
            raise ValueError('WorldCover output differs from assembled source pixels')
    temporary.replace(destination)
    classes, counts = np.unique(values, return_counts=True)
    return dict(passed=True, complete_coverage=True, no_resampling=True,
                identical_overlaps=True, overlap_pixels_checked=overlap_pixels,
                class_counts={str(int(k)): int(v) for k, v in zip(classes, counts)},
                grid=grid)


def acquire_worldcover_tiled(output_file, extent, *, gee_project):
    """Use native acquisition for four bounded requests of the same image."""
    output_file = Path(output_file)
    receipt_file = output_file.with_suffix(output_file.suffix+'.native.json')
    source = dict(collection=WORLDCOVER_ASSET, year=2021, band='Map',
                  citation='Zanaga et al. (2022), doi:10.5281/zenodo.7254221',
                  license='CC-BY-4.0')
    contract = dict(source=source, extent=list(extent), bands=['Map'],
                    transport='four geographic tiles; exact native-grid assembly v1')
    if output_file.exists() and receipt_file.exists():
        previous = json.loads(receipt_file.read_text())
        if (previous.get('contract') == contract
                and previous.get('validation', {}).get('passed')
                and previous.get('sha256') == sha256(output_file)):
            print('Reusing verified WorldCover mosaic:', output_file, flush=True)
            return previous
    tile_dir = output_file.parent / (output_file.stem+'_tiles')
    tile_dir.mkdir(parents=True, exist_ok=True)
    left, bottom, right, top = extent
    xs = [left, (left+right)/2, right]
    ys = [bottom, (bottom+top)/2, top]
    tiles = []
    for row in range(2):
        for col in range(2):
            path = tile_dir / f'r{row}_c{col}.tif'
            bounds = [xs[col], ys[row], xs[col+1], ys[row+1]]
            print(f'WorldCover tile {len(tiles)+1}/4: {path.name}', flush=True)
            tiles.append(acquire_worldcover(path, bounds, gee_project=gee_project))
    projection = tiles[0]['native_projection']
    if any(tile['native_projection'] != projection for tile in tiles):
        raise ValueError('WorldCover tile source projections differ')
    validation = assemble_tiles([t['path'] for t in tiles], projection, extent, output_file)
    record = dict(path=str(output_file), source=source, contract=contract, bands=['Map'],
                  native_projection=projection, grid=validation['grid'],
                  sha256=sha256(output_file), bytes=output_file.stat().st_size,
                  retrieved_utc=datetime.now(timezone.utc).isoformat(),
                  tiles=tiles, validation=validation)
    atomic_json(receipt_file, record)
    print('WorldCover assembly validated:', validation['grid']['shape'], flush=True)
    return record
