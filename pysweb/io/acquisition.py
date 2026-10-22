"""
Script: acquisition.py
Objective: Acquire exact native Landsat scenes and compact native meteorology with restart receipts.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-22
Inputs: Audited scene manifest, geographic envelope and an existing Earth Engine project.
Outputs: Original-grid rasters, explicit L7 optical masks and source metadata ledgers.
Usage: from pysweb.io.acquisition import acquire_landsat_manifest, acquire_monthly_meteorology
Dependencies: earthengine-api, rasterio, pandas, pysweb; existing authentication.
"""
from pathlib import Path
import json
import pandas as pd
from pysweb.io.provenance import atomic_json, sha256
from pysweb.io.native_grid import native_projection, export_parameters
from pysweb.io.ee_export import download_geotiff


def _download(image, path, bands, extent, source):
    import ee
    path = Path(path)
    receipt = path.with_suffix(path.suffix+'.native.json')
    previous = json.loads(receipt.read_text()) if receipt.exists() else {}
    contract = dict(source=source, extent=list(extent), bands=list(bands))
    if previous.get('contract') == contract and path.exists() and previous.get('sha256') == sha256(path):
        return previous
    projection = native_projection(image, bands)
    region = ee.Geometry.Rectangle(extent, proj='EPSG:4326', geodesic=False)
    record = download_geotiff(image,path,parameters={'region':region,**export_parameters(projection)},
                              bands=bands,source=source)
    record['contract'] = contract
    atomic_json(receipt,record)
    return record


def acquire_landsat_manifest(manifest_file, extent, *, gee_project, sample=False):
    """Honor audited product identifiers; acquire no unlisted sensor/date/WRS."""
    import ee
    from pysweb.ssebop.quality import landsat_c2_l2_bands
    ee.Initialize(project=gee_project)
    ee.data.setDeadline(180000)
    manifest_file = Path(manifest_file)
    catalogue = json.loads(manifest_file.read_text())
    records = catalogue['scenes']
    if sample:
        records = [min((r for r in records if r['sensor']==sensor),
                      key=lambda r:(r['cloud_cover'],r['date'],r['scene_id']))
                   for sensor in sorted({r['sensor'] for r in records})]
    ledger_path = manifest_file.with_name(manifest_file.stem+('_sample' if sample else '_downloads')+'.json')
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(scenes=[], files={})
    ledger.update(catalogue_sha256=sha256(manifest_file), extent=list(extent), complete=False)
    keys = ['LANDSAT_PRODUCT_ID','WRS_PATH','WRS_ROW','PROCESSING_LEVEL','system:time_start']
    # Fetch only five immutable properties per listed scene in one bounded
    # metadata request. Pixel data remain in explicit raster downloads.
    metadata = ee.Dictionary({r['scene_id']: ee.Image(r['scene_id']).toDictionary(keys)
                              for r in records}).getInfo()
    for record in records:
        windows = catalogue.get('sensor_windows', {})
        if record['sensor'] in windows:
            first,last=windows[record['sensor']]
            if not first <= record['date'] <= last:
                raise ValueError('Catalogue scene violates configured sensor window')
        image = ee.Image(record['scene_id'])
        properties = metadata[record['scene_id']]
        if properties['PROCESSING_LEVEL'] != 'L2SP' or properties['LANDSAT_PRODUCT_ID'] != record['product_id']:
            raise ValueError('Immutable Landsat product identity changed')
        if (properties['WRS_PATH'],properties['WRS_ROW']) != (record['path'],record['row']):
            raise ValueError('Landsat WRS differs from audited manifest')
        day = pd.Timestamp(properties['system:time_start'],unit='ms').strftime('%Y-%m-%d')
        if day != record['date']:
            raise ValueError('Landsat date differs from audited manifest')
        bands = [*landsat_c2_l2_bands(record['sensor']),'QA_PIXEL','QA_RADSAT','ST_QA']
        entry = _download(image.toUint16(),record['file'],bands,extent,{**record,'live_metadata':properties})
        ledger['files'][record['file']] = entry
        if record['sensor']=='LE07':
            projection=image.select('SR_B3').projection()
            footprint=ee.Image.constant(1).clip(image.geometry()).unmask(0).setDefaultProjection(projection)
            absent=image.select('SR_B3').mask().unmask(0).eq(0).And(image.select('SR_B4').mask().unmask(0).eq(0))
            masks=absent.And(footprint).rename('scan_gap_candidate').addBands(footprint.rename('source_footprint')).toUint8()
            entry=_download(masks,record['gap_mask_file'],['scan_gap_candidate','source_footprint'],extent,
                dict(scene_id=record['scene_id'],recipe='source optical mask absence within original footprint',source_projection=entry['native_projection']))
            ledger['files'][record['gap_mask_file']] = entry
        if record['scene_id'] not in ledger['scenes']:
            ledger['scenes'].append(record['scene_id'])
        atomic_json(ledger_path,ledger)
    ledger['complete'] = len(ledger['scenes']) == len(records)
    atomic_json(ledger_path,ledger)
    return ledger


def acquire_monthly_meteorology(output_dir, extent, start, end, *, gee_project):
    """Store all original daily bands in monthly containers on the source grid."""
    import ee
    from pysweb.met.era5land.download import ERA5LAND_BANDS
    ee.Initialize(project=gee_project)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    first,last=pd.Timestamp(start),pd.Timestamp(end)
    ledger_path=output_dir/'downloads.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(files={})
    collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate(
        str(first.date()), str((last+pd.Timedelta(days=1)).date())).sort('system:time_start')
    native_dates = collection.aggregate_array('system:time_start').getInfo()
    actual = pd.DatetimeIndex(pd.to_datetime(native_dates,unit='ms')).normalize()
    if not actual.equals(pd.date_range(first,last)):
        raise ValueError('Native meteorology calendar incomplete or duplicated')
    for month in pd.period_range(first,last,freq='M'):
        left=max(first,month.start_time.normalize())
        right=min(last,month.end_time.normalize())
        dates=pd.date_range(left,right)
        images=collection.filterDate(str(left.date()),str((right+pd.Timedelta(days=1)).date()))
        bands=[f'{date:%Y-%m-%d}_{name}' for date in dates for name in ERA5LAND_BANDS]
        # toBands uses collection element IDs even when a mapped image sets
        # system:index. Rename the verified, sorted stack explicitly instead.
        stack=images.select(ERA5LAND_BANDS).toBands().rename(bands)
        path=output_dir/f'era5_{month}.tif'
        ledger['files'][path.name]=_download(stack,path,bands,extent,
            dict(collection='ECMWF/ERA5_LAND/DAILY_AGGR',start=str(left.date()),end=str(right.date()),
                 recipe='original daily bands; no temporal reducer'))
        atomic_json(ledger_path,ledger)
    ledger.update(complete=True,start=str(first.date()),end=str(last.date()),extent=list(extent))
    atomic_json(ledger_path,ledger)
    return ledger


def unpack_monthly_meteorology(monthly_dir, daily_dir):
    """Expand compact raw containers into job-local daily files for maintained ET preparation."""
    import rasterio
    from pysweb.met.era5land.download import ERA5LAND_BANDS
    monthly_dir,daily_dir=Path(monthly_dir),Path(daily_dir)
    daily_dir.mkdir(parents=True,exist_ok=True)
    ledger=json.loads((monthly_dir/'downloads.json').read_text())
    if not ledger.get('complete'):
        raise ValueError('Meteorology acquisition is incomplete')
    paths=[]
    for name,record in sorted(ledger['files'].items()):
        path=monthly_dir/name
        if sha256(path)!=record['sha256']:
            raise ValueError('Native meteorology source changed')
        with rasterio.open(path) as src:
            names=list(src.descriptions)
            dates=sorted({s[:10] for s in names})
            for date in dates:
                indices=[names.index(date+'_'+band)+1 for band in ERA5LAND_BANDS]
                profile=src.profile.copy()
                profile.update(count=len(indices))
                target=daily_dir/f'ERA5LandDaily_{date}.tif'
                with rasterio.open(target,'w',**profile) as dst:
                    dst.write(src.read(indices))
                    dst.descriptions=tuple(ERA5LAND_BANDS)
                paths.append(target)
    return paths


def prepare_native_sources(manifest_file, protocol, *, gee_project, job_directory, sample=False):
    """Acquire shared site sources and prepare native meteorology through maintained APIs."""
    import ee
    import rasterio
    from rasterio.warp import transform_bounds
    from pysweb.met.era5land.stack import stack_era5land_daily_inputs
    catalogue=acquire_landsat_manifest(manifest_file,protocol['acquisition_extent'],gee_project=gee_project,sample=sample)
    if sample:
        return catalogue
    landcover_file = protocol.get('landcover_file',
        str(Path(protocol['dem_file']).parent/'worldcover_v200_2021.tif'))
    if 'landcover_file' not in protocol or not Path(landcover_file).exists():
        extent = worldcover_extent(catalogue['files'], protocol['acquisition_extent'])
        acquire_worldcover(landcover_file, extent, gee_project=gee_project)
    raw_met=Path(protocol['meteorology_raw_dir'])
    acquire_monthly_meteorology(raw_met,protocol['meteorology_extent'],protocol['climatology_start'],
        protocol['meteorology_end'],gee_project=gee_project)
    monthly=json.loads((raw_met/'downloads.json').read_text())
    first=raw_met/next(iter(sorted(monthly['files'])))
    with rasterio.open(first) as src:
        extent=list(transform_bounds(src.crs,'EPSG:4326',*src.bounds,densify_pts=21))
    dem=ee.Image('NASA/NASADEM_HGT/001').select('elevation')
    _download(dem,protocol['dem_file'],['elevation'],extent,dict(asset='NASA/NASADEM_HGT/001',
        purpose='native elevation covering complete exported meteorology cells'))
    # Source containers stay persistent; daily expansions belong to job-local I/O.
    daily=Path(job_directory)/'era5_daily'
    unpack_monthly_meteorology(raw_met,daily)
    stack_era5land_daily_inputs(daily,protocol['dem_file'],protocol['climatology_start'],
        protocol['meteorology_end'],protocol['meteorology_stack_dir'])
    return dict(landsat=catalogue,meteorology=monthly,dem=protocol['dem_file'],landcover=landcover_file)


def worldcover_extent(scene_files, requested_extent):
    """Cover actual exported scene cells, including UTM bounding-box corners.

    Native exports snap the requested geographic box to a projected rectangle;
    its corners can extend outside that original box. Inspect metadata only.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    left, bottom, right, top = requested_extent
    for path in scene_files:
        with rasterio.open(path) as source:
            bounds = source.bounds
            rx, ry = map(abs, source.res)
            lo, bo, hi, to = transform_bounds(source.crs, 'EPSG:4326',
                bounds.left-rx, bounds.bottom-ry, bounds.right+rx, bounds.top+ry, densify_pts=65)
        left, bottom, right, top = min(left, lo), min(bottom, bo), max(right, hi), max(top, to)
    return [left, bottom, right, top]


def acquire_worldcover(output_file, extent, *, gee_project):
    """Acquire a native categorical WorldCover v200 chip with a hash receipt.

    This is a static 2021 classification, not a scene-year land-cover product.
    The extent must cover the complete Landsat reference region, not just AOI.
    """
    import ee
    from pysweb.ssebop.landcover import WORLDCOVER_ASSET
    ee.Initialize(project=gee_project)
    ee.data.setDeadline(180000)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    image = ee.ImageCollection(WORLDCOVER_ASSET).first().select('Map')
    return _download(image.toUint8(), output_file, ['Map'], extent,
        dict(collection=WORLDCOVER_ASSET, year=2021, band='Map',
             citation='Zanaga et al. (2022), doi:10.5281/zenodo.7254221', license='CC-BY-4.0'))
