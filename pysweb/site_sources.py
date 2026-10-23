"""
Script: site_sources.py
Objective: Prepare missing site inputs with the established native acquisition APIs.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-23
Last updated: 2026-10-23
Inputs: Explicit site plan, source catalogue query and existing Earth Engine credentials.
Outputs: Frozen scene catalogue, native reference/soil/met inputs and restart receipts.
Usage: Called by the acquire stage of workflows/run_site.py.
Dependencies: pysweb, earthengine-api, pandas, rasterio; writable temporary directory.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import os
import tempfile

from pysweb.io.provenance import atomic_json, sha256


def ensure_catalogue(plan):
    """Freeze a bounded metadata query within this site's acquisition job."""
    cfg = plan['config_values']
    path = Path(cfg['landsat']['manifest_file'])
    query = dict(centre=cfg['run']['center'], wrs=cfg['landsat']['wrs'],
                 sensors=cfg['run']['sensors'], start=plan.get('catalogue_start', '2013-03-18'),
                 end_exclusive=(datetime.fromisoformat(cfg['run']['end'])+timedelta(days=1)).date().isoformat(),
                 processing_level='L2SP')
    if path.exists():
        catalogue = json.loads(path.read_text())
        if catalogue.get('query') != query:
            raise ValueError('Frozen scene query differs from requested site')
    else:
        import ee
        ee.Initialize(project=cfg['run']['gee_project'])
        ee.data.setDeadline(180000)
        dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
        support = dem.mask().unmask(0, False).reduceRegion(
            reducer=ee.Reducer.first(), geometry=ee.Geometry.Point(query['centre']),
            crs=dem.projection(), scale=30, maxPixels=16).getInfo()
        available = support.get('elevation') == 1
        atomic_json(Path(plan['base'])/'source_coverage.json', dict(
            passed=available, asset='NASA/NASADEM_HGT/001', location=query['centre'],
            native_dem_valid=support.get('elevation'),
            scope='Native model-centre mask; complete grids checked in later source/retrieval stages'))
        if not available:
            raise ValueError('Baseline NASADEM has no native support at the model centre')
        properties = ['system:id', 'system:index', 'system:time_start', 'LANDSAT_PRODUCT_ID',
                      'WRS_PATH', 'WRS_ROW', 'CLOUD_COVER', 'PROCESSING_LEVEL']
        packet = {}
        for sensor in query['sensors']:
            collection = (ee.ImageCollection(f'LANDSAT/{sensor}/C02/T1_L2')
                .filterBounds(ee.Geometry.Point(query['centre']))
                .filterDate(query['start'], query['end_exclusive'])
                .filter(ee.Filter.eq('WRS_PATH', query['wrs'][0]))
                .filter(ee.Filter.eq('WRS_ROW', query['wrs'][1]))
                .filter(ee.Filter.eq('PROCESSING_LEVEL', 'L2SP')))
            packet[sensor] = collection.map(lambda image: image.set(
                'inventory_record', image.toDictionary(properties))).aggregate_array('inventory_record')
        metadata = ee.Dictionary(packet).getInfo()
        scenes = []
        for sensor, records in metadata.items():
            for item in records:
                day = datetime.fromtimestamp(item['system:time_start']/1000, timezone.utc).date().isoformat()
                wrs_path, wrs_row = query['wrs']
                filename = f"{sensor}_{wrs_path:03d}{wrs_row:03d}_{day.replace('-', '')}.tif"
                scenes.append(dict(scene_id=item.get('system:id') or f"LANDSAT/{sensor}/C02/T1_L2/{item['system:index']}",
                    product_id=item['LANDSAT_PRODUCT_ID'], sensor=sensor, date=day,
                    path=wrs_path, row=wrs_row, cloud_cover=item['CLOUD_COVER'], source_catalogue=item,
                    file=str(Path(plan['base'])/'raw/landsat'/filename)))
        scenes.sort(key=lambda row: (row['date'], row['cloud_cover'], row['scene_id']))
        catalogue = dict(identity=plan['site_key'], query=query, scenes=scenes,
                         job_id=os.environ.get('PBS_JOBID'), acquired_at=datetime.now(timezone.utc).isoformat())
        if not scenes or len({row['file'] for row in scenes}) != len(scenes):
            raise ValueError('Empty or conflicting scene catalogue')
        atomic_json(path, catalogue)
    scenes = catalogue['scenes']
    if not scenes or any(not any(row['date'].startswith(str(year)) for row in scenes)
                         for year in range(int(cfg['run']['start'][:4]), int(cfg['run']['end'][:4])+1)):
        raise ValueError('Scene catalogue does not cover every simulation year')
    cfg['forcing']['retrieval']['template_file'] = scenes[0]['file']
    return path


def cached_step(plan, name, contract, operation):
    """Reuse only exact validated inputs/outputs, including after a partial job."""
    from pysweb.site_workflow import file_records, verified_records
    path = Path(plan['base'])/'source_receipts'/f'{name}.json'
    identity = dict(contract=contract, implementation=sha256(__file__))
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get('identity') == identity and verified_records(previous['outputs']):
            print(f'{name}: reuse verified source receipt', flush=True)
            return list(map(Path, previous['outputs']))
    outputs = operation()
    atomic_json(path, dict(passed=True, identity=identity, outputs=file_records(outputs),
                          job_id=os.environ.get('PBS_JOBID')))
    return outputs


def assemble_dem_tiles(paths, projection, extent, destination):
    """Assemble integer native windows without resampling or filling source gaps."""
    import numpy as np
    import rasterio
    from affine import Affine
    from pysweb.io.native_grid import export_parameters, verify_native_grid
    parameters = export_parameters(projection, extent)
    transform = Affine(*parameters['crs_transform'])
    width, height = parameters['dimensions']
    with rasterio.open(paths[0]) as first:
        dtype = first.dtypes[0]
    values = np.zeros((height, width), dtype=dtype)
    covered = np.zeros((height, width), dtype=bool)
    for path in paths:
        with rasterio.open(path) as source:
            verify_native_grid(source, projection)
            if source.count != 1 or source.dtypes != (dtype,):
                raise ValueError('DEM tile type/bands differ')
            offset = ~transform * source.transform
            col, row = round(offset.c), round(offset.f)
            if min(row, col) < 0 or row+source.height > height or col+source.width > width:
                raise ValueError('DEM tile outside native window')
            target = values[row:row+source.height, col:col+source.width]
            seen = covered[row:row+source.height, col:col+source.width]
            data = source.read(1)
            if not np.array_equal(target[seen], data[seen], equal_nan=True):
                raise ValueError('DEM tile overlap differs')
            target[:] = data
            seen[:] = True
    if not covered.all():
        raise ValueError('DEM assembly has missing source windows')
    destination = Path(destination)
    temporary = destination.with_suffix('.tif.partial')
    with rasterio.open(temporary, 'w', driver='GTiff', width=width, height=height,
                       count=1, dtype=dtype, crs=projection['crs'], transform=transform,
                       nodata=None, compress='deflate') as output:
        output.write(values, 1)
        output.descriptions = ('elevation',)
    with rasterio.open(temporary) as output:
        verify_native_grid(output, projection)
        np.testing.assert_array_equal(output.read(1), values)
    temporary.replace(destination)


def download_dem(plan, image, extent):
    """Bound the unchanged NASADEM export to four exact native-grid requests."""
    from pysweb.io.acquisition import _download
    destination = Path(plan['config_values']['forcing']['retrieval']['dem_file'])
    receipt = destination.with_suffix('.tif.native.json')
    contract = dict(asset='NASA/NASADEM_HGT/001', extent=extent,
                    transport='four geographic tiles; exact native assembly')
    if receipt.exists() and destination.exists():
        previous = json.loads(receipt.read_text())
        if previous.get('contract') == contract and previous.get('sha256') == sha256(destination):
            return
    tiles = []
    left, bottom, right, top = extent
    xs, ys = [left, (left+right)/2, right], [bottom, (bottom+top)/2, top]
    for row in range(2):
        for col in range(2):
            path = destination.parent/'nasadem_tiles'/f'r{row}_c{col}.tif'
            bounds = [xs[col], ys[row], xs[col+1], ys[row+1]]
            tiles.append(_download(image, path, ['elevation'], bounds,
                dict(asset='NASA/NASADEM_HGT/001', purpose='native elevation covering complete exported meteorology cells')))
    projection = tiles[0]['native_projection']
    if any(t['native_projection'] != projection for t in tiles):
        raise ValueError('DEM tile native projections differ')
    assemble_dem_tiles([t['path'] for t in tiles], projection, extent, destination)
    atomic_json(receipt, dict(passed=True, contract=contract, tiles=tiles,
                             native_projection=projection, sha256=sha256(destination)))


def prepare_auxiliary_sources(plan):
    """Apply the executed regional source recipe without calibration or fitting."""
    import ee
    import pandas as pd
    import rasterio
    from rasterio.warp import transform_bounds
    from pysweb.io.acquisition import acquire_monthly_meteorology, unpack_monthly_meteorology
    from pysweb.io.grids import open_template, subset_template, centered_template
    from pysweb.met.era5land.stack import stack_era5land_daily_inputs
    from pysweb.soil.openlandmap import _load_openlandmap_predictors
    from pysweb.swb.preprocess import _load_reference_ssm, write_dataarray, GSSM_ASSET
    from pysweb.workflow_native import buffered_extent
    cfg = plan['config_values']
    r = cfg['forcing']['retrieval']
    project = cfg['run']['gee_project']
    ee.Initialize(project=project)
    ee.data.setDeadline(180000)
    reference = Path(cfg['soil']['reference_file'])
    reference_extent = buffered_extent(cfg['run']['extent'], 2500)
    def reference_inputs():
        values = _load_reference_ssm(extent=tuple(reference_extent),
            dates=pd.date_range('2011-01-01', '2020-12-31'), reference_ssm_asset=GSSM_ASSET,
            gee_project=project)
        write_dataarray(values, reference)
        return [reference]
    reference_outputs = cached_step(plan, 'gssm', dict(asset=GSSM_ASSET, extent=reference_extent,
        start='2011-01-01', end='2020-12-31'), reference_inputs)
    def meteorology():
        raw = Path(r['meteorology_raw_dir'])
        monthly = acquire_monthly_meteorology(raw, r['meteorology_extent'], r['climatology_start'],
                                              r['meteorology_end'], gee_project=project)
        with rasterio.open(raw/next(iter(sorted(monthly['files'])))) as source:
            extent = list(transform_bounds(source.crs, 'EPSG:4326', *source.bounds, densify_pts=21))
        dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
        download_dem(plan, dem, extent)
        # Temporary daily expansion is bounded to this job/local invocation.
        with tempfile.TemporaryDirectory(prefix='sweb_met_') as temporary:
            daily = Path(temporary)/'daily'
            unpack_monthly_meteorology(raw, daily)
            stack_era5land_daily_inputs(daily, r['dem_file'], r['climatology_start'],
                                       r['meteorology_end'], r['meteorology_stack_dir'])
        return [Path(r['dem_file']), *map(Path, r['meteorology_files'].values()),
                *[path for path in raw.rglob('*') if path.is_file()]]
    met_outputs = cached_step(plan, 'meteorology', dict(extent=r['meteorology_extent'],
        start=r['climatology_start'], end=r['meteorology_end'], dem='NASA/NASADEM_HGT/001'), meteorology)
    coarse = subset_template(open_template(reference, 'reference_ssm'), cfg['run']['extent'])
    fine = centered_template(open_template(r['template_file']), cfg['run']['center'], cfg['run']['chip_size_m'])
    bounds = [list(transform_bounds(grid.rio.crs, 'EPSG:4326', *grid.rio.bounds(), densify_pts=21))
              for grid in (coarse, fine)]
    def soil_inputs():
        cache = Path(plan['retained_soil'])
        for left, bottom, right, top in bounds:
            _load_openlandmap_predictors((left-.01, bottom-.01, right+.01, top+.01), project, cache_dir=cache)
        return list(cache.glob('*.nc'))
    soil_outputs = cached_step(plan, 'native_soil', dict(bounds=bounds, margin_degrees=.01), soil_inputs)
    return reference_outputs+met_outputs+soil_outputs
