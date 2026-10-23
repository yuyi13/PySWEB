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


def prepare_auxiliary_sources(plan):
    """Apply the executed regional source recipe without calibration or fitting."""
    import ee
    import pandas as pd
    import rasterio
    from rasterio.warp import transform_bounds
    from pysweb.io.acquisition import acquire_monthly_meteorology, unpack_monthly_meteorology, _download
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
        _download(dem, r['dem_file'], ['elevation'], extent, dict(asset='NASA/NASADEM_HGT/001',
            purpose='native elevation covering complete exported meteorology cells'))
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
