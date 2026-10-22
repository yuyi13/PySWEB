"""
Script: workflow_native.py
Objective: Orchestrate native downloads, GSSM calibration and shared Landsat-grid products.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-22
Inputs: Validated workflow configuration and selected execution stages.
Outputs: Two final NetCDFs, coarse calibration inputs, parameters and grid checks.
Usage: pysweb workflow --config examples/workflow.toml
Dependencies: pandas, pyproj, rasterio, xarray, pysweb
"""
from pathlib import Path
import json
import re
import pandas as pd
from rasterio.warp import transform_bounds
from pysweb.contracts import month_bounds
from pysweb.io.grids import open_template, subset_template, centered_template
from pysweb.io.products import crop_product, assert_identical_grids
from pysweb.io.provenance import atomic_json
from pysweb.swb.spinup import enabled_spinup


def buffered_extent(extent, metres):
    """A download envelope only: actual grids always come from source metadata."""
    lon, lat = (extent[0]+extent[2])/2, (extent[1]+extent[3])/2
    if not -80 < lat < 84:
        raise ValueError('Landsat UTM workflow requires an AOI between 80 S and 84 N')
    zone = min(60, max(1, int((lon+180)//6)+1))
    crs = f'EPSG:{32600+zone if lat >= 0 else 32700+zone}'
    left,bottom,right,top = transform_bounds('EPSG:4326',crs,*extent,densify_pts=21)
    return list(transform_bounds(crs,'EPSG:4326',left-metres,bottom-metres,
                                 right+metres,top+metres,densify_pts=21))


def _cached_scene_files(directory, sensors, start, end, pattern='*.tif'):
    """Apply the selected sensors even when a reused cache contains older missions."""
    from pysweb.ssebop.api import list_landsat_files, filter_landsat_files_by_date, parse_date_from_filename
    files = filter_landsat_files_by_date(list_landsat_files(str(directory), pattern), f'{start} to {end}')
    ranked = []
    priority = ('LC09', 'LC08', 'LE07', 'LT05')
    for path in files:
        match = re.search(r'(?:^|_)(LC08|LC09|LE07|LT05)(?=_|\.)', Path(path).name)
        if not match:
            raise ValueError(f'Cannot identify Landsat sensor in {Path(path).name}; provide an explicit landsat.manifest_file')
        sensor = match.group(1)
        if sensor in sensors:
            ranked.append((parse_date_from_filename(path), priority.index(sensor), str(path)))
    selected = {}
    for date, _, path in sorted(ranked):
        selected.setdefault(date, path)
    if not selected:
        raise ValueError('No cached Landsat scenes match run.sensors and the requested dates')
    return list(selected.values())


def native_plan(config, stages):
    run, calibration = config['run'], config.get('calibration', {})
    start, end = pd.Timestamp(run['start']), pd.Timestamp(run['end'])
    modern_sensors = set(run.get('sensors', ['LC08', 'LC09'])) <= {'LC08', 'LC09'}
    if modern_sensors and start < pd.Timestamp('2013-01-01'):
        raise ValueError('Landsat 8/9 runs start in 2013 or later; explicitly select LT05/LE07 for earlier simulations')
    enabled = bool(calibration.get('enabled', False))
    default_scoring_start = pd.Timestamp('2014-01-01') if modern_sensors and start.year == 2013 else start
    first = pd.Timestamp(calibration.get('start', default_scoring_start))
    last = pd.Timestamp(calibration.get('end', end))
    if enabled and not start <= first <= last <= end:
        raise ValueError('Native calibration dates must lie within run.start/end; 2013 is initialization by default for Landsat 8/9. Set calibration.start explicitly to override')
    holdout = pd.Timestamp(run['holdout_start']) if run.get('holdout_start') else None
    if enabled and holdout is not None and last >= holdout:
        raise ValueError('Calibration must end before holdout_start')
    climate = config.get('ssebop', {})
    boundary = last if enabled else (holdout-pd.Timedelta(days=1) if holdout is not None else None)
    if boundary is not None and climate.get('climatology_end') and pd.Timestamp(climate['climatology_end']) > boundary:
        raise ValueError('SSEBop climatology cannot include dates after the development boundary')
    if boundary is not None and boundary < start:
        raise ValueError('The run must include development meteorology before the holdout boundary')
    if run.get('processing_buffer_m',14400) < 0:
        raise ValueError('processing_buffer_m must be non-negative')
    if calibration.get('sm_res') is not None:
        raise ValueError('Native calibration uses the GSSM affine grid; remove calibration.sm_res')
    if calibration.get('objective', 'cell') != 'cell':
        raise ValueError('Native calibration requires the area-weighted cell-day objective')
    if config.get('model',{}).get('nan_to_zero'):
        raise ValueError('Native workflow preserves missing forcing; nan_to_zero would change calibration/run semantics')
    spinup = enabled_spinup(config.get('spinup'))
    compact = bool(spinup and 'source_years' in spinup)
    compact_metadata = None
    if spinup:
        from pysweb.swb.spinup import cycle_indices
        if not boundary:
            raise ValueError('Configured spin-up requires an explicit development boundary')
        if compact:
            from pysweb.swb.spinup import source_block
            _, compact_metadata = source_block(pd.date_range(start, end), spinup)
            if not enabled and 'start' not in calibration:
                first = pd.Timestamp(compact_metadata['scoring_start_minimum'])
            if first < pd.Timestamp(compact_metadata['scoring_start_minimum']):
                raise ValueError('Scoring must begin after the retained spin-up source block')
            if pd.Timestamp(compact_metadata['scoring_start_minimum']) > boundary:
                raise ValueError('Spin-up source years must precede the development boundary')
        else:
            cycle_indices(pd.date_range(start, end), start, spinup['repeats'], spinup['source_year'], boundary)
        if not 0 < spinup['tolerance'] or not 0 < spinup['minimum_fraction'] <= 1:
            raise ValueError('Invalid spin-up acceptance thresholds')
        if spinup.get('convergence_mode', 'wet_dry') not in {'wet_dry', 'field_capacity_cycles'}:
            raise ValueError('Invalid spin-up convergence_mode')
        if spinup.get('convergence_action', 'require') not in {'require', 'record'}:
            raise ValueError('Invalid spin-up convergence_action')
        if run['state_timing'] != 'start':
            raise ValueError('Verified spin-up workflow requires start-of-day states')
    landsat = config.get('landsat', {})
    if landsat.get('gapfill') and (not boundary or pd.Timestamp(landsat['gapfill']['end']) > boundary):
        raise ValueError('L7 gap-fill donors must end within development')
    if config.get('forcing') and not boundary:
        raise ValueError('Forcing support requires a development boundary')
    analysis_start = first if enabled or compact else default_scoring_start
    warmup_end = min(end, analysis_start-pd.Timedelta(days=1))
    warmup = [str(start.date()), str(warmup_end.date())] if warmup_end >= start else None
    initialization = dict(initial_state='physical_field_capacity',
        repeated_spinup_enabled=bool(spinup),
        mode='repeated_source_block' if compact else ('repeated_year' if spinup else 'actual_forcing_prefix'),
        actual_start=str(start.date()), warmup_window=warmup,
        analysis_start=str(analysis_start.date()),
        actual_warmup_days=max(0, (warmup_end-start).days+1),
        artificial_warmup_days=(compact_metadata['artificial_days'] if compact
                               else (365*spinup['repeats'] if spinup else 0)),
        interpretation='Field-capacity assumption; actual prefix alone does not establish initial-state independence')
    if compact:
        initialization['repeated_source_block'] = compact_metadata
    if spinup:
        initialization['convergence_action'] = spinup.get('convergence_action', 'require')
        initialization['interpretation'] = (
            'Fixed-duration initialization experiment; inspect convergence receipts before interpreting results'
            if spinup.get('convergence_action') == 'record' else
            'Repeated forcing; convergence must pass the explicitly selected diagnostic')
    base = Path(run['root'])/run['name']
    return {'stages':list(stages),'run_directory':str(base), 'windows':[[run['start'],run['end']]],
            'calibration_scoring_window':[str(first.date()),str(last.date())] if enabled else None,
            'initialization_window':warmup,
            'initialization':initialization,
            'analysis_window':[str(analysis_start.date()),str(end.date())] if analysis_start <= end else None,
            'calibration_grid':'native GSSM; complete cells covering AOI',
            'output_grid':'selected Landsat native 30 m UTM; shared by SEB and SWB',
            'products':[str(base/'products'/name) for name in ('SSEBop_ET.nc','SWEB_RZSM.nc')],
            'config':config}


def run_native(config, stages, dry_run=False):
    import pysweb
    from pysweb.swb.preprocess import _load_reference_ssm, write_dataarray, GSSM_ASSET
    from pysweb.swb.paired import run_default_and_calibrated

    plan = native_plan(config, stages)
    spinup = enabled_spinup(config.get('spinup'))
    if dry_run:
        return plan
    run, soil = config['run'], config.get('soil', {'source':'openlandmap'})
    calibration = config.get('calibration', {})
    enabled = bool(calibration.get('enabled', False))
    base = Path(plan['run_directory'])
    prepared, internal, products = base/'inputs', base/'ssebop', base/'products'
    coarse, fine = base/'calibration_inputs', base/'run_inputs'
    base.mkdir(parents=True,exist_ok=True)
    atomic_json(base/'resolved-config.json',plan)
    start,end = run['start'],run['end']
    fit_start,fit_end = plan['calibration_scoring_window'] if enabled else (plan['initialization']['analysis_start'],end)
    boundary = fit_end if enabled else (str((pd.Timestamp(run['holdout_start'])-pd.Timedelta(days=1)).date()) if run.get('holdout_start') else None)
    reference = Path(soil.get('reference_file', prepared/'reference_ssm_native.nc'))
    extent = list(run['extent'])
    if enabled and any(stage in stages for stage in ('prepare','ssebop','preprocess')):
        if not reference.exists():
            if 'reference_file' in soil:
                raise FileNotFoundError(reference)
            reference_extent = buffered_extent(extent, 2500)
            data = _load_reference_ssm(extent=tuple(reference_extent),
                dates=pd.date_range(fit_start,fit_end),reference_ssm_asset=GSSM_ASSET,
                gee_project=run.get('gee_project',''))
            write_dataarray(data,reference)
        reference_grid = subset_template(open_template(reference,soil.get('reference_var','reference_ssm')),extent)
        extent = list(transform_bounds(reference_grid.rio.crs,'EPSG:4326',*reference_grid.rio.bounds(),densify_pts=21))
    download_extent = buffered_extent(extent,run.get('processing_buffer_m',14400))
    met = prepared/'met/era5land/stack'
    period = f'{start} to {end}'
    if 'prepare' in stages:
        retrieval = config.get('forcing', {}).get('retrieval')
        if retrieval and config.get('landsat'):
            import os
            from pysweb.io.acquisition import prepare_native_sources
            job_directory = os.environ.get('PBS_JOBFS') or os.environ.get('TMPDIR')
            if not job_directory:
                raise ValueError('Set TMPDIR for bounded temporary meteorology expansion')
            retrieval = dict(retrieval)
            if config.get('ssebop', {}).get('landcover'):
                retrieval['landcover_file'] = config['ssebop']['landcover']
            prepare_native_sources(config['landsat']['manifest_file'],retrieval,
                gee_project=run.get('gee_project',''),job_directory=job_directory)
        else:
            pysweb.ssebop.prepare_inputs(date_range=period,extent=download_extent,
                met_source='era5land',gee_project=run.get('gee_project',''),
                landsat_dir=str(prepared/'landsat'),met_raw_dir=str(prepared/'met/era5land/raw'),
                met_stack_dir=str(met),dem_dir=str(prepared/'dem'),sensors=run['sensors'],
                landcover_file=config.get('ssebop',{}).get('landcover'))
    seb_internal = internal/f'et_daily_ssebop_{start}_{end}.nc'
    seb_final = products/'SSEBop_ET.nc'
    if 'ssebop' in stages:
        options = {'lst_band':'auto','red_band':'SR_B4','nir_band':'SR_B5','ndvi_band':'ndvi',
                   'reference_et':str(met/f'et_tall_crop_daily_{start}_{end}.nc'),
                   'reference_et_var':'et_tall_crop','et_reference_type':'alfalfa'}
        options.update(config.get('ssebop',{}))
        if options.get('tcold_method', 'worldcover_fano') == 'worldcover_fano':
            retrieval = config.get('forcing', {}).get('retrieval', {})
            options.setdefault('landcover', retrieval.get('landcover_file',
                str(Path(retrieval.get('dem_file', prepared/'dem/nasadem.tif')).parent/'worldcover_v200_2021.tif')))
        if config.get('landsat'):
            from pysweb.ssebop.preprocessing import select_scenes, prepare_l7
            selection = config['landsat']
            ledger = json.loads(Path(selection['manifest_file']).read_text())
            records = ledger['scenes'] if isinstance(ledger, dict) else ledger
            selected = select_scenes(records, run['sensors'], start, end,
                sensor_windows=selection.get('sensor_windows'), wrs=selection.get('wrs'))
            options['scene_files'] = [r['file'] for r in selected]
            if any(r['sensor'] == 'LE07' for r in selected):
                if not selection.get('gapfill'):
                    raise ValueError('Configured L7 scenes require a verified gapfill protocol')
                rule = selection['gapfill']
                donors = select_scenes(records, ['LE07'], rule['start'], rule['end'],
                    sensor_windows=selection.get('sensor_windows'), wrs=selection.get('wrs'))
                options['gapfill_products'] = prepare_l7(donors, base/'l7_preprocessed', rule, development_end=boundary)
            atomic_json(base/'selected-scenes.json', dict(scenes=selected, rule='cloud_cover then immutable scene_id'))
        else:
            options['scene_files'] = _cached_scene_files(prepared/'landsat', run['sensors'], start, end,
                options.get('landsat_pattern', '*.tif'))
            atomic_json(base/'selected-scenes.json', dict(files=options['scene_files'],
                sensors=run['sensors'], rule='sensor priority then immutable path; one scene per date'))
        if config.get('forcing', {}).get('reconstruction'):
            options['reconstruction_protocol'] = config['forcing']['reconstruction']
        if boundary:
            options['development_end'] = boundary
            options.setdefault('climatology_start', start)
            options.setdefault('climatology_end', boundary)
        retrieval = config.get('forcing', {}).get('retrieval')
        if retrieval:
            from pysweb.ssebop.scene_protocol import run_scene_workflow
            if not config.get('landsat'):
                raise ValueError('Configured scene protocol requires an immutable scene manifest')
            scene_protocol = {**retrieval, 'development_end': boundary}
            if 'aoi_bounds' in run:
                scene_protocol['support_aoi'] = dict(bounds=run['aoi_bounds'], crs=run['aoi_crs'])
            run_scene_workflow(date_range=period,scene_files=options['scene_files'],
                gapfill_products=options.get('gapfill_products',{}),dem=retrieval['dem_file'],
                output_dir=internal,extent=extent,protocol=scene_protocol,
                tcold_options={key:value for key,value in options.items() if key.startswith('tcold_') or key == 'landcover'},
                reconstruction=config['forcing']['reconstruction'],workers=run['workers'])
        else:
            pysweb.ssebop.run(date_range=period,landsat_dir=str(prepared/'landsat'),met_dir=str(met),
                dem=str(prepared/'dem/nasadem.tif'),output_dir=str(internal),workers=run['workers'],**options)
    if 'ssebop' in stages or 'preprocess' in stages:
        source_template = open_template(seb_internal,'ET')
        if 'center' in run:
            template = centered_template(source_template, run['center'], run.get('chip_size_m',5100.0))
            atomic_json(base/'chip-window.json', dict(center=run['center'],
                size_m=run.get('chip_size_m',5100.0), bounds=list(template.rio.bounds()),
                crs=str(template.rio.crs), transform=list(template.rio.transform())[:6],
                shape=[template.sizes['y'],template.sizes['x']],
                selection='Nearest fixed-size native pixel window; no resampling'))
        else:
            template = subset_template(source_template,run.get('aoi_bounds',run['extent']),
                extent_crs=run.get('aoi_crs','EPSG:4326'))
        from pyproj import CRS
        import numpy as np
        if not CRS(template.rio.crs).utm_zone or not np.allclose(np.abs(template.rio.resolution()),30,rtol=0,atol=1e-6):
            raise ValueError('SSEBop source must be on a native 30 m Landsat UTM grid')
        crop_product(seb_internal,seb_final,template)
    if 'preprocess' in stages:
        first,last = month_bounds(start,end)
        soil_args = {'soil_source':soil.get('source','openlandmap')}
        for key in ('file','input_dir','mlcons_dir','layer_bottoms_mm'):
            if key in soil:
                soil_args['soil_'+key] = soil[key]
        if 'maximum_organic_matter_percent' in soil:
            soil_args['openlandmap_maximum_om_percent'] = soil['maximum_organic_matter_percent']
        rain_file = config.get('forcing',{}).get('retrieval',{}).get('meteorology_files',{}).get(
            'precipitation',str(met/f'precipitation_daily_{first:%Y-%m-%d}_{last:%Y-%m-%d}.nc'))
        if config.get('forcing',{}).get('precipitation_qc'):
            import xarray as xr
            from pysweb.forcing import precipitation_qc
            from pysweb.io.provenance import atomic_netcdf, input_manifest, save_output_manifest
            with xr.open_dataset(rain_file,decode_coords='all') as source:
                corrected,quality = precipitation_qc(source.precipitation.load(),**config['forcing']['precipitation_qc'])
            atomic_json(base/'precipitation-quality.json',quality)
            if not quality['passed']:
                raise ValueError('Native precipitation annual correction limit exceeded')
            corrected_path = prepared/'precipitation_corrected.nc'
            atomic_netcdf(corrected,corrected_path)
            save_output_manifest(corrected_path,input_manifest(config['forcing']['precipitation_qc'],[rain_file]))
            rain_file = str(corrected_path)
        common = dict(start_date=start,extent=run['extent'],gee_project=run.get('gee_project'),
            rain_file=rain_file,rain_var='precipitation',
            et_file=str(seb_internal),et_var='ET',soil_evaporation_var='E_s',transpiration_var='E_t',
            native_cache_dir=str(prepared/'soil_native'),workers=run['workers'],**soil_args)
        if 'center' in run:
            common.update(template_extent=list(template.rio.bounds()),template_extent_crs=str(template.rio.crs))
        elif 'aoi_bounds' in run:
            common.update(template_extent=run['aoi_bounds'],template_extent_crs=run['aoi_crs'])
        if enabled:
            pysweb.swb.preprocess(**common,end_date=fit_end,grid_mode='reference',output_dir=str(coarse),
                reference_file=str(reference),reference_var=soil.get('reference_var','reference_ssm'))
        pysweb.swb.preprocess(**common,end_date=end,grid_mode='landsat',grid_template=str(seb_final),
            grid_template_var='ET',output_dir=str(fine),skip_reference_ssm=True)
        if spinup or config.get('forcing'):
            from pysweb.workflow_protocol import write_support
            if enabled:
                write_support(coarse, start, fit_end, boundary)
                if config.get('model', {}).get('reference_file'):
                    from pysweb.swb.references import freeze_prepared_support
                    freeze_prepared_support(coarse, config['model']['reference_file'])
            write_support(fine, start, end, boundary)

    def forcing(directory,last,projected=False):
        tag = f'{start.replace("-","")}_{last.replace("-","")}'
        return dict(effective_precip=str(directory/f'effective_precip_daily_{tag}.nc'),
            et=str(directory/f'et_daily_{tag}.nc'),transpiration=str(directory/f'transpiration_daily_{tag}.nc'),
            ndvi=str(directory/f'ndvi_daily_{tag}.nc'),soil_dir=str(directory),date_range=[start,last],
            lat_dim='y' if projected else 'lat',lon_dim='x' if projected else 'lon',
            state_timing=run['state_timing'],workers=run['workers'])

    protocol = bool(spinup or config.get('forcing'))
    def initialize(directory, last, projected=False):
        from pysweb.workflow_protocol import initialize_inputs, initialized_forcing
        peers = [Path(p)/directory.name for p in config.get('forcing', {}).get('peer_runs', [])]
        record, mask = initialize_inputs(directory, start, last, fit_start, boundary, spinup, peers)
        arguments = initialized_forcing(record, directory, workers=run['workers'],
                                        state_timing=run['state_timing'], projected=projected)
        return record, mask, arguments

    calibration_file = base/'calibration.csv'
    model = dict(config.get('model',{}))
    if boundary:
        model['ndvi_reference_end'] = boundary
    if ('calibrate' in stages or 'benchmark' in stages) and enabled:
        options = {key:value for key,value in calibration.items() if key not in {'enabled','start','end','serial_polish'}}
        options['objective'] = 'cell'
        options['spinup_days'] = max(int(options.get('spinup_days',0)),(pd.Timestamp(fit_start)-pd.Timestamp(start)).days)
        for key in ('reference_file','max_root_depth','drainage_slope','drainage_upper_limit','drainage_lower_limit','use_ndvi_root_depth','ndvi_reference_end'):
            if key in model:
                options[key] = model[key]
        tag = f'{start.replace("-","")}_{fit_end.replace("-","")}'
        args = forcing(coarse,fit_end)
        reference_path = str(coarse/f'reference_ssm_daily_{tag}.nc')
        if protocol:
            record, mask, args = initialize(coarse, fit_end)
            reference_path = record['files']['reference_ssm'][0]
            options['spinup_days'] = record['spinup_days']
            if spinup:
                from pysweb.workflow_protocol import convergence_gate, default_parameters
                defaults = default_parameters(model)
                convergence_gate(coarse,record,mask,defaults,spinup,'default',model,workers=run['workers'])
        if 'benchmark' in stages:
            pysweb.swb.calibrate(**args,reference_ssm=reference_path,
                output=str(calibration_file),benchmark_output=base/'calibration-benchmark.json',**options)
        if 'calibrate' in stages:
            if calibration.get('serial_polish'):
                options['no_polish'] = True
                de_file = base/'calibration_de.csv'
                pysweb.swb.calibrate(**args,reference_ssm=reference_path,output=str(de_file),**options)
                serial_args = {**args, 'workers': 1}
                pysweb.swb.calibrate(**serial_args,reference_ssm=reference_path,output=str(calibration_file),
                                    polish_file=de_file,**options)
            else:
                pysweb.swb.calibrate(**args,reference_ssm=reference_path,output=str(calibration_file),**options)
            if spinup:
                parameters = pd.read_csv(calibration_file, float_precision="round_trip").iloc[0]
                values = {k:float(parameters[k]) for k in ('diff_factor','root_beta')}
                convergence_gate(coarse,record,mask,values,spinup,'calibrated',model,workers=run['workers'])
            if protocol:
                from pysweb.workflow_protocol import reproduce_calibration, calibration_identity
                reproduction = reproduce_calibration(coarse,record,mask,args,model,calibration_file,
                    surface_depth=calibration.get('surface_depth',50.))
                reproduction['accepted_input_identity'] = calibration_identity(config, coarse)
                atomic_json(base/'calibration-acceptance.json',reproduction)
    if 'run' in stages:
        if protocol and enabled:
            from pysweb.io.provenance import sha256
            from pysweb.workflow_protocol import calibration_identity
            acceptance = json.loads((base/'calibration-acceptance.json').read_text())
            if not acceptance['passed'] or acceptance['calibration_sha256'] != sha256(calibration_file):
                raise ValueError('Calibration acceptance missing or stale')
            current = calibration_identity(config, coarse)
            if acceptance['accepted_input_identity']['fingerprint'] != current['fingerprint']:
                raise ValueError('Calibration inputs, paired support, settings or software changed since acceptance')
        tag = f'{start.replace("-","")}_{end.replace("-","")}'
        args = forcing(fine,end,True)
        rain = str(fine/f'rain_daily_{tag}.nc')
        evaporation = (str(fine/f'soil_evaporation_daily_{tag}.nc'),'E_s')
        mask = None
        if protocol:
            record, mask, args = initialize(fine, end, True)
            rain = record['files']['precipitation'][0]
            evaporation = record['files']['E_s']
            if spinup:
                from pysweb.workflow_protocol import convergence_gate, default_parameters
                defaults = default_parameters(model)
                convergence_gate(fine,record,mask,defaults,spinup,'default',model,workers=run['workers'])
                if enabled:
                    parameters = pd.read_csv(calibration_file, float_precision="round_trip").iloc[0]
                    values = {k:float(parameters[k]) for k in defaults}
                    convergence_gate(fine,record,mask,values,spinup,'calibrated',model,workers=run['workers'])
        output = run_default_and_calibrated(**args,**model, precip=rain,output_dir=str(products),
            output_file='SWEB_with_spinup.nc' if spinup and 'source_years' not in spinup else 'SWEB_RZSM.nc',
            calibration_file=calibration_file if enabled else None,
            initialization=plan['initialization'],
            auxiliary_inputs={'soil_evaporation_forcing':evaporation})
        if spinup and 'source_years' not in spinup:
            from pysweb.workflow_protocol import trim_initialization
            output = trim_initialization(output, products/'SWEB_RZSM.nc', start)
        if protocol:
            from pysweb.evaluation import validate_simulation
            numerical = validate_simulation(output,mask,development_end=boundary)
            atomic_json(base/'simulation-acceptance.json',numerical)
            if not numerical['passed']:
                raise ValueError('Simulation validity/budget acceptance failed')
        if config.get('evaluation'):
            from pysweb.workflow_protocol import evaluate_product
            evaluation = dict(config['evaluation'])
            evaluation.setdefault('scoring_start', plan['initialization']['analysis_start'])
            evaluate_product(output,evaluation,base/'evaluation',mask)
        grid_contract = assert_identical_grids(seb_final,output)
        atomic_json(base/'products/grid-contract.json',grid_contract)
        if enabled:
            from pysweb.swb.transfer import evaluate_transfer
            tag = f'{start.replace("-","")}_{fit_end.replace("-","")}'
            score_start = max(pd.Timestamp(fit_start),pd.Timestamp(start)+pd.Timedelta(days=calibration.get('spinup_days',0)))
            evaluate_transfer(output,coarse/f'reference_ssm_daily_{tag}.nc',calibration_file,
                              str(score_start.date()),fit_end,base/'calibration-transfer.json',
                              surface_depth=calibration.get('surface_depth',50.))
    return plan
