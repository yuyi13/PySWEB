"""
Script: site_workflow.py
Objective: Compose resumable site stages around the maintained scientific APIs.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-23
Last updated: 2026-10-23
Inputs: Portable site JSON plan, scientific TOML, immutable inputs and observations.
Outputs: Stage receipts, validated products, evaluation, figures and site manifests.
Usage: python workflows/run_site.py --plan site.json --stages compute
Dependencies: pysweb, numpy, pandas, xarray, rasterio; POSIX file locks.
"""
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
import os
import shutil
import socket
import time

from pysweb.io.provenance import atomic_json, sha256
from pysweb.workflow import load_config

STAGES = ('acquire', 'retrieve', 'preprocess', 'references', 'simulate', 'evaluate', 'figures', 'finalise')


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_records(paths, root=None):
    result = {}
    for path in sorted(set(map(Path, paths))):
        key = str(path.relative_to(root)) if root else str(path)
        result[key] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    return result


def verified_records(records, root=None):
    for name, record in records.items():
        path = Path(root)/name if root else Path(name)
        if not path.is_file() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            return False
    return bool(records)


@contextmanager
def site_lock(directory):
    """Kernel lock survives failures without leaving an unrecoverable stale lock."""
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.writer.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another writer owns this site') from error
        handle.seek(0)
        handle.truncate()
        json.dump(dict(pid=os.getpid(), host=socket.gethostname(), job=os.environ.get('PBS_JOBID')), handle)
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def offline():
    """Fail an accidental network fallback in compute stages before any transfer."""
    original = socket.socket.connect
    def denied(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError('Network access forbidden in offline compute stages')
        return original(self, address)
    socket.socket.connect = denied
    try:
        yield
    finally:
        socket.socket.connect = original


def load_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    for key in ('config', 'destination', 'retained_soil', 'observation_file', 'observation_manifest'):
        if key in plan:
            value = Path(plan[key]).expanduser()
            plan[key] = str(value if value.is_absolute() else (path.parent/value).resolve())
    cfg = load_config(plan['config'])
    if plan.get('acquire_missing_sources') and Path(cfg['landsat']['manifest_file']).exists():
        catalogue = json.loads(Path(cfg['landsat']['manifest_file']).read_text())
        cfg['forcing']['retrieval']['template_file'] = catalogue['scenes'][0]['file']
    plan['config_values'] = cfg
    plan['base'] = str(Path(cfg['run']['root'])/cfg['run']['name'])
    if Path(plan['base']).resolve() == Path(plan['destination']).resolve():
        raise ValueError('Working directory and dataset destination must differ')
    return plan


def science_identity(plan):
    """Paths and worker counts do not change the declared experiment."""
    cfg = copy.deepcopy(plan['config_values'])
    for key in ('root', 'name', 'workers', 'gee_project'):
        cfg['run'].pop(key, None)
    retrieval = cfg['forcing']['retrieval']
    for key in ('landcover_file', 'dem_file', 'template_file', 'scene_cache_dir',
                'meteorology_raw_dir', 'meteorology_stack_dir', 'meteorology_files'):
        retrieval.pop(key, None)
    for section in ('soil', 'model'):
        cfg[section].pop('reference_file', None)
    cfg['landsat'].pop('manifest_file', None)
    manifest = Path(plan['config_values']['landsat']['manifest_file'])
    catalogue = json.loads(manifest.read_text()) if manifest.exists() else {'scenes': []}
    if not manifest.exists() and not plan.get('acquire_missing_sources'):
        raise FileNotFoundError(manifest)
    scenes = [{k:v for k,v in row.items() if k not in ('file', 'gap_mask_file')}
              for row in catalogue['scenes']]
    result = dict(config=cfg, scenes=scenes, scientific_baseline=plan['scientific_baseline'],
                reference_policy='derive_missing_with_baseline_recipe' if plan.get('derive_physical_reference') else 'reuse_fixed')
    if 'reference_coverage_policy' in plan:
        result['reference_coverage_policy'] = plan['reference_coverage_policy']
    return result


def source_identity():
    root = Path(__file__).parent
    return fingerprint({str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*.py'))
                        if p.name != 'site_evaluation.py' and 'visualisation' not in p.parts})


def external_inputs(plan):
    cfg = plan['config_values']
    r = cfg['forcing']['retrieval']
    fixed = [] if plan.get('derive_physical_reference') else [Path(cfg['model']['reference_file'])]
    return [Path(r['dem_file']), *fixed, Path(cfg['soil']['reference_file']),
            *map(Path, r['meteorology_files'].values()), *Path(plan['retained_soil']).glob('*.nc')]


def copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checksum = sha256(source)
    if destination.exists() and sha256(destination) == checksum:
        return
    temporary = destination.with_name(destination.name+'.partial')
    shutil.copyfile(source, temporary)
    if sha256(temporary) != checksum:
        raise ValueError(f'Copy checksum differs: {source}')
    temporary.replace(destination)


def import_pilot(plan, stage):
    """Import an explicitly proven compatible run without rerunning its solver."""
    import numpy as np
    reuse = plan['reuse_pilot']
    old, base = Path(reuse['base']), Path(plan['base'])
    parity = json.loads(Path(reuse['parity_receipt']).read_text())
    if not parity.get('passed') or parity['source_identity'] != source_identity():
        raise ValueError('Pilot reuse requires parity for this scientific implementation')
    if sha256(reuse['source_archive']) != reuse['source_sha256']:
        raise ValueError('Frozen pilot source archive changed')
    old_cfg = load_config(reuse['config'])
    old_cfg['run']['analysis_start'] = plan['config_values']['run']['analysis_start']
    old_plan = {**plan, 'config_values':old_cfg}
    a, b = science_identity(old_plan), science_identity(plan)
    if a['config'] != b['config']:
        raise ValueError('Pilot scientific configuration differs')
    if {s['product_id'] for s in a['scenes']} != {s['product_id'] for s in b['scenes']}:
        raise ValueError('Pilot immutable scene cohort differs')
    for name in ('acquire','forcing','model'):
        receipt = json.loads((old/f'{name}_acceptance.json').read_text())
        if not receipt['passed'] or receipt['config_sha256'] != sha256(reuse['config']) or receipt['source_sha256'] != reuse['source_sha256']:
            raise ValueError(f'Pilot {name} acceptance is stale')
    copied = []
    if stage == 'acquire':
        catalogue = json.loads(Path(plan['config_values']['landsat']['manifest_file']).read_text())
        files = [Path(s['file']) for s in catalogue['scenes']]
        files.append(Path(plan['config_values']['forcing']['retrieval']['landcover_file']))
        for path in files:
            receipt = json.loads(path.with_suffix(path.suffix+'.native.json').read_text())
            if receipt['sha256'] != sha256(path):
                raise ValueError(f'Pilot acquisition changed: {path}')
        for p in Path(plan['retained_soil']).glob('*.nc'):
            copy_verified(p,base/'inputs/soil_native'/p.name)
        original = json.loads((old/'acquire_acceptance.json').read_text())
        for path, checksum in original['retained_inputs'].items():
            if sha256(path) != checksum:
                raise ValueError(f'Pilot retained input changed: {path}')
        atomic_json(base/'acquisition_inputs.json',dict(passed=True,reused_from=str(old),
            retained_inputs=file_records(external_inputs(plan)),source_files=file_records(files),
            original_job_id=original['job_id'],parity_receipt=reuse['parity_receipt']))
        copied = [base/'acquisition_inputs.json',*files,*(base/'inputs/soil_native').glob('*.nc')]
    else:
        if stage == 'retrieve':
            selected = [*list((old/'ssebop').rglob('*')),old/'selected-scenes.json']
        elif stage == 'preprocess':
            selected = [*(old/'run_inputs').glob('*'),old/'products/SSEBop_ET.nc',old/'chip-window.json']
        elif stage == 'references':
            return references(plan)
        elif stage == 'simulate':
            selected = [*(old/'products').glob('*'),*(old/'run_inputs/initialized').glob('*'),old/'simulation-acceptance.json']
        else:
            raise ValueError('Only acquisition and numerical stages can reuse a pilot')
        for path in selected:
            if path.is_file():
                target = base/path.relative_to(old)
                copy_verified(path,target)
                copied.append(target)
        if stage == 'simulate':
            validate_product(plan)
            copied.append(base/'product-validation.json')
    atomic_json(base/f'pilot_reuse_{stage}.json',dict(passed=True,stage=stage,source=str(old),
        parity_receipt=reuse['parity_receipt'],scientific_baseline=plan['scientific_baseline'],
        source_archive_sha256=reuse['source_sha256']))
    return copied+[base/f'pilot_reuse_{stage}.json']


def acquire(plan):
    from pysweb.io.acquisition import acquire_landsat_manifest, worldcover_extent
    from pysweb.io.worldcover_tiles import acquire_worldcover_tiled
    from pysweb.io.grids import open_template, centered_template
    from pysweb.soil.openlandmap import _load_openlandmap_predictors
    from rasterio.warp import transform_bounds
    import rasterio
    import xarray as xr
    import pandas as pd
    cfg, base = plan['config_values'], Path(plan['base'])
    r = cfg['forcing']['retrieval']
    acquired = []
    if plan.get('acquire_missing_sources'):
        catalogue = acquire_landsat_manifest(cfg['landsat']['manifest_file'], r['acquisition_extent'],
                                            gee_project=cfg['run']['gee_project'])
        from pysweb.site_sources import prepare_auxiliary_sources
        acquired = prepare_auxiliary_sources(plan)
    inputs = external_inputs(plan)
    missing = [str(p) for p in inputs if not p.is_file()]
    for variable in ('clay', 'sand', 'soc'):
        if not list(Path(plan['retained_soil']).glob(variable+'_*.nc')):
            missing.append(variable+' retained soil predictors')
    if missing:
        raise FileNotFoundError(str(missing))
    if not plan.get('acquire_missing_sources'):
        catalogue = acquire_landsat_manifest(cfg['landsat']['manifest_file'], r['acquisition_extent'],
                                            gee_project=cfg['run']['gee_project'])
    cover = acquire_worldcover_tiled(r['landcover_file'],
        worldcover_extent(catalogue['files'], r['acquisition_extent']), gee_project=cfg['run']['gee_project'])
    soil = base/'inputs/soil_native'
    for p in Path(plan['retained_soil']).glob('*.nc'):
        copy_verified(p, soil/p.name)
    template = centered_template(open_template(r['template_file']), cfg['run']['center'], cfg['run']['chip_size_m'])
    bounds = transform_bounds(template.rio.crs, 'EPSG:4326', *template.rio.bounds(), densify_pts=21)
    _load_openlandmap_predictors((bounds[0]-.01,bounds[1]-.01,bounds[2]+.01,bounds[3]+.01),
                                cfg['run']['gee_project'], cache_dir=soil)
    for path in catalogue['files']:
        with rasterio.open(path) as ds:
            bounds = transform_bounds(ds.crs, r['padding']['crs'], *ds.bounds, densify_pts=21)
        wanted = r['padding']['comparison_bounds']
        if not (bounds[0] <= wanted[0] and bounds[1] <= wanted[1] and bounds[2] >= wanted[2] and bounds[3] >= wanted[3]):
            raise ValueError(f'Scene does not contain comparison window: {path}')
    for key, path in r['meteorology_files'].items():
        with xr.open_dataset(path, decode_coords='all') as ds:
            dates = pd.DatetimeIndex(ds.time.values)
            if not dates.equals(pd.date_range(r['climatology_start'], r['meteorology_end'])):
                raise ValueError(f'Incomplete meteorology calendar: {key}')
            extent = transform_bounds(ds.rio.crs,r['padding']['crs'],*ds.rio.bounds(),densify_pts=21)
            wanted = r['padding']['comparison_bounds']
            if not (extent[0] <= wanted[0] and extent[1] <= wanted[1] and extent[2] >= wanted[2] and extent[3] >= wanted[3]):
                raise ValueError(f'Meteorology does not cover the comparison window: {key}')
    with rasterio.open(r['dem_file']) as ds:
        extent = transform_bounds(ds.crs,r['padding']['crs'],*ds.bounds,densify_pts=21)
    if not (extent[0] <= wanted[0] and extent[1] <= wanted[1] and extent[2] >= wanted[2] and extent[3] >= wanted[3]):
        raise ValueError('DEM does not cover the comparison window')
    record = dict(passed=True, retained_inputs=file_records(inputs), worldcover=cover,
                  scene_count=len(catalogue['files']), comparison_extent_verified=True)
    atomic_json(base/'acquisition_inputs.json', record)
    return [base/'acquisition_inputs.json', *map(Path, catalogue['files']), Path(r['landcover_file']),
            *soil.glob('*.nc'), *acquired]


def retrieve(plan):
    import rasterio
    import xarray as xr
    from pysweb.io.grids import open_template, centered_template
    from pysweb.ssebop.preprocessing import select_scenes
    from pysweb.ssebop.scene_protocol import run_scene_workflow
    cfg, base = plan['config_values'], Path(plan['base'])
    r = cfg['forcing']['retrieval']
    records = json.loads(Path(cfg['landsat']['manifest_file']).read_text())['scenes']
    selected = select_scenes(records, cfg['run']['sensors'], plan.get('catalogue_start','2013-01-01'),
                             cfg['run']['end'], wrs=cfg['landsat']['wrs'])
    grid = centered_template(open_template(r['template_file']), cfg['run']['center'], cfg['run']['chip_size_m'])
    extent = cfg['run']['extent']
    if plan.get('derive_physical_reference'):
        from pysweb.io.grids import subset_template
        from rasterio.warp import transform_bounds
        reference_grid = subset_template(open_template(cfg['soil']['reference_file'],'reference_ssm'),extent)
        extent = list(transform_bounds(reference_grid.rio.crs,'EPSG:4326',*reference_grid.rio.bounds(),densify_pts=21))
    r = {**r, 'development_end':plan['development_end'],
         'support_aoi':dict(bounds=list(grid.rio.bounds()), crs=str(grid.rio.crs))}
    workers = min(2, cfg['run']['workers'], plan.get('scene_workers',2))
    with rasterio.Env(GDAL_CACHEMAX=256*1024**2), xr.set_options(file_cache_maxsize=8):
        run_scene_workflow(date_range=f"{cfg['run']['start']} to {cfg['run']['end']}",
            scene_files=[s['file'] for s in selected], gapfill_products={}, dem=r['dem_file'],
            output_dir=base/'ssebop', extent=extent, protocol=r,
            reconstruction=cfg['forcing']['reconstruction'], workers=workers, tcold_options=cfg['ssebop'])
    atomic_json(base/'selected-scenes.json', dict(scenes=selected))
    return [p for p in (base/'ssebop').rglob('*') if p.is_file()] + [base/'selected-scenes.json']


def preprocess(plan):
    from pysweb.workflow_native import run_native
    cfg, base = plan['config_values'], Path(plan['base'])
    preparation = copy.deepcopy(cfg)
    if plan.get('derive_physical_reference'):
        # The baseline uses native reference-grid complete-development support.
        # Enable only that preprocessing grid; no calibration call or fit occurs.
        preparation['calibration'] = dict(enabled=True,start=cfg['run']['analysis_start'],end=plan['development_end'])
        preparation['model'].pop('reference_file')
    run_native(preparation, ['preprocess'])
    paths = [p for p in (base/'run_inputs').glob('*') if p.is_file()]
    if plan.get('derive_physical_reference'):
        paths += [p for p in (base/'calibration_inputs').glob('*') if p.is_file()]
    return paths + [base/'products/SSEBop_ET.nc', base/'chip-window.json']


def references(plan):
    """Freeze missing references using the executed baseline native-grid recipe."""
    from types import SimpleNamespace
    import xarray as xr
    import numpy as np
    from pysweb.io.grids import spatial,open_template,centered_template,intersecting_cells
    from pysweb.soil.openlandmap import process_soil_properties_from_openlandmap
    from pysweb.swb.references import derive_reference
    from pysweb.swb.formulation import load_reference
    cfg,base = plan['config_values'],Path(plan['base'])
    output = Path(cfg['model']['reference_file'])
    if not plan.get('derive_physical_reference'):
        if plan.get('reference_coverage_policy') is not None:
            raise ValueError('A changed reference coverage policy requires deriving a new artifact')
        load_reference(output)
        atomic_json(base/'reference-validation.json',dict(passed=True,mode='reuse_fixed',path=str(output),sha256=sha256(output)))
        return [base/'reference-validation.json']
    reference = Path(cfg['soil']['reference_file'])
    support = base/'calibration_inputs/development_support.nc'
    with xr.open_dataset(support,decode_coords='all') as ds:
        mask = spatial(ds.development_support).astype(bool).load()
    with xr.open_dataset(reference,decode_coords='all') as ds:
        gssm = spatial(ds.reference_ssm).sel(x=mask.x,y=mask.y).load()
    predictors,inputs = {},[reference,support]
    for key in ('clay','sand','soc'):
        candidates = []
        for path in (base/'inputs/soil_native').glob(key+'_*.nc'):
            with xr.open_dataset(path,decode_coords='all') as ds:
                candidates.append((ds[key].sizes['y']*ds[key].sizes['x'],path))
        path = max(candidates)[1]
        with xr.open_dataset(path,decode_coords='all') as ds:
            predictors[key] = ds[key].load()
        inputs.append(path)
    xr.align(*predictors.values(),join='exact')
    native = predictors['clay']
    native_grid = SimpleNamespace(lat_dim='y',lon_dim='x',latitudes=native.y.values,longitudes=native.x.values)
    hydraulics = process_soil_properties_from_openlandmap(
        SimpleNamespace(dtype='float64',openlandmap_maximum_om_percent=8.),native_grid,
        predictors,reproject_to_template=lambda da,*a,**k:da)
    for key in hydraulics:
        hydraulics[key] = hydraulics[key].rio.write_crs(native.rio.crs).rio.write_transform(native.rio.transform())
    hydraulics['field_capacity'] = hydraulics['wilting_point']+hydraulics['available_water_capacity']
    physical = derive_reference(gssm,mask,{k:spatial(v.isel(layer=0,drop=True)) for k,v in hydraulics.items()},
        site=plan['site_key'],sources=[dict(path=str(p),sha256=sha256(p)) for p in inputs],
        hydraulic_provenance=dict(method='Native Saxton–Rawls before area averaging',maximum_organic_matter_percent=8.,
            soc_missing_g_per_kg=5.,source_module_sha256=sha256(Path(__import__('pysweb.soil.openlandmap',fromlist=['x']).__file__)),
            predictor_crs=str(native.rio.crs),predictor_transform=list(native.rio.transform())[:6]),
        coverage_policy=plan.get('reference_coverage_policy'))
    fine = centered_template(open_template(cfg['forcing']['retrieval']['template_file']),cfg['run']['center'])
    if np.any(mask.values & ~intersecting_cells(mask,fine.rio.bounds(),str(fine.rio.crs)).values):
        raise ValueError('Physical reference includes cells outside the frozen site footprint')
    atomic_json(output,physical)
    load_reference(output)
    atomic_json(base/'reference-validation.json',dict(passed=True,mode='derived_missing_with_baseline_recipe',
        artifact_id=physical['artifact_id'],source_inputs=file_records(inputs),in_situ_used=False,
        calibration_performed=False,sha256=sha256(output)))
    return [output,base/'reference-validation.json']


def simulate(plan):
    from pysweb.workflow_native import run_native
    cfg, base = plan['config_values'], Path(plan['base'])
    run_native(cfg, ['run'])
    validate_product(plan)
    return [p for p in (base/'products').glob('*') if p.is_file()] + [base/'simulation-acceptance.json', base/'product-validation.json', base/'run_inputs/initialized/fit_mask.nc']


def validate_product(plan):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from pysweb.evaluation import validate_simulation
    from pysweb.io.products import assert_identical_grids
    base, cfg = Path(plan['base']), plan['config_values']
    with xr.open_dataset(base/'run_inputs/initialized/fit_mask.nc') as ds:
        mask = ds.fit_mask.load().astype(bool)
    path = base/'products/SWEB_RZSM.nc'
    result = validate_simulation(path, mask, development_end=plan['development_end'])
    if not result['passed']:
        raise ValueError(f'Numerical validation failed: {result}')
    grid = assert_identical_grids(base/'products/SSEBop_ET.nc', path)
    with xr.open_dataset(path, decode_coords='all') as ds:
        dates = pd.DatetimeIndex(ds.time.values)
        assert dates.equals(pd.date_range(cfg['run']['start'],cfg['run']['end']))
        initial = json.loads(ds.attrs['initialization_json'])
        assert initial['analysis_start'] == cfg['run']['analysis_start']
        assert initial['artificial_warmup_days'] == 0 and not initial['repeated_spinup_enabled']
        assert ds.attrs['scenarios'] == 'default'
        assert not any(n.endswith('_calibrated') for n in ds.data_vars)
        np.testing.assert_array_equal(ds.is_warmup.values.astype(bool), dates < pd.Timestamp(cfg['run']['analysis_start']))
        initial_state = ds.rzsm_default.isel(time=0).transpose('layer','y','x').values
        capacity = (ds.soil_wilting_point+ds.soil_available_water_capacity).transpose('layer','y','x').values
        np.testing.assert_allclose(initial_state[:,mask.values],capacity[:,mask.values],rtol=0,atol=1e-7)
    atomic_json(base/'product-validation.json', dict(passed=True,numerical=result,grid=grid,initialization=initial))


def evaluate(plan):
    from pysweb.site_evaluation import evaluate_site
    evaluate_site(plan)
    return [p for p in (Path(plan['base'])/'evaluation').rglob('*') if p.is_file()]


def figures(plan):
    from pysweb.visualisation.site_figures import make_figures
    make_figures(plan)
    return [p for p in (Path(plan['base'])/'figures').rglob('*') if p.is_file()]


def finalise(plan):
    """Promote verified files; human figure review remains a distinct gate."""
    base, destination = Path(plan['base']), Path(plan['destination'])
    records = {}
    for folder in ('products','evaluation','figures'):
        # Solver intermediates remain in work; export final products and their
        # companion manifests without duplicating the intermediate trajectory.
        paths = (base/folder).glob('*') if folder == 'products' else (base/folder).rglob('*')
        for source in sorted(paths):
            if source.is_file():
                target = destination/source.relative_to(base)
                copy_verified(source,target)
                records[str(target.relative_to(destination))] = dict(bytes=target.stat().st_size,sha256=sha256(target))
    for source in [*base.glob('*.json'), *(base/'receipts').glob('*.json')]:
        copy_verified(source,destination/'provenance'/source.name)
    for source in (base/'ssebop').glob('*.json'):
        copy_verified(source,destination/'provenance'/source.name)
    copy_verified(plan['config'],destination/'provenance/config.toml')
    copy_verified(plan['config_values']['landsat']['manifest_file'],destination/'provenance/scenes.json')
    atomic_json(destination/'provenance/site_plan.json',
                {key:value for key,value in plan.items() if key not in ('config_values','base')})
    atomic_json(destination/'provenance/resolved_config.json',plan['config_values'])
    package_inputs(plan)
    import xarray as xr
    dictionary = {}
    for name in ('SSEBop_ET.nc','SWEB_RZSM.nc'):
        with xr.open_dataset(destination/'products'/name,decode_coords='all') as ds:
            dictionary[name] = dict(dimensions=dict(ds.sizes),attributes={k:str(v) for k,v in ds.attrs.items()},
                variables={key:dict(dimensions=list(v.dims),dtype=str(v.dtype),
                    attributes={k:str(value) for k,value in v.attrs.items()},
                    encoding={k:str(v.encoding[k]) for k in ('_FillValue','scale_factor','add_offset') if k in v.encoding})
                    for key,v in ds.variables.items()})
    atomic_json(destination/'provenance/data_dictionary.json',dictionary)
    atomic_json(destination/'provenance/input_dependencies.json',dict(
        external=file_records(external_inputs(plan)),work_directory=str(base),
        source_manifest=plan['config_values']['landsat']['manifest_file'],
        self_contained=False,retention='Scratch inputs remain external; public release not claimed'))
    metadata_path = destination/'metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update(status='numerically_validated_pending_visual_review',run_id=plan['run_id'],
        scientific_baseline=plan['scientific_baseline'],implementation_commit=plan['implementation_commit'],
        actual_grid=json.loads((base/'product-validation.json').read_text())['grid'],
        observations=json.loads((base/'evaluation/observation_evaluation.json').read_text()),updated_at=now())
    atomic_json(metadata_path,metadata)
    for folder in ('inputs','provenance'):
        for path in (destination/folder).rglob('*'):
            if path.is_file() and path.name != 'output_manifest.json':
                records[str(path.relative_to(destination))] = dict(bytes=path.stat().st_size,sha256=sha256(path))
    records['metadata.json'] = dict(bytes=metadata_path.stat().st_size,sha256=sha256(metadata_path))
    atomic_json(destination/'provenance/output_manifest.json',dict(passed=True,files=records,
        status='numerically_validated_pending_visual_review',scope='Products, evaluation, figures, inputs, provenance and site metadata; excludes this manifest and logs'))
    return [destination/'provenance/output_manifest.json']


def package_inputs(plan):
    """Keep one checksum-verified source container with the reusable dataset."""
    import tarfile
    base, destination = Path(plan['base']), Path(plan['destination'])/'inputs'
    destination.mkdir(exist_ok=True)
    if plan.get('reuse_pilot'):
        old = Path(plan['reuse_pilot']['base'])
        original = json.loads((old/'sources.manifest.json').read_text())
        if not original['passed'] or sha256(old/'sources.tar') != original['archive_sha256']:
            raise ValueError('Pilot source container changed')
        copy_verified(old/'sources.tar',destination/'sources.tar')
        copy_verified(old/'sources.manifest.json',destination/'sources.manifest.json')
        return
    files = [p for directory in (base/'raw',base/'inputs/soil_native') for p in directory.rglob('*') if p.is_file()]
    records = file_records(files,base)
    receipt_path = destination/'sources.manifest.json'
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        if prior.get('files') == records and (destination/'sources.tar').exists() and sha256(destination/'sources.tar') == prior['archive_sha256']:
            return
    temporary = destination/'sources.tar.partial'
    with tarfile.open(temporary,'w') as archive:
        for relative in records:
            if (base/relative).is_symlink():
                raise ValueError('Source archives must contain regular files')
            archive.add(base/relative,arcname=relative,recursive=False)
    with tarfile.open(temporary) as archive:
        if set(archive.getnames()) != set(records):
            raise ValueError('Source archive membership differs')
        for member in archive:
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda:stream.read(8*1024**2),b''):
                    digest.update(block)
            if digest.hexdigest() != records[member.name]['sha256']:
                raise ValueError('Source archive payload differs')
    temporary.replace(destination/'sources.tar')
    atomic_json(receipt_path,dict(passed=True,files=records,archive_sha256=sha256(destination/'sources.tar')))


def run_plan(plan, stages):
    cfg, base = plan['config_values'], Path(plan['base'])
    if cfg['calibration']['enabled'] or cfg.get('spinup',{}).get('enabled',True):
        raise ValueError('Site workflow requires an uncalibrated actual-prefix trajectory')
    software = source_identity()
    with site_lock(base), site_lock(plan['destination']):
        if plan.get('acquire_missing_sources') and 'acquire' in stages:
            from pysweb.site_sources import ensure_catalogue
            try:
                ensure_catalogue(plan)
            except BaseException as error:
                atomic_json(base/'checkpoint.json',dict(stage='source_catalogue',passed=False,
                    updated_at=now(),error_type=type(error).__name__,error=str(error),
                    job_id=os.environ.get('PBS_JOBID')))
                raise
        science = fingerprint(science_identity(plan))
        receipts = base/'receipts'
        receipts.mkdir(exist_ok=True)
        identity = dict(science=science,software=software)
        # Hash external dependencies in allocated execution; no existence-only reuse.
        # Fresh inputs are outputs of acquisition and verified through that
        # receipt. Pre-existing inputs retain the original external contract.
        external = {} if plan.get('acquire_missing_sources') else file_records(external_inputs(plan))
        checked = {}
        def prerequisite(name):
            # An isolated downstream invocation must reject stale ancestors too.
            # Memoisation avoids hashing the same accepted files repeatedly while
            # this process holds the exclusive site lock.
            if name in checked:
                return checked[name]
            record = json.loads((receipts/(name+'.json')).read_text())
            if not record.get('passed') or record['identity'] != identity:
                raise ValueError(f'Missing or stale prerequisite: {name}')
            prior = STAGES.index(name)
            parents = [prerequisite(STAGES[prior-1])['fingerprint']] if prior else []
            inputs = record['inputs']
            if inputs['external'] != external or inputs['parents'] != parents:
                raise ValueError(f'Prerequisite inputs changed: {name}; resume from the affected stage')
            if 'observations' in inputs:
                if (inputs['observations'] != file_records([plan['observation_file'],plan['observation_manifest']])
                        or inputs['operator'] != plan['observation_operator']
                        or inputs['evaluation_code'] != sha256(Path(__file__).with_name('site_evaluation.py'))):
                    raise ValueError(f'Prerequisite evaluation changed: {name}')
            if 'figure_code' in inputs and inputs['figure_code'] != file_records((Path(__file__).parent/'visualisation').glob('*.py')):
                raise ValueError(f'Prerequisite figure code changed: {name}')
            if not verified_records(record['outputs']):
                raise ValueError(f'Prerequisite outputs changed: {name}')
            checked[name] = record
            return record
        for stage in stages:
            index = STAGES.index(stage)
            parents = []
            if index:
                parent = prerequisite(STAGES[index-1])
                parents.append(parent['fingerprint'])
            inputs = dict(identity=identity,external=external,parents=parents)
            if stage in ('evaluate','figures','finalise'):
                inputs['observations'] = file_records([plan['observation_file'],plan['observation_manifest']])
                inputs['operator'] = plan['observation_operator']
                inputs['evaluation_code'] = sha256(Path(__file__).with_name('site_evaluation.py'))
            if stage in ('figures','finalise'):
                inputs['figure_code'] = file_records((Path(__file__).parent/'visualisation').glob('*.py'))
            key = fingerprint(inputs)
            target = receipts/(stage+'.json')
            old = json.loads(target.read_text()) if target.exists() else {}
            if old.get('passed') and old.get('fingerprint') == key and verified_records(old['outputs']):
                print(f'{stage}: reuse verified receipt',flush=True)
                checked[stage] = old
                continue
            attempt = int(old.get('attempt',0))+1
            record = dict(stage=stage,attempt=attempt,identity=identity,fingerprint=key,inputs=inputs,
                started_at=now(),job_id=os.environ.get('PBS_JOBID'),passed=False,
                implementation_commit=plan['implementation_commit'])
            clock = time.monotonic()
            try:
                print(f'{stage}: starting attempt {attempt}',flush=True)
                if plan.get('reuse_pilot') and stage in STAGES[:5]:
                    with offline():
                        paths = import_pilot(plan,stage)
                elif stage == 'acquire':
                    paths = acquire(plan)
                else:
                    with offline():
                        paths = globals()[stage](plan)
                record.update(passed=True,outputs=file_records(paths))
                checked[stage] = record
            except BaseException as error:
                record.update(error_type=type(error).__name__,error=str(error))
                raise
            finally:
                record.update(finished_at=now(),elapsed_seconds=time.monotonic()-clock)
                atomic_json(receipts/f'{stage}_attempt_{attempt}.json',record)
                atomic_json(target,record)
                atomic_json(base/'checkpoint.json',dict(stage=stage,passed=record['passed'],updated_at=now(),identity=identity))
            print(f'{stage}: passed',flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--stages',default='compute')
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    stages = list(STAGES[1:]) if args.stages == 'compute' else list(STAGES) if args.stages == 'all' else args.stages.split(',')
    if args.stages == 'compute' and plan.get('reuse_pilot'):
        stages = list(STAGES)
    if any(s not in STAGES for s in stages) or stages != sorted(set(stages),key=STAGES.index):
        parser.error('Select unique stages in scientific order')
    if args.dry_run:
        print(json.dumps(dict(stages=stages,base=plan['base'],destination=plan['destination'],science=science_identity(plan)),indent=2))
    else:
        run_plan(plan,stages)
    return 0
