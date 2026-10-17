"""
Script: scene_protocol.py
Objective: Retrieve padded native scenes using bounded development climatology and acceptance gates.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Native meteorology, DEM, Landsat fields and configured padding/climatology rules.
Outputs: Calendar-day dT, local-anchor fractions and independently screened retrievals.
Usage: from pysweb.ssebop.scene_protocol import calendar_day_dt, screened_fraction
Dependencies: numpy, pandas, rasterio, xarray, pysweb
"""
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.warp import transform_bounds
from pysweb.io.grids import align_to_grid
from pysweb.ssebop.core import (compute_dt_daily, tcold_fano_local_xr, et_fraction_xr,
    _window_from_scale, _rolling_mean_2d, _coarsen_mean_2d, _interp_like_2d)
from pysweb.forcing import padding_acceptance


def calendar_day_dt(meteorology, dem, latitude, date, *, start, end, temperature_units="celsius"):
    """Match calendar days in complete development years; Feb29 uses Feb28/Mar1.

    Only the selected 21 (or 42) native daily slices are reprojected. This retains
    the inherited evaluate-then-average calculation without allocating decades
    of meteorology over the padded Landsat grid.
    """
    first, last, date = pd.Timestamp(start), pd.Timestamp(end), pd.Timestamp(date)
    if (first.month, first.day) != (1, 1) or (last.month, last.day) != (12, 31):
        raise ValueError("Calendar-day climatology requires complete configured years")
    selected = []
    for year in range(first.year, last.year+1):
        if (date.month, date.day) == (2, 29):
            selected.extend([pd.Timestamp(year, 2, 28), pd.Timestamp(year, 3, 1)])
        else:
            selected.append(pd.Timestamp(year, date.month, date.day))
    fields = {}
    for name in ("tmax", "tmin", "rs", "ea"):
        data = meteorology[name]
        if not pd.DatetimeIndex(selected).isin(pd.DatetimeIndex(data.time.values)).all():
            raise ValueError("Configured climatology meteorology is missing calendar dates")
        fields[name] = align_to_grid(data.sel(time=selected), dem, resampling="bilinear")
    if temperature_units == "celsius":
        fields["tmax"] = fields["tmax"]+273.15
        fields["tmin"] = fields["tmin"]+273.15
    elif temperature_units != "kelvin":
        raise ValueError("Unknown meteorological temperature units")
    dt = compute_dt_daily(fields["tmax"], fields["tmin"], dem, latitude, fields["rs"], fields["ea"]).mean("time", skipna=False)
    dt = dt.where(dt > 0)
    dt.attrs.update(climatology_start=str(start), climatology_end=str(end),
        climatology_calendar="month/day; February29 averages February28 and March1 from each development year")
    return dt


def fallback_fraction(ndvi, valid, config):
    fy, fx = _window_from_scale(ndvi, config.fine_scale_m)
    cy, cx = _window_from_scale(ndvi, config.coarse_scale_m)
    sy, sx = _window_from_scale(ndvi, config.smooth_scale_m)
    anchors = ndvi.where(valid & (_rolling_mean_2d(ndvi.where(valid), fy, fx) >= config.anchor_ndvi_threshold))
    coarse = _coarsen_mean_2d(anchors, cy, cx)
    return _rolling_mean_2d(_interp_like_2d(coarse.isnull().astype(float), ndvi), sy, sx).where(valid)


def _window(data, bounds, crs):
    left, bottom, right, top = transform_bounds(crs, data.rio.crs, *bounds, densify_pts=21)
    selected = data.sel(x=slice(left, right), y=slice(top, bottom))
    if not selected.sizes["x"] or not selected.sizes["y"]:
        raise ValueError("Padding bounds miss the native scene")
    return selected


def screened_fraction(lst, ndvi, dt, template, config, protocol):
    """Compare fixed padding windows after excluding cold-anchor fallback."""
    fractions, fallback, cold_fields = {}, {}, {}
    for label in ("primary", "comparison"):
        temperature = _window(lst, protocol[label+"_bounds"], protocol["crs"])
        vegetation = ndvi.sel(x=temperature.x, y=temperature.y)
        delta = dt.sel(x=temperature.x, y=temperature.y)
        cold = tcold_fano_local_xr(temperature, vegetation, delta, config=config)
        fraction = et_fraction_xr(temperature, cold, delta)
        weight = fallback_fraction(vegetation, np.isfinite(temperature) & np.isfinite(vegetation) & np.isfinite(delta), config)
        fallback[label] = align_to_grid(weight, template, resampling="bilinear")
        fractions[label] = align_to_grid(fraction, template, resampling="bilinear").where(
            fallback[label] <= protocol["maximum_fallback_fraction"])
        cold_fields[label] = align_to_grid(cold, template, resampling="bilinear")
    gate = padding_acceptance(fractions["primary"], fractions["comparison"],
        minimum_agreement=protocol["minimum_agreement"], maximum_p99_difference=protocol["maximum_p99_difference"])
    # A rejected scene contributes no ET observations; subsequent reconstruction
    # can use the date but must retain its unavailable original support.
    accepted = fractions["primary"] if gate["passed"] else xr.full_like(fractions["primary"], np.nan)
    fields = dict(etf_unfiltered_padding_primary=fractions["primary"],
        etf_comparison_padding=fractions["comparison"], cold_boundary_fallback_fraction=fallback["primary"],
        tcold_K=cold_fields["primary"])
    return accepted, fields, gate


_CONTEXT = None


def _initialize_context(context):
    global _CONTEXT
    _CONTEXT = context


def _retrieve_scene(path):
    from pathlib import Path
    import json
    from pysweb.io.provenance import input_manifest, atomic_json, sha256
    from pysweb.ssebop.api import process_landsat_scene, parse_date_from_filename
    context = _CONTEXT
    date = parse_date_from_filename(path)
    # Scene identity prevents paired workflows selecting different sensors on
    # one date from overwriting each other's products or receipts.
    folder = Path(context['cache'])/Path(path).stem
    for sub in ('etf','ndvi'):
        (folder/sub).mkdir(parents=True,exist_ok=True)
    paths = [path, context['meteorology']['dem'], *context['met_paths']]
    gapfill = context['gapfill_products'].get(str(Path(path).resolve()))
    if gapfill:
        paths.append(gapfill)
    config = dict(padding=context['padding'], climatology_start=context['meteorology']['start'],
                  climatology_end=context['meteorology']['end'], template=context['template_identity'])
    manifest = input_manifest(config, paths)
    outputs = [folder/'etf'/f'auxiliary_{date}.nc', folder/'etf'/f'etf_{date}.tif', folder/'ndvi'/f'ndvi_{date}.tif']
    receipt = folder/f'{date}.json'
    previous = json.loads(receipt.read_text()) if receipt.exists() else {}
    if previous.get('fingerprint') == manifest['fingerprint'] and all(
            p.exists() and previous.get('outputs', {}).get(str(p)) == sha256(p) for p in outputs):
        return np.datetime64(date, 'ns'), str(outputs[1]), str(outputs[2])
    from pysweb.ssebop.core import LocalFanoConfig
    result = process_landsat_scene(str(path), 'auto', 'ndvi', 'SR_B4', 'SR_B5', False, None,
        None, context['template'].rio.crs, str(folder/'etf'), str(folder/'ndvi'),
        tcold_config=LocalFanoConfig(), gapfill_product=gapfill, output_template=context['template'],
        meteorology_context=context['meteorology'], padding_protocol=context['padding'])
    atomic_json(receipt, {**manifest, 'outputs': {str(p): sha256(p) for p in outputs}})
    return result


def scene_support_calendar(fractions, support_aoi=None):
    """Locate actual scene support inside the configured analysis footprint."""
    from pysweb.io.grids import intersecting_cells
    valid = np.isfinite(fractions)
    scope = 'complete processing window'
    if support_aoi is not None:
        mask = intersecting_cells(fractions.isel(time=0, drop=True),
            support_aoi['bounds'], support_aoi['crs'])
        valid = valid & mask
        scope = 'complete native pixels intersecting the analysis AOI'
    supported = valid.any(('y', 'x')).values
    if not supported.any():
        raise ValueError('No accepted scene support inside the configured analysis area')
    return dict(first_supported_scene=str(pd.Timestamp(
        fractions.time.values[np.flatnonzero(supported)[0]]).date()),
        supported_scenes=int(supported.sum()), support_scope=scope, support_aoi=support_aoi)


def run_scene_workflow(*, date_range, scene_files, gapfill_products, dem,
                       output_dir, extent, protocol, reconstruction, workers=1):
    """Configured bounded-memory retrieval with persistent shared scene caching.

    The meteorology files explicitly cover the climatology and actual model
    calendars. Daily output uses the exact native template over the full coarse
    footprints; only scene retrieval uses the larger FANO padding envelope.
    """
    from pathlib import Path
    import json
    from concurrent.futures import ProcessPoolExecutor
    from pysweb.ssebop.api import parse_date_range, load_output_stack, _pool_context
    from pysweb.io.grids import open_template, subset_template, spatial
    from pysweb.io.provenance import input_manifest, atomic_netcdf, save_output_manifest, atomic_json
    from pysweb.forcing import reconstruct
    from pysweb.ssebop.core import et_from_fraction
    if workers < 1 or not scene_files:
        raise ValueError('Positive workers and selected scenes required')
    start, end = parse_date_range(date_range)
    boundary = protocol['development_end']
    if pd.Timestamp(protocol['climatology_end']) > pd.Timestamp(boundary):
        raise ValueError('Climatology includes holdout meteorology')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(protocol['scene_cache_dir'])
    for sub in ('etf','ndvi'):
        (cache/sub).mkdir(parents=True, exist_ok=True)
    template = subset_template(open_template(protocol.get('template_file', scene_files[0])), extent)
    meteorology, met_paths = {}, []
    for name in ('tmax','tmin','rs','ea','et_tall_crop'):
        path = protocol['meteorology_files'][name]
        with xr.open_dataset(path, decode_coords='all') as ds:
            meteorology[name] = spatial(ds[name]).load()
        met_paths.append(path)
    context = dict(template=template, template_identity=dict(crs=str(template.rio.crs),
        transform=list(template.rio.transform())[:6], shape=list(template.shape)),
        cache=str(cache), met_paths=met_paths, gapfill_products=gapfill_products,
        meteorology=dict(fields={k:meteorology[k] for k in ('tmax','tmin','rs','ea')},
            dem=str(dem), start=protocol['climatology_start'], end=protocol['climatology_end']),
        padding=protocol['padding'])
    if workers == 1:
        _initialize_context(context)
        outputs = [_retrieve_scene(path) for path in scene_files]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=_pool_context(),
                initializer=_initialize_context, initargs=(context,)) as pool:
            outputs = list(pool.map(_retrieve_scene, scene_files, chunksize=1))
    etf = load_output_stack([(t,e) for t,e,_ in outputs], 'etf', reference_grid=template)
    ndvi = load_output_stack([(t,v) for t,_,v in outputs], 'ndvi', reference_grid=template)
    original, scene_fields, gates = [], [], []
    for time, fraction_path, _ in outputs:
        path = Path(fraction_path).parent/f'auxiliary_{pd.Timestamp(time):%Y-%m-%d}.nc'
        with xr.open_dataset(path, decode_coords='all') as ds:
            original.append(ds.original_support.expand_dims(time=[time]).load())
            scene_fields.append(ds.load().expand_dims(scene_time=[time]))
            gates.append(dict(date=str(pd.Timestamp(time).date()), **json.loads(ds.attrs['padding_gate_json'])))
    atomic_json(output_dir/'padding_acceptance.json', dict(scenes=gates,
        accepted_scenes=sum(g['passed'] for g in gates), total_scenes=len(gates)))
    if not any(g['passed'] for g in gates):
        raise ValueError('No scenes pass fixed padding/local-anchor acceptance')
    calendar = scene_support_calendar(etf, protocol.get('support_aoi'))
    first_supported = calendar['first_supported_scene']
    atomic_json(output_dir/'scene-calendar.json',dict(**calendar,
        requested_start=start, development_end=boundary))
    if protocol.get('require_supported_start', False) and pd.Timestamp(start) < pd.Timestamp(first_supported):
        raise ValueError(f'Run starts before first supported scene {first_supported}; freeze that start before reconstruction')
    scenes = xr.Dataset(dict(etf=etf, ndvi=ndvi, original_support=xr.concat(original,'time').sortby('time')))
    dates = pd.date_range(start,end)
    daily = reconstruct(scenes, dates, development_end=boundary, workers=workers, **reconstruction)
    # Align only the final/full-GSSM-footprint preparation window, never decades
    # of daily meteorology over the large FANO padding domain.
    demand = align_to_grid(meteorology['et_tall_crop'].sel(time=dates),template).astype('float32')
    et = et_from_fraction(daily.etf,demand,fraction_type='alfalfa',reference_type='alfalfa')
    ft = (1.26*daily.ndvi-.18).clip(0,1)
    result = daily.rename(etf='etf_interp',ndvi='ndvi_interp')
    result['ET'], result['E_t'], result['E_s'], result['f_t'] = et, et*ft, et*(1-ft), ft
    result['reference_et'] = demand
    for name in ('ET','E_t','E_s','reference_et'):
        result[name].attrs['units'] = 'mm day-1'
    for name, out in (('tmax','tmax_K'),('tmin','tmin_K'),('rs','solar_radiation'),('ea','vapour_pressure')):
        values = align_to_grid(meteorology[name].sel(time=dates),template).astype('float32')
        result[out] = values+273.15 if name in {'tmax','tmin'} else values
    import rioxarray
    with rioxarray.open_rasterio(dem, masked=True) as source:
        result['elevation'] = align_to_grid(source.squeeze('band',drop=True),template).astype('float32')
    auxiliaries = xr.concat(scene_fields,'scene_time').sortby('scene_time')
    for name,value in auxiliaries.data_vars.items():
        result[name] = value
    from pysweb.components import COMPONENT_SCHEMA
    result = result.rio.write_crs(template.rio.crs).rio.write_transform(template.rio.transform())
    result.attrs.update(component_schema=COMPONENT_SCHEMA, selected_scenes='\n'.join(map(str,scene_files)),
        climatology_start=protocol['climatology_start'], climatology_end=protocol['climatology_end'],
        climatology_calendar='month/day; Feb29 uses Feb28/Mar1', development_end=boundary,
        same_day_selection='configured immutable catalogue: cloud cover then scene id',
        dt_provenance='Each scene_time dt_K is the directly used development calendar-day climatology')
    path = output_dir/f'et_daily_ssebop_{start}_{end}.nc'
    # Supplying encoding replaces xarray's per-variable encoding; preserve the
    # CF grid-mapping link written by rioxarray for every spatial auxiliary.
    encoding = {}
    from pysweb.io.netcdf_stream import stream_chunks
    for name, field in result.data_vars.items():
        encoding[name] = {key: value for key, value in field.encoding.items()
                          if key in ('grid_mapping', '_FillValue')}
        encoding[name].update(zlib=True, complevel=3)
        chunks = stream_chunks(field.dims, field.shape)
        if chunks is not None:
            encoding[name]['chunksizes'] = chunks
        if field.dtype.kind == 'f':
            encoding[name]['dtype'] = 'float32'
        else:
            field.attrs.pop('_FillValue', None)
            encoding[name]['_FillValue'] = None
    atomic_netcdf(result,path,encoding=encoding)
    save_output_manifest(path,input_manifest(dict(protocol=protocol,reconstruction=reconstruction),
        [*scene_files,*met_paths,dem,*gapfill_products.values()]))
    return path
