"""
Script: paired.py
Objective: Run default and calibrated SWB and stream both scenarios into one NetCDF.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Common prepared forcing/soil, optional calibrated scalars and run options.
Outputs: Consolidated RZSM scenarios, shared auxiliaries, budgets and provenance.
Usage: from pysweb.swb.paired import run_default_and_calibrated
Dependencies: numpy, netCDF4, pysweb
"""
from pathlib import Path
import json
import os
import numpy as np
import netCDF4 as nc
from pysweb.io.provenance import input_manifest, save_output_manifest
from pysweb.swb.run import run_swb_workflow
from pysweb.swb.core import SOIL_FILE_STEMS
from pysweb.io.products import assert_identical_grids, auxiliary_grid_contract
from pysweb.io.netcdf_stream import stream_chunks, sequential_read_cache


def _copy_variable(destination, variable, name=None, time_indices=None):
    name = name or variable.name
    if name in destination.variables:
        return destination.variables[name]
    for dim, length in zip(variable.dimensions, variable.shape):
        if dim not in destination.dimensions:
            destination.createDimension(dim, length)
        elif len(destination.dimensions[dim]) != length and not (dim == 'time' and time_indices is not None):
            raise ValueError(f"Auxiliary {name} has incompatible {dim} dimension")
    fill = variable.getncattr('_FillValue') if '_FillValue' in variable.ncattrs() else None
    options = {'zlib': True, 'complevel': 3} if variable.ndim else {}
    chunks = stream_chunks(variable.dimensions,
        tuple(len(destination.dimensions[d]) for d in variable.dimensions))
    if chunks is not None:
        options['chunksizes'] = chunks
    target = destination.createVariable(name, variable.datatype, variable.dimensions, fill_value=fill, **options)
    target.setncatts({k:variable.getncattr(k) for k in variable.ncattrs() if k != '_FillValue'})
    if 'time' not in variable.dimensions:
        target[...] = variable[...]
    else:
        axis = variable.dimensions.index('time')
        count = len(destination.dimensions['time'])
        with sequential_read_cache(variable):
            for begin in range(0, count, 32):
                source_index = [slice(None)]*variable.ndim
                target_index = [slice(None)]*variable.ndim
                target_index[axis] = slice(begin, min(begin+32, count))
                selected = target_index[axis] if time_indices is None else time_indices[begin:begin+32]
                if not isinstance(selected, slice) and len(selected) and np.all(np.diff(selected) == 1):
                    selected = slice(int(selected[0]), int(selected[-1])+1)
                source_index[axis] = selected
                target[tuple(target_index)] = variable[tuple(source_index)]
    return target


def _time_indices(source, destination):
    def dates(variable):
        return [str(value) for value in nc.num2date(variable[:], variable.units,
                calendar=getattr(variable, 'calendar', 'standard'), only_use_cftime_datetimes=True)]
    lookup = {value:index for index,value in enumerate(dates(source.variables['time']))}
    try:
        return np.asarray([lookup[value] for value in dates(destination.variables['time'])])
    except KeyError as exc:
        raise ValueError("Auxiliary input does not cover the simulation calendar") from exc


def combine_scenarios(scenarios, output, auxiliaries=None, config=None):
    """Copy in bounded time blocks; no second full-domain in-memory result cube."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.{os.getpid()}.tmp')
    auxiliaries = auxiliaries or {}
    auxiliary_grids = {}
    first_path = next(iter(scenarios.values()))
    for path in scenarios.values():
        assert_identical_grids(first_path,path)
    try:
        with nc.Dataset(first_path) as first, nc.Dataset(temporary, 'w') as result:
            for name, dimension in first.dimensions.items():
                result.createDimension(name, len(dimension))
            for name in first.dimensions:
                if name in first.variables:
                    _copy_variable(result, first[name])
            if 'spatial_ref' in first.variables:
                _copy_variable(result, first['spatial_ref'])
            layers = sorted((name for name in first.variables if name.startswith('rzsm_layer_')),
                            key=lambda name:int(name.rsplit('_',1)[1]))
            if not layers:
                raise ValueError("Scenario has no soil-moisture layers")
            result.createDimension('layer', len(layers))
            result.createVariable('layer', 'i4', ('layer',))[:] = np.arange(1, len(layers)+1)
            depth = result.createVariable('layer_bottom', 'f8', ('layer',))
            depth[:] = [first[name].depth_bottom_mm for name in layers]
            depth.units = 'mm'
            result.setncatts({k:first.getncattr(k) for k in first.ncattrs()})
            result.title = 'SWB default and calibrated soil moisture'
            result.scenarios = ','.join(scenarios)
            result.configuration_json = json.dumps(config or {}, default=str, sort_keys=True)
            for name in ('precipitation', 'effective_precipitation'):
                _copy_variable(result, first[name])
            for scenario, path in scenarios.items():
                with nc.Dataset(path) as source:
                    for dim in first.dimensions:
                        if dim not in source.dimensions or len(source.dimensions[dim]) != len(first.dimensions[dim]):
                            raise ValueError("Scenario dimensions differ")
                        if dim in first.variables and not np.array_equal(source[dim][:], first[dim][:]):
                            raise ValueError("Scenario coordinates differ")
                    base = source[layers[0]]
                    dims = ('time', 'layer')+base.dimensions[1:]
                    chunks = (min(32,base.shape[0]),1)+tuple(min(64,n) for n in base.shape[1:])
                    moisture = result.createVariable(f'rzsm_{scenario}', base.datatype, dims,
                        zlib=True, complevel=3, chunksizes=chunks, fill_value=np.nan)
                    moisture.setncatts(dict(units='m3 m-3', long_name=f'Layer soil moisture: {scenario}',
                        grid_mapping='spatial_ref', state_timing=source.state_timing))
                    for layer,name in enumerate(layers):
                        for begin in range(0,base.shape[0],32):
                            moisture[begin:begin+32,layer] = source[name][begin:begin+32]
                    common = set(first.dimensions)|set(layers)|{'spatial_ref','precipitation','effective_precipitation'}
                    for name,variable in source.variables.items():
                        if name not in common:
                            _copy_variable(result,variable,f'{name}_{scenario}')
                    for key in ('diffusivity_factor','sm_max_factor','sm_min_factor','root_beta'):
                        if key in source.ncattrs():
                            result.setncattr(f'{key}_{scenario}',source.getncattr(key))
            for name,(path,variable_name) in auxiliaries.items():
                auxiliary_grids[name] = auxiliary_grid_contract(first_path,path)
                with nc.Dataset(path) as source:
                    variable = source[variable_name]
                    for dim in variable.dimensions:
                        if dim in result.variables and dim not in {'time','layer'}:
                            if not np.array_equal(source[dim][:], result[dim][:]):
                                raise ValueError(f'Auxiliary {name} is not on the output grid')
                    indices = _time_indices(source,result) if 'time' in variable.dimensions else None
                    copied = _copy_variable(result,variable,name,indices)
                    copied.setncattr('grid_mapping', 'spatial_ref')
            result.auxiliary_grid_metadata_json = json.dumps(auxiliary_grids, sort_keys=True)
            result.Conventions = 'CF-1.8'
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    paths = list(scenarios.values())+[path for path,_ in auxiliaries.values()]
    manifest = input_manifest(config or {}, paths)
    manifest['scenarios'] = {key:str(value) for key,value in scenarios.items()}
    manifest['auxiliary_grids'] = auxiliary_grids
    save_output_manifest(output,manifest)
    return output


def run_default_and_calibrated(*, output_dir, output_file='SWEB_RZSM.nc', calibration_file=None,
                               auxiliary_inputs=None, **kwargs):
    """Use identical forcing, initial-state convention and soil for both runs."""
    output_dir = Path(output_dir)
    scenarios = {}
    options = dict(kwargs)
    for scenario in ('default', 'calibrated') if calibration_file else ('default',):
        scenario_options = dict(options)
        if scenario == 'calibrated':
            scenario_options['calibration_file'] = str(calibration_file)
        scenarios[scenario] = run_swb_workflow(**scenario_options,
            output_dir=str(output_dir/'intermediate'/scenario), output_file='SWB.nc')
    auxiliaries = dict(auxiliary_inputs or {})
    for key,name,default in [('et','et_forcing','et'),('transpiration','transpiration_forcing','E_t'),('ndvi','ndvi','ndvi')]:
        if options.get(key):
            auxiliaries[name] = (options[key],options.get(key+'_var',default))
    for name,filename in SOIL_FILE_STEMS.items():
        auxiliaries['soil_'+name] = (str(Path(options['soil_dir'])/filename),name)
    return combine_scenarios(scenarios,output_dir/output_file,auxiliaries,
        config={**options,'calibration_file':str(calibration_file) if calibration_file else None})
