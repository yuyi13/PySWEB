#!/usr/bin/env python3
"""
Script: time_reindex.py
Objective: Stream numeric NetCDF variables onto an explicit time-index mapping with bounded memory.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: A flat numeric NetCDF dataset, output dates and source indices (-1 for missing).
Outputs: An atomic NetCDF copy with repeated/reordered/missing time slices and retained metadata.
Usage: reindex_netcdf_time(source, destination, dates, source_indices)
Dependencies: netCDF4, numpy, pandas; no dask requirement.
"""
from pathlib import Path
import uuid
import netCDF4
import numpy as np
import pandas as pd


def reindex_netcdf_time(source, destination, dates, source_indices, *,
                        time_dim='time', chunk_days=32, attributes=None):
    """Copy values without interpolation, keeping one temporal block in memory.

    Time must be the first dimension of time-dependent numeric variables.
    Packed data are copied in raw encoded units. Source index-1 produces the
    variable's fill value. Other dimensions, coordinates and attributes persist.
    A failure leaves its unique partial file for diagnosis and does not replace
    an existing destination. This helper supports flat numeric datasets only.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Source and destination must differ')
    dates = pd.DatetimeIndex(dates)
    index = np.asarray(source_indices)
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing or not len(dates):
        raise ValueError('Output dates must be nonempty, unique and increasing')
    if dates.tz is not None:
        raise ValueError('Use explicit timezone-naive UTC output dates')
    if index.ndim != 1 or index.dtype.kind not in 'iu' or len(index) != len(dates):
        raise ValueError('Source indices must be one integer per output date')
    if not isinstance(chunk_days, int) or chunk_days < 1:
        raise ValueError('chunk_days must be a positive integer')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.partial')
    with netCDF4.Dataset(source) as src:
        if src.groups or time_dim not in src.dimensions or time_dim not in src.variables:
            raise ValueError('A flat dataset with an explicit time coordinate is required')
        if np.any(index < -1) or np.any(index >= len(src.dimensions[time_dim])):
            raise ValueError('Source time index outside available data')
        for name, variable in src.variables.items():
            if np.dtype(variable.dtype).kind not in 'iuf':
                raise ValueError(f'Unsupported nonnumeric variable: {name}')
            if time_dim in variable.dimensions and variable.dimensions[0] != time_dim:
                raise ValueError(f'Time must be the first dimension: {name}')
        src.set_auto_maskandscale(False)
        with netCDF4.Dataset(temporary, 'w', format='NETCDF4') as dst:
            dst.setncatts({name:src.getncattr(name) for name in src.ncattrs() if name != '_NCProperties'})
            if attributes:
                dst.setncatts(attributes)
            dst.setncattr('time_reindex_method', 'Exact indexed copy in bounded temporal blocks; no interpolation')
            for name, dimension in src.dimensions.items():
                dst.createDimension(name, len(dates) if name == time_dim else len(dimension))
            for name, variable in src.variables.items():
                attrs = {key:variable.getncattr(key) for key in variable.ncattrs() if key != '_FillValue'}
                if name == time_dim:
                    target = dst.createVariable(name, 'f8', (time_dim,), fill_value=False)
                    for key in ['scale_factor','add_offset','missing_value','_Unsigned']:
                        attrs.pop(key,None)
                    attrs.update(units='days since 1970-01-01 00:00:00', calendar='proleptic_gregorian')
                    target.setncatts(attrs)
                    target[:] = (dates-pd.Timestamp('1970-01-01')).total_seconds().to_numpy()/86400.
                    continue
                dtype = np.dtype(variable.dtype)
                fill = variable.getncattr('_FillValue') if '_FillValue' in variable.ncattrs() else (
                    np.nan if dtype.kind == 'f' else netCDF4.default_fillvals[dtype.kind+str(dtype.itemsize)])
                timed = time_dim in variable.dimensions
                if not timed and '_FillValue' not in variable.ncattrs():
                    fill = False
                options = {}
                if timed:
                    options = dict(zlib=True, complevel=3,
                        chunksizes=(min(chunk_days,len(dates)),)+tuple(variable.shape[1:]))
                target = dst.createVariable(name, variable.dtype, variable.dimensions, fill_value=fill, **options)
                target.setncatts(attrs)
                target.set_auto_maskandscale(False)
                if not timed:
                    target[...] = variable[...]
                    continue
                for start in range(0, len(index), chunk_days):
                    selected = index[start:start+chunk_days]
                    block = np.full((len(selected),)+variable.shape[1:], fill, dtype=dtype)
                    available = selected >= 0
                    if available.any():
                        block[available] = variable[selected[available], ...]
                    target[start:start+len(selected), ...] = block
    temporary.replace(destination)
    return destination
