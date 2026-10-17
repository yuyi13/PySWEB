"""
Script: forcing.py
Objective: Enforce development-only forcing support and retain reconstruction provenance.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Scene and daily arrays, fixed development dates and configured thresholds.
Outputs: Reconstructed forcing, support masks and source-quality flags.
Usage: from pysweb.forcing import reconstruct, paired_support
Dependencies: numpy, pandas, xarray, pysweb
"""
import numpy as np
import pandas as pd
import xarray as xr


def daily_support(valid, dates, target, max_gap_days):
    """Support from two actual observations no more than max_gap_days apart."""
    dates, target = pd.DatetimeIndex(dates), pd.DatetimeIndex(target)
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Scene dates must be unique and increasing")
    values = np.asarray(valid, dtype=bool)
    flat = values.reshape(len(dates), -1)
    output = np.zeros((len(target), flat.shape[1]), dtype=bool)
    # Per-pixel bounds distinguish missing observations from catalogue gaps.
    for cell in range(flat.shape[1]):
        observed = dates[flat[:, cell]]
        if not len(observed):
            continue
        left = observed.searchsorted(target, side="right")-1
        right = observed.searchsorted(target, side="left")
        inside = (left >= 0) & (right < len(observed))
        use = np.flatnonzero(inside)
        output[use, cell] = (observed[right[use]]-observed[left[use]]).days <= max_gap_days
    return output.reshape((len(target),)+values.shape[1:])


def reconstruct(scenes, dates, *, development_end, acquisition_max_gap_days,
                original_max_gap_days=32, window_days=32, minimum_samples=5,
                boundary_carry=False, workers=1):
    """Retrospective reconstruction in independent development/test partitions.

    Input etf/ndvi is already QA/padding screened. original_support excludes
    spatial estimates. Boundary carry is explicit and never crosses a partition.
    """
    from pysweb.ssebop.api import gapfill_etf_savgol, interpolate_etf_daily
    dates = pd.DatetimeIndex(dates)
    if not len(dates) or not dates.equals(pd.date_range(dates[0], dates[-1])):
        raise ValueError("Reconstruction requires a complete daily output calendar")
    if acquisition_max_gap_days < original_max_gap_days or original_max_gap_days < 1:
        raise ValueError("Invalid gap limits")
    boundary = pd.Timestamp(development_end)
    pieces = []
    for target in (dates[dates <= boundary], dates[dates > boundary]):
        if not len(target):
            continue
        selected = scenes.sel(time=slice(target[0], target[-1]))
        if not selected.sizes.get("time"):
            raise ValueError("No scene support in a requested temporal partition")
        source = pd.DatetimeIndex(selected.time.values)
        time = xr.DataArray(target, dims="time", coords={"time": target})
        result = {}
        for name, bounds in (("etf", (0, 1)), ("ndvi", (-1, 1))):
            filled = gapfill_etf_savgol(selected[name], window_days, minimum_samples, workers=workers).clip(*bounds)
            daily = interpolate_etf_daily(filled, time, acquisition_max_gap_days)
            if boundary_carry:
                daily = daily.where(time >= source[0], filled.isel(time=0, drop=True))
                daily = daily.where(time <= source[-1], filled.isel(time=-1, drop=True))
            result[name] = daily
        raw = np.isfinite(selected.etf.values) & np.isfinite(selected.ndvi.values)
        original = raw & (selected.original_support.fillna(0).values >= 1-1e-6) if "original_support" in selected else raw
        for name, valid in (("scene_supported", raw), ("original_supported", original)):
            result[name] = result["etf"].copy(data=daily_support(valid, source, target, original_max_gap_days).astype("uint8"))
        result["spatial_fill_needed"] = ((result["scene_supported"] > 0) & (result["original_supported"] == 0)).astype("uint8")
        edge = (time < source[0]) | (time > source[-1])
        result["boundary_extrapolated"] = xr.broadcast(edge & boundary_carry, result["etf"])[0].astype("uint8")
        extended = np.zeros(len(target), dtype="uint8")
        for left, right in zip(source[:-1], source[1:]):
            if (right-left).days > original_max_gap_days:
                extended[(target > left) & (target < right)] = 1
        result["acquisition_gap_extended"] = xr.broadcast(xr.DataArray(extended, dims="time", coords={"time": target}), result["etf"])[0]
        result["reconstructed"] = (np.isfinite(result["etf"]) & np.isfinite(result["ndvi"]) & (result["original_supported"] == 0)).astype("uint8")
        pieces.append(xr.Dataset(result))
    output = xr.concat(pieces, "time")
    output.attrs.update(scientific_status="retrospective reconstruction sensitivity",
        development_end=str(development_end), acquisition_max_gap_days=int(acquisition_max_gap_days),
        original_max_gap_days=int(original_max_gap_days), boundary_carry=int(boundary_carry))
    return output


def complete_support(forcing, soil_valid, development_start, development_end,
                     variables=("effective_precip", "et", "E_t", "ndvi")):
    expected = pd.date_range(development_start, development_end)
    selected = forcing.sel(time=slice(development_start, development_end))
    if not pd.DatetimeIndex(selected.time.values).equals(expected):
        raise ValueError("Development forcing calendar incomplete")
    mask = soil_valid.astype(bool).copy()
    for name in variables:
        field, _ = xr.align(selected[name], mask, join="exact")
        mask &= np.isfinite(field).all("time")
    return mask.rename("development_support")


def paired_support(masks):
    """Intersect exact grids; callers supply development-only masks."""
    if len(masks) < 2:
        raise ValueError("Paired support requires at least two workflows")
    crs = masks[0].rio.crs
    if any(mask.rio.crs != crs for mask in masks[1:]):
        raise ValueError('Paired support CRS differs')
    aligned = xr.align(*masks, join="exact")
    result = aligned[0].astype(bool).copy()
    for mask in aligned[1:]:
        result &= mask.astype(bool)
    if not bool(result.any()):
        raise ValueError("No common complete development forcing/soil cells")
    return result.rename("fit_mask")


def precipitation_qc(precipitation, *, small_negative_tolerance_mm, maximum_annual_correction_mm):
    if small_negative_tolerance_mm < 0 or maximum_annual_correction_mm < 0:
        raise ValueError("Nonnegative precipitation QC thresholds required")
    raw = precipitation
    small = np.isfinite(raw) & (raw < 0) & (raw >= -small_negative_tolerance_mm)
    bad = ~np.isfinite(raw) | (raw < -small_negative_tolerance_mm)
    corrected = xr.where(small, 0, raw).where(~bad).rename(raw.name or "precipitation")
    correction = xr.where(small, -raw, 0).rename("precipitation_correction")
    annual = correction.groupby("time.year").sum("time")
    flag = xr.where(bad, 2, xr.where(small, 1, 0)).astype("uint8").rename("precipitation_qc")
    flag.attrs.update(flag_values=[0, 1, 2], flag_meanings="unchanged small_negative_corrected unavailable")
    record = dict(passed=bool((annual <= maximum_annual_correction_mm).all()),
        small_negative_values=int(small.sum()), unavailable_values=int(bad.sum()),
        maximum_annual_correction_mm=float(annual.max()),
        annual_correction_limit_mm=float(maximum_annual_correction_mm))
    return xr.Dataset({corrected.name: corrected, correction.name: correction, flag.name: flag}), record


def padding_acceptance(primary, comparison, *, minimum_agreement, maximum_p99_difference):
    primary, comparison = xr.align(primary, comparison, join="exact")
    a, b = np.isfinite(primary.values), np.isfinite(comparison.values)
    common = a & b
    agreement = float(np.mean(a == b))
    p99 = float(np.quantile(np.abs(primary.values[common]-comparison.values[common]), .99)) if common.any() else None
    return dict(passed=bool(common.any() and agreement >= minimum_agreement and p99 <= maximum_p99_difference),
                common_cells=int(common.sum()), validity_agreement=agreement, p99_abs_difference=p99)


def seasonal_fraction(values, times, development_start, development_end, output_times,
                      *, half_window=15, minimum_years=2):
    """Circular development-only 365-day ETf control with fixed yearly support.

    February 29 is excluded from fitting and averages February 28/March 1 at
    application. NDVI, reference demand and rainfall are not climatologized.
    """
    times, output_times = pd.DatetimeIndex(times), pd.DatetimeIndex(output_times)
    values = np.asarray(values, dtype=float)
    if values.shape[0] != len(times) or not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('Seasonal values require unique increasing source dates')
    if not 0 <= half_window <= 30 or minimum_years < 1:
        raise ValueError('Invalid seasonal climatology settings')
    keep = (times >= pd.Timestamp(development_start)) & (times <= pd.Timestamp(development_end))
    if not keep.any():
        raise ValueError('No development data for seasonal control')
    shape = (365,)+values.shape[1:]
    total, count, years = np.zeros(shape), np.zeros(shape), np.zeros(shape,dtype='int32')
    for year in sorted(set(times[keep].year)):
        sums, counts = np.zeros(shape), np.zeros(shape)
        for index in np.flatnonzero(keep & (times.year == year)):
            date = times[index]
            if (date.month,date.day) == (2,29):
                continue
            day = pd.Timestamp(2001,date.month,date.day).dayofyear-1
            finite = np.isfinite(values[index])
            sums[day] += np.where(finite,values[index],0.)
            counts[day] += finite
        smoothed_sum = sum(np.roll(sums,shift,axis=0) for shift in range(-half_window,half_window+1))
        smoothed_count = sum(np.roll(counts,shift,axis=0) for shift in range(-half_window,half_window+1))
        total += smoothed_sum
        count += smoothed_count
        years += smoothed_count > 0
    climatology = np.divide(total,count,out=np.full(shape,np.nan),where=(count>0)&(years>=minimum_years))
    output = np.full((len(output_times),)+values.shape[1:],np.nan)
    for index,date in enumerate(output_times):
        output[index] = .5*(climatology[58]+climatology[59]) if (date.month,date.day)==(2,29) else climatology[pd.Timestamp(2001,date.month,date.day).dayofyear-1]
    return output


def write_seasonal_seb(source, destination, *, development_start, development_end,
                       half_window=15, minimum_years=2, block_rows=8):
    """Write a bounded-memory control SEB for the same maintained preprocessing.

    Climatologize ET fraction only. Keep actual NDVI/component partition and
    reference demand. This is a separate control file; the primary is unchanged.
    """
    from pathlib import Path
    import json
    import netCDF4 as nc
    from contextlib import ExitStack
    from pysweb.io.provenance import input_manifest, save_output_manifest
    from pysweb.io.netcdf_stream import stream_chunks, sequential_read_cache
    if block_rows < 1:
        raise ValueError('Positive control block size required')
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = destination.with_suffix('.partial.nc')
    rule = dict(development_start=development_start,development_end=development_end,
                half_window=half_window,minimum_years=minimum_years)
    with xr.open_dataset(source,decode_coords='all') as ds:
        dates = pd.DatetimeIndex(ds.time.values)
        if pd.Timestamp(development_end) > dates[-1]:
            raise ValueError('Seasonal development boundary outside source calendar')
        template = ds[['ET','E_t','E_s','ndvi_interp','f_t','etf_interp','reference_et']]
        # Preserve native coordinates/CRS without first allocating the full cube.
        skeleton = template.isel(time=slice(0,0))
        encoding = {}
        for name in skeleton.data_vars:
            if template[name].dims != ('time', 'y', 'x'):
                raise ValueError('Seasonal SEB fields must have time/y/x dimensions')
            # An explicit encoding avoids xarray dropping chunks because this
            # empty skeleton differs from the source's original_shape metadata.
            encoding[name] = {key:value for key,value in template[name].encoding.items()
                if key in ('dtype', '_FillValue', 'grid_mapping', 'scale_factor', 'add_offset')}
            encoding[name].update(zlib=True, complevel=3,
                chunksizes=stream_chunks(template[name].dims, template[name].shape,
                    block_dim='y', block_size=block_rows))
        skeleton.to_netcdf(temporary,unlimited_dims=['time'],encoding=encoding)
        with nc.Dataset(source) as raw, nc.Dataset(temporary,'a') as output, ExitStack() as caches:
            source_fields = ('etf_interp', 'ndvi_interp', 'reference_et', 'f_t')
            for name in source_fields:
                caches.enter_context(sequential_read_cache(raw[name], 'y'))
            output.variables['time'][:] = raw.variables['time'][:]
            for begin in range(0,ds.sizes['y'],block_rows):
                rows = slice(begin,min(begin+block_rows,ds.sizes['y']))
                block = {name:np.ma.filled(raw[name][:,rows,:], np.nan) for name in source_fields}
                fraction = seasonal_fraction(block['etf_interp'],dates,
                    development_start,development_end,dates,half_window=half_window,minimum_years=minimum_years)
                values = dict(etf_interp=fraction,ndvi_interp=block['ndvi_interp'],
                    reference_et=block['reference_et'],f_t=block['f_t'])
                values['ET'] = fraction*values['reference_et']
                values['E_t'] = values['ET']*values['f_t']
                values['E_s'] = values['ET']*(1-values['f_t'])
                for name,value in values.items():
                    output.variables[name][:,rows,:] = value
            output.setncattr('control','development-only seasonal ET fraction; actual NDVI and reference demand')
            output.setncattr('seasonal_rule_json',json.dumps(rule,sort_keys=True))
            output.setncattr('source_seb',str(Path(source).resolve()))
    temporary.replace(destination)
    save_output_manifest(destination,input_manifest(rule,[source]))
    return destination
