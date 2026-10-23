"""
Script: site_evaluation.py
Objective: Evaluate one site against processed observations with explicit support operators.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-23
Last updated: 2026-10-23
Inputs: Accepted default product, frozen mask, processed daily observations and site plan.
Outputs: Exact paired samples, metrics, exclusions, point series and native 2025 map arrays.
Usage: from pysweb.site_evaluation import evaluate_site
Dependencies: numpy, pandas, xarray, pyproj, pysweb.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from pysweb.evaluation import extract_point, depth_weights, metrics
from pysweb.io.provenance import atomic_json, sha256

BOTTOMS = np.array([50.,150.,300.,600.,1000.])


def align_day(frame, offset_hours):
    """Integrate piecewise linear states and uniform UTC fluxes to station days."""
    a = float(offset_hours)/24
    if not 0 <= a < 1:
        raise ValueError('This operator supports UTC and positive standard-time offsets')
    frame = frame.copy()
    good = frame.continuous_trajectory_valid.eq(True)
    frame['paired_valid'] = good & good.shift(1,fill_value=False) if a else good
    for name in ('actual_et','et_forcing'):
        frame[name+'_aligned'] = a*frame[name].shift(1)+(1-a)*frame[name] if a else frame[name]
    for layer in range(1,6):
        start = frame[f'start_layer_{layer}']
        end = 2*frame[f'dailymean_layer_{layer}']-start
        frame[f'aligned_layer_{layer}'] = (.5*a*a*start.shift(1)+(.5+a-a*a)*start+
                                          .5*(1-a)**2*end) if a else (start+end)/2
    return frame


def observation_groups(raw, source):
    groups, excluded = [], []
    if source == 'FLUXNET':
        et = raw.loc[raw.variable.eq('ET_from_LE_F_MDS')]
        if len(et):
            groups.append(dict(variable='ET_from_LE_F_MDS',values=et,interval_mm=None,
                operator='Tower flux compared with fixed native station pixel',units='mm day-1',layer=None))
        for key, values in raw.loc[raw.variable.str.startswith('SWC_')].groupby(
                ['variable','depth_cm','deployment_group'],dropna=False):
            depth = key[1]*10
            if not np.isfinite(depth) or not 0 <= depth <= 1000:
                excluded.append(dict(variable=str(key),reason='Unknown depth or outside 0–100 cm profile',n=0))
                continue
            layer = min(int(np.searchsorted(BOTTOMS,depth,side='left')),4)
            groups.append(dict(variable=str(key),values=values,interval_mm=np.r_[0,BOTTOMS][layer:layer+2].tolist(),
                operator='Point probe approximated by containing model layer; not a layer-average observation',
                units='m3 m-3',layer=layer+1))
    else:
        for sensor, values in raw.loc[raw.variable.eq('soil_moisture')].groupby('sensor_id'):
            bounds = values[['depth_from_cm','depth_to_cm']].drop_duplicates().to_numpy(float)
            if bounds.shape != (1,2):
                raise ValueError(f'Sensor depth changes without a new deployment: {sensor}')
            low,high = bounds[0]*10
            # Cosmic-ray and other footprint estimates have no fixed depth operator.
            if values.layer.eq('footprint_review').any() or values.support.str.contains('footprint|cosmic',case=False,na=False).any():
                excluded.append(dict(variable=str(sensor),reason='Footprint/effective depth requires an independently validated operator',n=0))
                continue
            if not np.isfinite([low,high]).all() or not 0 <= low <= high <= 1000:
                excluded.append(dict(variable=str(sensor),reason='Unknown interval or outside model profile',n=0))
                continue
            description = 'Reported depth interval; overlap-weight model layers'
            layer = None
            if low == high:
                layer = min(int(np.searchsorted(BOTTOMS,low,side='left')),4)
                low,high = np.r_[0,BOTTOMS][layer:layer+2]
                layer += 1
                description = 'Point probe approximated by containing model layer; not a layer-average observation'
            groups.append(dict(variable=str(sensor),values=values,interval_mm=[float(low),float(high)],
                operator=description,units='m3 m-3',layer=layer))
    return groups, excluded


def export_maps(product, mask, destination, location, year):
    """Stream complete-history masks and monthly means; retain missing cells."""
    months = [1,4,7,10]
    with xr.open_dataset(product,decode_coords='all') as ds:
        dates = pd.DatetimeIndex(ds.time.values)
        continuous = mask.values.copy()
        totals = {m:np.zeros((3,)+continuous.shape) for m in months}
        counts = {m:np.zeros(continuous.shape,dtype='int32') for m in months}
        weights = depth_weights(ds.layer_bottom.values,[30.,750.])
        for begin in range(0,len(dates),32):
            stop = min(begin+32,len(dates))
            flags = ds.valid_forcing_fraction_default.isel(time=slice(begin,stop)).values
            valid = np.logical_and.accumulate(np.isfinite(flags)&(flags >= 1-1e-8),axis=0)&continuous[None]
            continuous = valid[-1]
            indices = np.flatnonzero(dates[begin:stop].year == year)
            if not len(indices):
                continue
            select = begin+indices
            sm = ds.rzsm_default.isel(time=select).transpose('time','layer','y','x').values
            storage = np.sum(sm*weights[None,:,None,None],axis=1)
            actual = ds.actual_et_default.isel(time=select).values
            forcing = ds.et_forcing.isel(time=select).values
            values = np.stack([storage,actual,forcing],axis=1).astype(float)
            good = valid[indices]&np.isfinite(values).all(axis=1)
            for month in months:
                use = dates[select].month == month
                totals[month] += np.where(good[use,None],values[use],0).sum(axis=0)
                counts[month] += good[use].sum(axis=0)
        arrays = dict(x=ds.x.values,y=ds.y.values,months=np.array(months),counts=np.stack([counts[m] for m in months]))
        for j,name in enumerate(('storage','actual_et','et_forcing')):
            arrays[name] = np.stack([np.divide(totals[m][j],counts[m],out=np.full(continuous.shape,np.nan),
                where=counts[m] == pd.Period(f'{year}-{m:02d}').days_in_month) for m in months])
        checks = []
        for row,col in ((0,0),(ds.sizes['y']//2,ds.sizes['x']//2),(ds.sizes['y']-1,ds.sizes['x']-1)):
            point = ds.isel(y=row,x=col).sel(time=slice(f'{year}-01-01',f'{year}-12-31')).load()
            for mi,m in enumerate(months):
                sample = point.sel(time=point.time.dt.month == m)
                if np.isfinite(arrays['storage'][mi,row,col]):
                    direct = np.sum(sample.rzsm_default.transpose('time','layer').values*weights[None],axis=1).mean()
                    np.testing.assert_allclose(arrays['storage'][mi,row,col],direct,atol=1e-10,rtol=0)
                    np.testing.assert_allclose(arrays['actual_et'][mi,row,col],sample.actual_et_default.values.astype(float).mean(),atol=1e-12,rtol=0)
                    checks.append(dict(row=row,col=col,month=m,passed=True))
        marker = Transformer.from_crs('EPSG:4326',ds.rio.crs,always_xy=True).transform(*location)
        bounds = list(ds.rio.bounds())
        metadata = dict(crs=str(ds.rio.crs),bounds_m=bounds,affine=list(ds.rio.transform())[:6],
            marker_xy_m=list(marker),marker_lonlat=location,
            extent_km=[(bounds[0]-marker[0])/1000,(bounds[2]-marker[0])/1000,
                       (bounds[1]-marker[1])/1000,(bounds[3]-marker[1])/1000],
            storage_depth_mm=[30,750],year=year,shape=list(continuous.shape),
            statistic='Monthly mean start-of-day storage and daily ET; complete-month history support',
            finite_cells_per_month={k:np.isfinite(arrays[k]).sum(axis=(1,2)).tolist() for k in ('storage','actual_et','et_forcing')})
        dictionary = {name:dict(dimensions=list(var.dims),dtype=str(var.dtype),attributes={k:str(v) for k,v in var.attrs.items()}) for name,var in ds.variables.items()}
    np.savez_compressed(destination/f'maps_{year}.npz',**arrays)
    atomic_json(destination/'map_metadata.json',metadata)
    atomic_json(destination/'map_checks.json',dict(passed=True,checks=checks,scope='All support masks; direct spot checks where finite'))
    atomic_json(destination/'data_dictionary.json',dictionary)


def evaluate_site(plan):
    base = Path(plan['base'])
    destination = base/'evaluation'
    destination.mkdir(parents=True,exist_ok=True)
    product = base/'products/SWEB_RZSM.nc'
    definition = plan['observation_operator']
    raw = pd.read_parquet(plan['observation_file'])
    assert sha256(plan['observation_file']) == plan['observation_sha256'], 'Processed observations changed'
    assert sha256(plan['observation_manifest']) == plan['observation_manifest_sha256']
    basis = 'UTC' if definition['source']=='ISMN' else 'local_standard_time_no_DST'
    if set(raw.time_basis) != {basis}:
        raise ValueError('Observation time basis differs from configured operator')
    phases = dict(development=[plan['config_values']['run']['analysis_start'],plan['development_end']],
                  evaluation=[plan['config_values']['run']['holdout_start'],plan['config_values']['run']['end']])
    with xr.open_dataset(base/'run_inputs/initialized/fit_mask.nc') as ds:
        mask = ds.fit_mask.load().astype(bool)
    location = definition['location']
    unavailable = None
    try:
        frame,operator = extract_point(product,longitude=location[0],latitude=location[1],
            depth_interval_mm=[0,1000],admitted_mask=mask,scenario='default')
    except ValueError as error:
        if 'outside the model grid' not in str(error) and 'outside the frozen development mask' not in str(error):
            raise
        unavailable = str(error)
        # Diagnostics use the explicitly labelled centre; evaluation never does.
        location = plan['config_values']['run']['center']
        frame,operator = extract_point(product,longitude=location[0],latitude=location[1],
            depth_interval_mm=[0,1000],admitted_mask=None,scenario='default')
    frame = align_day(frame,definition['utc_offset_hours'])
    frame.to_csv(destination/'point_series.csv',float_format='%.17g',index_label='date')
    atomic_json(destination/'point_operator.json',dict(**operator,diagnostic_only=bool(unavailable),
        observation_location=definition['location'],temporal_operator=definition,
        spatial_limitation='Fixed native pixel differs from tower or sensor footprint'))
    groups,excluded = observation_groups(raw,definition['source'])
    if unavailable:
        excluded.append(dict(variable='all',reason=unavailable,n=0))
    rows,samples,operators,coverage = [],[],[],[]
    for number,group in enumerate(groups):
        values = group['values'].copy()
        values['date'] = pd.to_datetime(values.date)
        values = values.set_index('date').sort_index()
        if not values.index.is_unique:
            raise ValueError(f'Duplicate observation dates: {group["variable"]}')
        spec = {k:v for k,v in group.items() if k!='values'}
        spec['operator_id'] = f'operator_{number:03d}'
        operators.append(spec)
        interval = group['interval_mm']
        weights = depth_weights(BOTTOMS,interval) if interval else None
        for qc in ('clean','strict'):
            pairs = values[[f'value_{qc}']].rename(columns={f'value_{qc}':'observed'})
            model = frame.actual_et_aligned if interval is None else frame[[f'aligned_layer_{i}' for i in range(1,6)]] @ (weights/weights.sum())
            pairs['default'] = model.where(frame.paired_valid & (not unavailable))
            if interval is None:
                pairs['SSEBop_ET_forcing'] = frame.et_forcing_aligned.where(frame.paired_valid & (not unavailable))
            for phase,(start,end) in phases.items():
                window = pairs.loc[start:end]
                common = window.replace([np.inf,-np.inf],np.nan).dropna().astype('float64')
                coverage.append(dict(operator_id=spec['operator_id'],qc=qc,phase=phase,
                    observed_days=int(np.isfinite(window.observed).sum()),matched_days=len(common),
                    first_match=str(common.index.min().date()) if len(common) else None,
                    last_match=str(common.index.max().date()) if len(common) else None,
                    reason=None if len(common) else unavailable or 'No shared finite observation/model support'))
                for model_name in common.columns.drop('observed'):
                    result = metrics(common[model_name],common.observed)
                    if len(common):
                        error = common[model_name].to_numpy()-common.observed.to_numpy()
                        np.testing.assert_allclose([result['bias'],result['rmse']],[error.mean(),np.sqrt((error**2).mean())],atol=1e-12,rtol=0)
                    rows.append(dict(operator_id=spec['operator_id'],variable=group['variable'],qc=qc,phase=phase,
                        model=model_name,units=group['units'],**result))
                samples.append(common.reset_index().assign(operator_id=spec['operator_id'],qc=qc,phase=phase))
    columns = ['date','observed','default','SSEBop_ET_forcing','operator_id','qc','phase']
    paired = pd.concat(samples,ignore_index=True).reindex(columns=columns) if samples else pd.DataFrame(columns=columns)
    paired.to_csv(destination/'observation_pairs.csv.gz',index=False,float_format='%.17g')
    scores = pd.DataFrame(rows,columns=['operator_id','variable','qc','phase','model','units','n','rmse','mae','bias','unbiased_rmse','pearson_r'])
    scores.to_csv(destination/'observation_metrics.csv',index=False,float_format='%.17g')
    # Independently reread the delivered pairs and reproduce every delivered score.
    reread = pd.read_csv(destination/'observation_pairs.csv.gz',float_precision='round_trip')
    for row in rows:
        selected = reread.loc[reread.operator_id.eq(row['operator_id'])&reread.qc.eq(row['qc'])&reread.phase.eq(row['phase'])]
        check = metrics(selected[row['model']],selected.observed)
        assert check['n']==row['n']
        for key in ('rmse','mae','bias','unbiased_rmse','pearson_r'):
            if row[key] is not None:
                np.testing.assert_allclose(check[key],row[key],rtol=0,atol=1e-12)
    atomic_json(destination/'coverage.json',dict(comparisons=coverage,exclusions=excluded))
    atomic_json(destination/'observation_evaluation.json',dict(passed=True,scores_available=any(r['n'] for r in rows),
        source=plan['observation_file'],source_sha256=plan['observation_sha256'],
        manifest_sha256=plan['observation_manifest_sha256'],operators=operators,exclusions=excluded,
        windows=phases,time_operator=definition,exact_exported_pairs_rechecked=True,
        in_situ_used_for_calibration=False,sample_rule='Finite default/observation pairs; ET forcing uses identical dates',
        limitations=['Daily time alignment approximates subdaily variability','GSSM training overlap unresolved',
                    'Nearby/collocated stations are not independent spatial replicates','WorldCover 2021 is static across all years']))
    with xr.open_dataset(product,decode_coords='all') as ds:
        point = ds.sel(x=operator['selected_x'],y=operator['selected_y']).load()
        lon,lat = Transformer.from_crs(operator['crs'],'EPSG:4326',always_xy=True).transform(operator['selected_x'],operator['selected_y'])
        heatmap = xr.Dataset(coords=dict(time=('time',frame.index.to_numpy()),lat=[lat],lon=[lon]))
        for i,bottom in enumerate(point.layer_bottom.values,1):
            heatmap[f'rzsm_layer_{i}'] = (('time','lat','lon'),frame[f'start_layer_{i}'].where(frame.continuous_trajectory_valid).to_numpy()[:,None,None])
            heatmap[f'rzsm_layer_{i}'].attrs.update(units='m3 m-3',depth_bottom_mm=float(bottom),layer_thickness_mm=float(bottom-(0 if i==1 else point.layer_bottom.values[i-2])))
        heatmap.attrs.update(source_product=str(product),state_timing='start',extraction=json.dumps(operator))
        heatmap.to_netcdf(destination/'heatmap_input.nc')
        atomic_json(destination/'hydraulics.json',dict(wilting_point=point.soil_wilting_point.values.tolist(),porosity=point.soil_porosity.values.tolist()))
    export_maps(product,mask,destination,definition['location'],plan.get('map_year',2025))
