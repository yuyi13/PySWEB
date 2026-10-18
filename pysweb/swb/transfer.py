"""
Script: transfer.py
Objective: Assess coarse-to-fine parameter transfer using development observations only.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-18
Inputs: Final default/calibrated SWB, native GSSM reference and scoring dates.
Outputs: Area-weighted development diagnostics without any further parameter tuning.
Usage: from pysweb.swb.transfer import evaluate_transfer
Dependencies: numpy, pandas, xarray, pysweb
"""
import numpy as np
import pandas as pd
import xarray as xr
from pysweb.io.grids import spatial, align_to_grid
from pysweb.io.provenance import atomic_json
from pysweb.contracts import geographic_weights


def evaluate_transfer(product, reference, calibration_file, start, end, output, surface_depth=50.0):
    sums = {key:0.0 for key in ('default','calibrated')}
    weight_sum, count = 0.0, 0
    with xr.open_dataset(product,decode_coords='all') as source, xr.open_dataset(reference,decode_coords='all') as target:
        from pysweb.swb.calibrate import _surface_layer_index
        layer_index = _surface_layer_index(source['layer_bottom'].values,surface_depth)
        support = (source['valid_forcing_fraction_default'].sel(time=slice(None,end)) == 1).all('time')
        support &= (source['valid_forcing_fraction_calibrated'].sel(time=slice(None,end)) == 1).all('time')
        observed = spatial(target['reference_ssm'].sel(time=slice(start,end)))
        template = observed.isel(time=0,drop=True)
        weights = geographic_weights(template,'y','x')
        for begin in range(0,observed.sizes['time'],32):
            obs = observed.isel(time=slice(begin,begin+32)).load()
            fields = {key:align_to_grid(source[f'rzsm_{key}'].sel(time=obs.time).isel(layer=layer_index).where(support),
                                        template,resampling='average') for key in sums}
            valid = np.isfinite(obs)
            for field in fields.values():
                valid &= np.isfinite(field)
            weight_sum += float(weights.where(valid,0).sum())
            count += int(valid.sum())
            for key,field in fields.items():
                sums[key] += float(((field-obs)**2*weights).where(valid,0).sum())
    calibration = pd.read_csv(calibration_file, float_precision="round_trip").iloc[0]
    report = {'scoring_start':start,'scoring_end':end,'surface_layer':int(layer_index+1),
              'objective':'area-weighted cell-day RMSE', 'n_cell_days':count,
              'coarse_fit_rmse':float(calibration['rmse']),
              'fine_aggregated_rmse':{key:float(np.sqrt(value/weight_sum)) if weight_sum else None for key,value in sums.items()},
              'status':'evaluated' if weight_sum else 'no complete common cell-day support',
              'notes':'Diagnostic only; no fine-grid retuning. Fine outputs cover the final AOI; incomplete GSSM edge footprints are excluded. Coarse-fit and transfer populations may differ.'}
    atomic_json(output,report)
    return report
