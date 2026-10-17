"""
Script: test_native_workflow.py
Objective: Exercise both native preparation grids, calibration and paired products offline.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Small synthetic UTM SEB, geographic rain/reference and custom hydraulic data.
Outputs: Full workflow, holdout and consolidated output-contract assertions.
Usage: python -m pytest tests/workflows/test_native_workflow.py
Dependencies: numpy, pandas, pytest, rasterio, xarray, pysweb
"""
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio import Affine
from rasterio.warp import transform_bounds
from pysweb.workflow import run_config
from pysweb.workflow_native import native_plan
from pysweb.io.products import assert_identical_grids
from tests.io.test_native_grids import raster


def configuration(tmp_path):
    return {'run':{'name':'native','root':str(tmp_path),'start':'2020-01-01','end':'2020-01-06',
                   'extent':[9.001,55.031,9.004,55.034], 'workers':1,'state_timing':'end',
                   'grid_mode':'native','sensors':['LC08','LC09'],'holdout_start':'2020-01-05'},
            'soil':{'source':'custom'},
            'calibration':{'enabled':True,'start':'2020-01-02','end':'2020-01-04',
                           'max_iter':0,'no_polish':True,'allow_unconverged':True,'seed':2},
            'model':{},'ssebop':{}}


def test_native_plan_rejects_holdout_tuning(tmp_path):
    config = configuration(tmp_path)
    config['calibration']['end'] = '2020-01-05'
    with pytest.raises(ValueError,match='holdout'):
        native_plan(config,['calibrate'])
    config['calibration']['end'] = '2020-01-04'
    config['ssebop']['climatology_end'] = '2020-01-06'
    with pytest.raises(ValueError,match='climatology'):
        native_plan(config,['ssebop'])


def test_native_full_offline_workflow(tmp_path):
    config = configuration(tmp_path)
    base = tmp_path/'native'
    (base/'ssebop').mkdir(parents=True)
    (base/'inputs/met/era5land/stack').mkdir(parents=True)
    grid = raster(np.ones((120,120)),Affine(30,0,498200,0,-30,6101800))
    final_grid = grid.isel(y=slice(55,65),x=slice(55,65))
    # Slightly inset geographic AOI still selects complete native UTM pixels.
    bounds = transform_bounds(grid.rio.crs,'EPSG:4326',*final_grid.rio.bounds(),densify_pts=21)
    config['run']['extent'] = list(bounds)
    dates = pd.date_range('2020-01-01','2020-01-06')
    daily = grid.expand_dims(time=dates)
    seb = xr.Dataset({'ET':daily*2,'E_t':daily*1.3,'E_s':daily*.7,
                      'ndvi_interp':daily*.65,'f_t':daily*.65,'elevation':grid*100})
    seb.rio.write_crs(grid.rio.crs).rio.write_transform(grid.rio.transform()).to_netcdf(base/'ssebop/et_daily_ssebop_2020-01-01_2020-01-06.nc')
    # Native reference cells are around 1 km; none are expanded into 30 m observations.
    geographic = raster(np.ones((4,4)),Affine(.008983152841195215,0,8.982,0,-.008983152841195215,55.064),'EPSG:4326')
    reference = (geographic*.25).expand_dims(time=dates[:4]).rename('reference_ssm')
    reference.attrs['units'] = 'm3 m-3'
    reference.to_netcdf(base/'inputs/reference_ssm_native.nc')
    source = geographic.rename({'y':'lat','x':'lon'}).rio.set_spatial_dims(x_dim='lon',y_dim='lat')
    rain = (source*5).expand_dims(time=pd.date_range('2020-01-01','2020-01-31')).rename('precipitation')
    rain.attrs['units'] = 'mm day-1'
    rain.to_netcdf(base/'inputs/met/era5land/stack/precipitation_daily_2020-01-01_2020-01-31.nc')
    fields = {'porosity':(.45,'m3 m-3'),'wilting_point':(.12,'m3 m-3'),
              'available_water_capacity':(.2,'m3 m-3'),'b_coefficient':(4.5,'1'),
              'conductivity_sat':(8.,'mm day-1')}
    soil = xr.Dataset()
    for name,(value,units) in fields.items():
        soil[name] = (source*value).expand_dims(layer=[1,2])
        soil[name].attrs.update(units=units,layer_bottoms_mm=[50.,200.])
    soil = soil.assign_coords(layer_depth=('layer',[50.,200.])).rio.write_crs('EPSG:4326')
    soil.to_netcdf(base/'inputs/soil.nc')
    config['soil']['file'] = str(base/'inputs/soil.nc')
    run_config(config,stages=['preprocess','calibrate','run'])
    seb_path,swb_path = base/'products/SSEBop_ET.nc',base/'products/SWEB_RZSM.nc'
    assert assert_identical_grids(seb_path,swb_path)['identical']
    with xr.open_dataset(swb_path,decode_coords='all') as result:
        assert result.rio.crs.to_epsg() == 32632
        assert {'rzsm_default','rzsm_calibrated','soil_porosity','soil_evaporation_forcing','ndvi'} <= set(result.data_vars)
        assert result.rzsm_default.dims == ('time','layer','y','x')
        assert result.sizes['time'] == 6 and result.sizes['layer'] == 2
        assert not np.allclose(result.rzsm_default,result.rzsm_calibrated)
        for scenario in ('default','calibrated'):
            assert float(np.abs(result[f'balance_residual_{scenario}']).max()) < 1e-5
    with xr.open_dataset(base/'calibration_inputs/reference_ssm_daily_20200101_20200104.nc') as coarse:
        assert coarse.sizes['lat']*coarse.sizes['lon'] < 10
        np.testing.assert_allclose(coarse.reference_ssm,.25)
    parameters = pd.read_csv(base/'calibration.csv')
    assert len(parameters) == 1
    assert set(['diff_factor','sm_max_factor','sm_min_factor','root_beta']) <= set(parameters)
    transfer = json.loads((base/'calibration-transfer.json').read_text())
    assert transfer['scoring_end'] == '2020-01-04'
    # A chip smaller than a GSSM cell cannot support a full-cell transfer score.
    assert transfer['n_cell_days'] == 0
