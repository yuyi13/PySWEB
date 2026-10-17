"""
Script: test_projected_aoi.py
Objective: Prevent metric footprint enlargement and admission of nonintersecting reference cells.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Offline rasters with native southern UTM and geographic grid metadata.
Outputs: Exact crop, coarse-cell admission, preparation and configured workflow assertions.
Usage: python -m pytest tests/test_projected_aoi.py
Dependencies: numpy, pandas, pytest, rasterio, xarray, pysweb
"""
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio import Affine
from rasterio.warp import transform_bounds
from pysweb.io.grids import intersecting_cells, subset_template
from pysweb.swb.preprocess import preprocess_inputs
from tests.io.test_native_grids import raster


BOUNDS = [774141.3598386383,6646126.87310592,779141.3598386383,6651126.87310592]


def test_scene_start_requires_support_in_actual_analysis_area():
    from pysweb.ssebop.scene_protocol import scene_support_calendar
    grid = raster(np.full((3,3),np.nan),Affine(30,0,770000,0,-30,-3345000),'EPSG:32655')
    scenes = grid.expand_dims(time=pd.date_range('2014-01-01',periods=2)).copy()
    scenes.values[0,0,0] = .6
    scenes.values[1,1,1] = .7
    x, y = float(grid.x[1]), float(grid.y[1])
    area = dict(bounds=[x-14,y-14,x+14,y+14],crs='EPSG:32655')
    assert scene_support_calendar(scenes)['first_supported_scene'] == '2014-01-01'
    scoped = scene_support_calendar(scenes,area)
    assert scoped['first_supported_scene'] == '2014-01-02'
    assert scoped['supported_scenes'] == 1 and scoped['support_aoi'] == area
    scenes.values[1,1,1] = np.nan
    with pytest.raises(ValueError,match='No accepted scene support'):
        scene_support_calendar(scenes,area)


def reference_grid():
    # Native source metadata only; all values below are synthetic.
    return raster(np.ones((12,14)),Affine(.008983152841195215,0,149.8120399326126,
        0,-.008983152841195215,-30.210343004939503),'EPSG:4326')


def test_metric_rectangle_retains_native_negative_northings():
    source = raster(np.ones((600,600)),Affine(30,0,770000,0,-30,-3345000),'EPSG:32655')
    geographic = transform_bounds('EPSG:32755','EPSG:4326',*BOUNDS,densify_pts=65)
    inflated = subset_template(source,geographic)
    exact = subset_template(source,BOUNDS,extent_crs='EPSG:32755')
    assert inflated.sizes['x']*30 > 5200 and inflated.sizes['y']*30 > 5200
    assert 5000 <= exact.sizes['x']*30 < 5060 and 5000 <= exact.sizes['y']*30 < 5060
    assert exact.rio.crs.to_epsg() == 32655 and (exact.y < 0).all()
    expected = transform_bounds('EPSG:32755',source.rio.crs,*BOUNDS)
    actual = exact.rio.bounds()
    for delta in [expected[0]-actual[0],expected[1]-actual[1],actual[2]-expected[2],actual[3]-expected[3]]:
        assert -1e-6 <= delta < 30


def test_reference_corner_cells_are_excluded_without_fractional_weights():
    grid = subset_template(reference_grid(),BOUNDS,extent_crs='EPSG:32755')
    mask = intersecting_cells(grid,BOUNDS,'EPSG:32755')
    assert grid.size == 42 and int(mask.sum()) == 37
    assert mask.dtype == bool
    assert np.array_equal(mask.x,grid.x) and np.array_equal(mask.y,grid.y)


def test_preparation_masks_soil_and_retains_native_reference(tmp_path):
    grid = reference_grid()
    dates = pd.date_range('2020-01-01','2020-01-03')
    reference = (grid*.25).expand_dims(time=dates).rename('reference_ssm')
    reference.attrs['units'] = 'm3 m-3'
    reference.to_netcdf(tmp_path/'reference.nc')
    rain = (grid*5).expand_dims(time=pd.date_range('2020-01-01','2020-01-31')).rename('precipitation')
    rain.attrs['units'] = 'mm day-1'
    rain.to_netcdf(tmp_path/'rain.nc')
    fields = {name:(grid*value).expand_dims(time=dates) for name,value in
              dict(ET=2.,E_t=1.3,E_s=.7,ndvi_interp=.65).items()}
    for name in ['ET','E_t','E_s']:
        fields[name].attrs['units'] = 'mm day-1'
    xr.Dataset(fields).rio.write_crs('EPSG:4326').to_netcdf(tmp_path/'et.nc')
    properties = dict(porosity=(.45,'m3 m-3'),wilting_point=(.12,'m3 m-3'),
        available_water_capacity=(.2,'m3 m-3'),b_coefficient=(4.5,'1'),conductivity_sat=(8.,'mm day-1'))
    soil = xr.Dataset()
    for name,(value,units) in properties.items():
        soil[name] = (grid*value).expand_dims(layer=[1,2])
        soil[name].attrs.update(units=units,layer_bottoms_mm=[50.,200.])
    soil.assign_coords(layer_depth=('layer',[50.,200.])).rio.write_crs('EPSG:4326').to_netcdf(tmp_path/'soil.nc')
    extent = list(transform_bounds('EPSG:32755','EPSG:4326',*BOUNDS,densify_pts=65))
    outputs = preprocess_inputs(start_date='2020-01-01',end_date='2020-01-03',extent=extent,
        template_extent=BOUNDS,template_extent_crs='EPSG:32755',grid_mode='reference',
        reference_file=str(tmp_path/'reference.nc'),reference_var='reference_ssm',
        rain_file=str(tmp_path/'rain.nc'),rain_var='precipitation',et_file=str(tmp_path/'et.nc'),et_var='ET',
        soil_evaporation_var='E_s',transpiration_var='E_t',soil_source='custom',soil_file=str(tmp_path/'soil.nc'),
        output_dir=str(tmp_path/'prepared'))
    with xr.open_dataset(tmp_path/'prepared/aoi_support.nc') as ds:
        assert ds.aoi_support.dtype == np.dtype('uint8')
        assert np.isin(ds.aoi_support.values,[0,1]).all()
        mask = ds.aoi_support.values.astype(bool)
    with xr.open_dataset(tmp_path/'prepared/soil_porosity.nc') as ds:
        assert np.isfinite(ds.porosity.values[:,mask]).all()
        assert np.isnan(ds.porosity.values[:,~mask]).all()
    with xr.open_dataset(tmp_path/'prepared/reference_ssm_daily_20200101_20200103.nc') as ds:
        np.testing.assert_allclose(ds.reference_ssm,.25)
        assert ds.reference_ssm.shape == (3,6,7)
    report = json.loads((tmp_path/'prepared/aoi-selection.json').read_text())
    assert report['geometric_window_cells'] == 42 and report['intersecting_cells'] == 37
    assert 'aoi_support.nc' in outputs


def test_configured_exact_final_pair(tmp_path,monkeypatch):
    from tests.workflows import test_native_workflow as existing
    original = existing.configuration
    def configuration(path):
        config = original(path)
        config['run'].update(aoi_bounds=[499850,6099850,500150,6100150],aoi_crs='EPSG:32632')
        return config
    monkeypatch.setattr(existing,'configuration',configuration)
    existing.test_native_full_offline_workflow(tmp_path)
    with xr.open_dataset(tmp_path/'native/products/SWEB_RZSM.nc') as ds:
        assert (ds.sizes['y'],ds.sizes['x']) == (10,10)


def test_config_requires_aoi_bounds_crs_pair_and_envelope(tmp_path):
    from pysweb.workflow import load_config
    path = tmp_path/'run.toml'
    base = '[run]\nname="test"\nroot="runs"\nstart="2014-01-01"\nend="2015-01-01"\nextent=[149.8,-30.4,150.0,-30.1]\n'
    path.write_text(base+'aoi_bounds='+str(BOUNDS)+'\n')
    with pytest.raises(ValueError,match='together'):
        load_config(path)
    path.write_text(base+'aoi_bounds='+str(BOUNDS)+'\naoi_crs="EPSG:32755"\n')
    assert load_config(path)['run']['aoi_crs'] == 'EPSG:32755'
    path.write_text(path.read_text().replace('150.0','149.81'))
    with pytest.raises(ValueError,match='must contain'):
        load_config(path)


def test_smaller_soil_footprint_reuses_verified_native_cache(tmp_path,monkeypatch):
    from pysweb.soil import openlandmap
    from pysweb.io.provenance import sha256
    grid = reference_grid()
    originals = {}
    for key,asset in openlandmap.OPENLANDMAP_DATASETS.items():
        value = grid.expand_dims(band=['b0','b10','b30','b60','b100','b200']).rename(key)
        value.attrs.update(source_asset=asset,native_projection=json.dumps(dict(
            crs='EPSG:4326',transform=list(grid.rio.transform())[:6])))
        path = tmp_path/f'{key}_larger.nc'
        value.to_netcdf(path)
        originals[key] = (path,sha256(path))
    def forbidden(*args,**kwargs):
        raise AssertionError('Completed covering downloads must be reused without Earth Engine')
    monkeypatch.setattr(openlandmap,'_initialize_ee',forbidden)
    bounds = grid.isel(y=slice(3,8),x=slice(3,10)).rio.bounds()
    predictors = openlandmap._load_openlandmap_predictors(tuple(bounds),'unused',cache_dir=tmp_path)
    for key,value in predictors.items():
        assert value.shape == (6,5,7)
        np.testing.assert_equal(value.values,np.ones((6,5,7)))
        assert value.attrs['cache_parent_sha256'] == originals[key][1]
        assert sha256(originals[key][0]) == originals[key][1]
    assert len(list(tmp_path.glob('*.manifest.json'))) == len(openlandmap.OPENLANDMAP_DATASETS)
