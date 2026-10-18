"""
Script: test_native_grids.py
Objective: Verify source alignment, conservative aggregation and exact final product grids.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-18
Inputs: Synthetic projected and geographic rasters and temporary NetCDF files.
Outputs: Geospatial contract assertions.
Usage: python -m pytest tests/io/test_native_grids.py
Dependencies: numpy, pytest, rasterio, rioxarray, xarray, pysweb
"""
from types import SimpleNamespace
import numpy as np
import pytest
import xarray as xr
import rioxarray
from rasterio import Affine
from rasterio.crs import CRS
from pysweb.io.native_grid import verify_native_grid, native_projection, export_parameters
from pysweb.io.grids import align_to_grid, subset_template
from pysweb.io.products import crop_product, assert_identical_grids, auxiliary_grid_contract


def raster(values, transform=Affine(30,0,500000,0,-30,6100000), crs='EPSG:32632'):
    values = np.asarray(values,dtype=float)
    y,x = values.shape
    return xr.DataArray(values,dims=('y','x'),coords={
        'x':transform.c+(np.arange(x)+.5)*transform.a,
        'y':transform.f+(np.arange(y)+.5)*transform.e}).rio.write_crs(crs).rio.write_transform(transform)


def test_native_download_rejects_shift_resolution_and_wrong_crs():
    source = {'crs':'EPSG:32632','transform':[30,0,500000,0,-30,6100000]}
    ds = SimpleNamespace(crs=CRS.from_epsg(32632),transform=Affine(*source['transform'])*Affine.translation(7,9),height=3,width=4)
    assert verify_native_grid(ds,source)['native_grid_verified']
    for transform in (ds.transform*Affine.translation(.5,0),ds.transform*Affine.scale(2)):
        ds.transform = transform
        with pytest.raises(ValueError):
            verify_native_grid(ds,source)
    ds.crs = CRS.from_epsg(4326)
    with pytest.raises(ValueError,match='CRS'):
        verify_native_grid(ds,source)


def test_native_projection_rejects_mixed_band_grids(monkeypatch):
    import ee
    projections = {'red':{'crs':'EPSG:32632','transform':[30,0,0,0,-30,0]},
                   'thermal':{'crs':'EPSG:32632','transform':[60,0,0,0,-60,0]}}
    image = SimpleNamespace(select=lambda names: SimpleNamespace(projection=lambda:projections[names[0]]))
    monkeypatch.setattr(ee,'Dictionary',lambda value:SimpleNamespace(getInfo=lambda:value))
    with pytest.raises(ValueError,match='different native grid'):
        native_projection(image,['red','thermal'])


def test_average_preserves_depth_and_rejects_missing_or_partial_footprints():
    source = raster([[1,2],[3,4]])
    target = raster([[0]],Affine(60,0,500000,0,-60,6100000))
    np.testing.assert_allclose(align_to_grid(source,target,'average'),2.5)
    source.values[0,0] = np.nan
    assert np.isnan(align_to_grid(source,target,'average')).all()
    source = raster(np.full((2,2),5.0))
    partial = raster([[0]],Affine(60,0,499970,0,-60,6100000))
    assert np.isnan(align_to_grid(source,partial,'average')).all()


def test_exact_subsets_keep_reference_values_and_final_affine(tmp_path):
    source = raster(np.arange(30).reshape(5,6))
    source.values[2,2] = np.nan
    target = source.isel(y=slice(1,4),x=slice(1,5))
    aligned = align_to_grid(source,target,'nearest')
    np.testing.assert_equal(aligned.values,target.values)
    times = np.arange('2020-01-01','2020-03-01',dtype='datetime64[D]')
    product = xr.Dataset({'ET':source.expand_dims(time=times),
                          'elevation':source,'qa':source.expand_dims(scene_time=times[::7])})
    product = product.rio.write_crs(source.rio.crs).rio.write_transform(source.rio.transform())
    original, cropped = tmp_path/'original.nc',tmp_path/'cropped.nc'
    product.to_netcdf(original)
    crop_product(original,cropped,target)
    expected = product.isel(y=slice(1,4),x=slice(1,5)).rio.write_transform(target.rio.transform())
    expected.to_netcdf(tmp_path/'expected.nc')
    assert assert_identical_grids(cropped,tmp_path/'expected.nc')['identical']
    with xr.open_dataset(cropped,decode_coords='all') as actual:
        for name in ('ET','elevation','qa'):
            np.testing.assert_equal(actual[name].values,expected[name].values)
    import netCDF4 as nc
    with nc.Dataset(cropped) as actual:
        assert actual['ET'].chunking()[0] <= 32
        assert actual['qa'].chunking()[0] <= 32
    wrong = expected.assign_coords(x=expected.x+15).rio.write_transform(Affine(30,0,500045,0,-30,6099970))
    wrong.to_netcdf(tmp_path/'wrong.nc')
    with pytest.raises(ValueError):
        assert_identical_grids(cropped,tmp_path/'wrong.nc')


def test_global_origin_download_uses_bounded_native_pixel_window():
    projection = {'crs':'EPSG:4326','transform':[1/3600,0,-179.0001388888889,0,-1/3600,61.00013888888889]}
    request = export_parameters(projection,[146.28,-35.00,146.30,-34.98])
    assert request['dimensions'] == [73,73]
    assert 'region' not in request and 'scale' not in request
    grid = SimpleNamespace(crs=CRS.from_epsg(4326),transform=Affine(*request['crs_transform']),width=73,height=73)
    assert verify_native_grid(grid,projection)['native_grid_verified']


def test_auxiliary_affine_roundoff_keeps_final_contract_strict(tmp_path):
    transform = Affine(.008983152841195215,0,146.25471140749926,
                       0,-.008983152841195215,-34.96243085793178)
    source = raster(np.ones((6,8)),transform,'EPSG:4326').rename('field')
    derived = Affine(float((source.x[-1]-source.x[0])/7),0,transform.c,
                     0,float((source.y[-1]-source.y[0])/5),transform.f)
    source.to_netcdf(tmp_path/'canonical.nc')
    source.rio.write_transform(derived).to_netcdf(tmp_path/'derived.nc')
    contract = auxiliary_grid_contract(tmp_path/'canonical.nc',tmp_path/'derived.nc')
    assert contract['identical_coordinates'] and contract['affine_metadata_normalized']
    assert 0 < contract['maximum_affine_adjustment'] < 1e-13
    with pytest.raises(ValueError,match='affine'):
        assert_identical_grids(tmp_path/'canonical.nc',tmp_path/'derived.nc')


@pytest.mark.parametrize('change', ['origin', 'scale', 'coordinate', 'crs'])
def test_auxiliary_grid_rejects_mismatch_despite_metadata_normalization(tmp_path,change):
    source = raster(np.ones((3,4))).rename('field')
    transform = source.rio.transform()
    source.to_netcdf(tmp_path/'canonical.nc')
    if change == 'origin':
        wrong = source.rio.write_transform(Affine(transform.a,0,transform.c+.01,
                                                  0,transform.e,transform.f))
    elif change == 'scale':
        wrong = source.rio.write_transform(Affine(transform.a+.01,0,transform.c,
                                                  0,transform.e,transform.f))
    elif change == 'coordinate':
        wrong = source.assign_coords(x=np.nextafter(source.x.values,np.inf))
    else:
        wrong = source.rio.write_crs('EPSG:32633')
    wrong.to_netcdf(tmp_path/'wrong.nc')
    with pytest.raises(ValueError):
        auxiliary_grid_contract(tmp_path/'canonical.nc',tmp_path/'wrong.nc')
    with pytest.raises(ValueError):
        assert_identical_grids(tmp_path/'canonical.nc',tmp_path/'wrong.nc')


def test_combined_auxiliary_uses_exact_output_metadata_and_preserves_values(tmp_path):
    import json
    from pysweb.swb.paired import combine_scenarios
    transform = Affine(.008983152841195215,0,146.25471140749926,
                       0,-.008983152841195215,-34.96243085793178)
    grid = raster(np.arange(48).reshape(6,8),transform,'EPSG:4326')
    days = np.array(['2014-01-01','2014-01-02'],dtype='datetime64[D]')
    daily = grid.expand_dims(time=days)
    scenario = xr.Dataset(dict(rzsm_layer_1=daily/200,
        precipitation=daily*0,effective_precipitation=daily*0))
    scenario.rzsm_layer_1.attrs['depth_bottom_mm'] = 50.
    scenario.attrs['state_timing'] = 'start'
    scenario.rio.write_crs(grid.rio.crs).rio.write_transform(transform).to_netcdf(tmp_path/'scenario.nc')
    derived = Affine(float((grid.x[-1]-grid.x[0])/7),0,transform.c,
                     0,float((grid.y[-1]-grid.y[0])/5),transform.f)
    auxiliary = grid.rename('porosity').rio.write_transform(derived)
    auxiliary.to_netcdf(tmp_path/'soil.nc')
    result = combine_scenarios({'default':tmp_path/'scenario.nc'},tmp_path/'combined.nc',
        {'soil_porosity':(tmp_path/'soil.nc','porosity')})
    assert assert_identical_grids(result,tmp_path/'scenario.nc')['identical']
    with xr.open_dataset(result) as actual:
        np.testing.assert_array_equal(actual.soil_porosity.values,auxiliary.values)
        assert actual.soil_porosity.attrs['grid_mapping'] == 'spatial_ref'
        assert json.loads(actual.auxiliary_grid_metadata_json)['soil_porosity']['affine_metadata_normalized']
    manifest = json.loads((tmp_path/'combined.nc.manifest.json').read_text())
    assert manifest['auxiliary_grids']['soil_porosity']['identical_coordinates']


def test_normal_run_product_marks_actual_2013_as_warmup(tmp_path):
    import json
    from pysweb.swb.paired import combine_scenarios
    grid = raster(np.ones((2, 2)), Affine(30, 0, 500000, 0, -30, 6100000))
    daily = grid.expand_dims(time=np.array(['2013-12-31', '2014-01-01'], dtype='datetime64[D]'))
    source = xr.Dataset(dict(rzsm_layer_1=daily*.3,
        precipitation=daily*0, effective_precipitation=daily*0))
    source.rzsm_layer_1.attrs['depth_bottom_mm'] = 50.
    source.attrs['state_timing'] = 'start'
    source.rio.write_crs(grid.rio.crs).rio.write_transform(grid.rio.transform()).to_netcdf(tmp_path/'normal.nc')
    initialization = dict(initial_state='physical_field_capacity', mode='actual_forcing_prefix',
        analysis_start='2014-01-01', artificial_warmup_days=0)
    result = combine_scenarios({'default':tmp_path/'normal.nc'}, tmp_path/'combined.nc',
        config={'initialization':initialization})
    with xr.open_dataset(result) as ds:
        assert ds.attrs['scenarios'] == 'default'
        np.testing.assert_array_equal(ds.is_warmup, [1, 0])
        assert json.loads(ds.attrs['initialization_json']) == initialization
        np.testing.assert_array_equal(ds.time, source.time)
    manifest = json.loads((tmp_path/'combined.nc.manifest.json').read_text())
    assert manifest['config']['initialization'] == initialization
