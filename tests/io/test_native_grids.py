"""
Script: test_native_grids.py
Objective: Verify source alignment, conservative aggregation and exact final product grids.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
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
from pysweb.io.products import crop_product, assert_identical_grids


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
