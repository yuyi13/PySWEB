#!/usr/bin/env python3
"""
Script: test_time_reindex.py
Objective: Verify bounded indexed NetCDF copying against explicit array results and packing semantics.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Small numeric datasets with missing values, repeated indices and packed data.
Outputs: Pytest equality, metadata and invalid-input checks.
Usage: pytest tests/test_time_reindex.py
Dependencies: numpy, pandas, xarray, netCDF4, pytest and pysweb.
"""
import netCDF4
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pysweb.io.time_reindex import reindex_netcdf_time


def test_repeated_reordered_and_missing_slices(tmp_path):
    dates = pd.date_range('2014-01-01', periods=5)
    values = np.arange(30,dtype='float32').reshape(5,2,3)
    values[2,0,1] = np.nan
    ds = xr.Dataset(dict(et=(('time','lat','lon'),values), spatial_ref=((),0)),
                    coords=dict(time=dates,lat=[-30.,-31.],lon=[149.,150.,151.]), attrs=dict(role='fixture'))
    ds.et.attrs['units'] = 'mm day-1'
    ds.to_netcdf(tmp_path/'source.nc')
    index = np.asarray([3,4,0,1,2,3,4,-1,0])
    output_dates = pd.date_range('2009-12-26',periods=len(index))
    result = reindex_netcdf_time(tmp_path/'source.nc',tmp_path/'result.nc',output_dates,index,chunk_days=2)
    with xr.open_dataset(result) as actual:
        expected = np.concatenate([values[[3,4,0,1,2,3,4]],np.full((1,2,3),np.nan,dtype='float32'),values[:1]])
        np.testing.assert_array_equal(actual.et.values,expected)
        np.testing.assert_array_equal(actual.time.values,output_dates.values)
        np.testing.assert_array_equal(actual.lat.values,ds.lat.values)
        assert actual.et.attrs['units']=='mm day-1' and actual.attrs['role']=='fixture'
        assert int(actual.spatial_ref)==0


def test_packed_values_not_scaled_twice(tmp_path):
    source = tmp_path/'packed.nc'
    with netCDF4.Dataset(source,'w') as ds:
        ds.createDimension('time',3)
        time = ds.createVariable('time','f8',('time',))
        time.units='days since 2000-01-01'
        time[:]=[0,1,2]
        data=ds.createVariable('rain','i2',('time',),fill_value=-9999)
        data.setncatts(dict(scale_factor=.1,add_offset=2.))
        data.set_auto_maskandscale(False)
        data[:]=np.asarray([3,7,11],dtype='int16')
    result=reindex_netcdf_time(source,tmp_path/'copy.nc',pd.date_range('1999-12-01',periods=5),
                               np.asarray([2,0,2,-1,1]),chunk_days=2)
    with xr.open_dataset(result) as ds:
        np.testing.assert_allclose(ds.rain.values,[3.1,2.3,3.1,np.nan,2.7],atol=1e-12)


@pytest.mark.parametrize('indices', [[0,9],[0,-2],[0.,1.]])
def test_invalid_index_rejected_without_replacing_destination(tmp_path,indices):
    xr.Dataset(dict(v=('time',[1.,2.])),coords=dict(time=pd.date_range('2000-01-01',periods=2))).to_netcdf(tmp_path/'source.nc')
    output=tmp_path/'result.nc'
    output.write_bytes(b'preserve existing output')
    with pytest.raises(ValueError):
        reindex_netcdf_time(tmp_path/'source.nc',output,pd.date_range('1999-01-01',periods=2),np.asarray(indices))
    assert output.read_bytes()==b'preserve existing output'
