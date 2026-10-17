#!/usr/bin/env python3
"""
Script: test_ee_export.py
Objective: Verify explicit-grid export preserves raw QA zeros and rejects ambiguous requests before network access.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Small synthetic GeoTIFF and fake Earth Engine image.
Outputs: pytest assertions.
Usage: pytest tests/io/test_ee_export.py
Dependencies: pytest, rasterio, numpy, pysweb.
"""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from pysweb.io import ee_export


class Image:
    def select(self,bands):
        self.bands=bands
        return self
    def getDownloadURL(self,params):
        self.params=params
        return 'mock-url'


def test_raw_qa_export(monkeypatch,tmp_path):
    def download(url,path,**kwargs):
        with rasterio.open(path,'w',driver='GTiff',height=2,width=2,count=2,dtype='uint16',crs='EPSG:32755',transform=from_origin(500000,6000000,30,30)) as dst:
            dst.write(np.array([[[64,8],[64,128]],[[0,0],[512,0]]],dtype='uint16'))
    monkeypatch.setattr(ee_export,'_retry_download',download)
    monkeypatch.setattr(ee_export,'native_projection',lambda image,bands:{'crs':'EPSG:32755','transform':[30,0,500000,0,-30,6000000],'bands':bands})
    img=Image()
    path=tmp_path/'scene.tif'
    record=ee_export.download_geotiff(img,path,parameters={'region':[1,2,3,4],'crs':'EPSG:32755','crs_transform':[30,0,500000,0,-30,6000000]},bands=['QA_PIXEL','QA_RADSAT'],source={'asset':'mock'})
    with rasterio.open(path) as src:
        assert src.nodata is None
        assert src.read(2,masked=True).count()==4
        assert src.descriptions==('QA_PIXEL','QA_RADSAT')
    assert len(record['sha256'])==64
    assert 'scale' not in img.params


@pytest.mark.parametrize('params',[{'region':[], 'crs':'EPSG:32755'}, {'region':[], 'crs':'EPSG:32755','scale':30,'crs_transform':[30,0,0,0,-30,0]}])
def test_ambiguous_grid_rejected(tmp_path,params):
    with pytest.raises(ValueError,match='exactly one'):
        ee_export.download_geotiff(Image(),tmp_path/'bad.tif',parameters=params,bands=['QA_PIXEL'],source='mock')


def test_source_grid_override_is_rejected(monkeypatch,tmp_path):
    monkeypatch.setattr(ee_export,'native_projection',lambda image,bands:{'crs':'EPSG:32755','transform':[30,0,500000,0,-30,6000000],'bands':bands})
    with pytest.raises(ValueError,match='differs from native'):
        ee_export.download_geotiff(Image(),tmp_path/'bad.tif',parameters={'region':[], 'crs':'EPSG:32755','crs_transform':[30,0,500015,0,-30,6000000]},bands=['QA_PIXEL'],source='mock')
