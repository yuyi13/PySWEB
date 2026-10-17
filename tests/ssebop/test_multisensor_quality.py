#!/usr/bin/env python3
"""
Script: test_multisensor_quality.py
Objective: Check sensor-specific band mappings, unused cirrus bits and Landsat 7 dropped-pixel handling.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Independently specified Collection 2 bit-field fixtures.
Outputs: pytest assertions.
Usage: pytest tests/ssebop/test_multisensor_quality.py
Dependencies: numpy, pytest, pysweb.
"""
import pytest
from pysweb.ssebop.quality import landsat_c2_qa_mask, landsat_c2_l2_bands


def test_sensor_bands_and_unused_bit():
    for sensor in ('LT05','LE07'):
        assert landsat_c2_l2_bands(sensor)==('ST_B6','SR_B3','SR_B4')
        assert landsat_c2_qa_mask([64,68,72,192],[0,0,0,0],sensor=sensor).tolist()==[True,True,False,False]
    for sensor in ('LC08','LC09'):
        assert landsat_c2_l2_bands(sensor)==('ST_B10','SR_B4','SR_B5')
        assert landsat_c2_qa_mask([64,68],[0,0],sensor=sensor).tolist()==[True,False]
    assert not landsat_c2_qa_mask([64],[512],sensor='LE07').item()
    with pytest.raises(ValueError,match='Unsupported'):
        landsat_c2_qa_mask([64],[0],sensor='unknown')


@pytest.mark.parametrize("sensor", ["LT05", "LE07", "LC08", "LC09"])
def test_scene_auto_bands_preserve_the_same_physical_signal(sensor, tmp_path, monkeypatch):
    import numpy as np
    import rasterio
    import xarray as xr
    from rasterio.transform import from_origin
    from pysweb.ssebop import api
    path = tmp_path / f"{sensor}_2017-01-01.tif"
    names = list(landsat_c2_l2_bands(sensor)) + ["QA_PIXEL", "QA_RADSAT"]
    values = np.stack([np.full((2,2), v, dtype="uint16") for v in [45000,10000,12000,64,0]])
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=5,
                       dtype="uint16", crs="EPSG:32755", transform=from_origin(430000,6128000,30,30)) as dst:
        dst.write(values); dst.descriptions = names
    template = api.read_geotiff_bands(str(path))[names[0]]
    dt = xr.full_like(template, 8.).expand_dims(dayofyear=[1])
    monkeypatch.setattr(api, "reproject_match", lambda source, match, **kw: source)
    monkeypatch.setattr(api, "tcold_fano_simple_xr", lambda lst, ndvi, dt, **kw: lst - 4.)
    _, etf, ndvi = api.process_landsat_scene(str(path), "auto", "ndvi", "auto", "auto", False, None, dt,
                                           template.rio.crs, str(tmp_path), str(tmp_path))
    with rasterio.open(etf) as src: np.testing.assert_allclose(src.read(1), .5)
    with rasterio.open(ndvi) as src: np.testing.assert_allclose(src.read(1), .055/.205, rtol=1e-6)


def test_reference_type_mismatch_requires_explicit_conversion():
    import numpy as np
    from pysweb.ssebop.core import et_from_fraction
    with pytest.raises(ValueError, match="differ"):
        et_from_fraction(np.array([.5]), np.array([4.]), fraction_type="alfalfa", reference_type="grass")
    result = et_from_fraction(np.array([.5,np.nan]), np.array([6.,6.]), fraction_type="alfalfa", reference_type="alfalfa")
    np.testing.assert_equal(result, [3.,np.nan])
