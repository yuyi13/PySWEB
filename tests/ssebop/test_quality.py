#!/usr/bin/env python3
"""
Script: test_quality.py
Objective: Verify Landsat QA retention, cloud rejection and scene processing with tiny GeoTIFFs.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-16
Last updated: 2026-10-16
Inputs: Synthetic Collection 2 flags and temporary raw Landsat rasters.
Outputs: Regression assertions for masks, zero-valued QA and NDVI exports.
Usage: python -m pytest tests/ssebop/test_quality.py
Dependencies: numpy, pytest, rasterio, xarray; pysweb
"""
import json
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import xarray as xr
from pysweb.ssebop.quality import landsat_c2_qa_mask
from pysweb.ssebop import api, landsat


def test_qa_rejects_contaminants_but_keeps_clear_and_low_confidence():
    qa = np.array([0, 64, 256, 512, 1, 2, 4, 8, 16, 32, 128])
    expected = [True, True, True, True, False, False, False, False, False, False, False]
    assert landsat_c2_qa_mask(qa, np.zeros_like(qa)).tolist() == expected
    assert landsat_c2_qa_mask([128], [0], exclude_water=False).item()
    assert not landsat_c2_qa_mask([64], [1 << 4]).item()


def test_noninteger_and_missing_qa_cannot_become_clear_pixels():
    qa = np.array([np.nan, -1, 64.5, 65536, 64, 64])
    sat = np.array([0, 0, 0, 0, np.nan, .5])
    assert not landsat_c2_qa_mask(qa, sat).any()
    with pytest.raises(ValueError, match="identical shapes"):
        landsat_c2_qa_mask([64, 64], [0])


def write_scene(path, include_qa=True):
    shape = (2, 3)
    arrays = [np.full(shape, 45000, dtype="uint16"),
              np.full(shape, 10000, dtype="uint16"),
              np.full(shape, 12000, dtype="uint16")]
    names = ["ST_B10", "SR_B4", "SR_B5"]
    # Clear, cloud, saturated, fill, water, shadow in row-major order.
    arrays[0][1, 0] = 0
    if include_qa:
        arrays += [np.array([[64, 8, 64], [1, 128, 16]], dtype="uint16"),
                   np.array([[0, 0, 16], [0, 0, 0]], dtype="uint16")]
        names += ["QA_PIXEL", "QA_RADSAT"]
    with rasterio.open(path, "w", driver="GTiff", width=3, height=2,
                       count=len(arrays), dtype="uint16", nodata=0,
                       crs="EPSG:32755", transform=from_origin(430000, 6128000, 30, 30)) as dst:
        dst.write(np.stack(arrays))
        dst.descriptions = names


def run_scene(path, monkeypatch, tmp_path, **kwargs):
    bands = api.read_geotiff_bands(str(path))
    template = bands["ST_B10"]
    dt = xr.full_like(template, 8.).expand_dims(dayofyear=[1])
    monkeypatch.setattr(api, "reproject_match", lambda source, match, **kw: source)
    # Isolate QA/output handling from cold-boundary physics in this regression.
    monkeypatch.setattr(api, "tcold_fano_simple_xr", lambda lst, ndvi, dt, **kw: lst)
    return api.process_landsat_scene(
        str(path), "ST_B10", "ndvi", "SR_B4", "SR_B5", False, None, dt,
        template.rio.crs, str(tmp_path), str(tmp_path), **kwargs)


def test_scene_keeps_zero_saturation_and_masks_contaminated_ndvi(tmp_path, monkeypatch):
    path = tmp_path / "Landsat_2017-01-01.tif"
    write_scene(path)
    bands = api.read_geotiff_bands(str(path))
    assert bands["QA_RADSAT"].values[0, 0] == 0
    assert np.isnan(bands["ST_B10"].values[1, 0])
    _, etf_path, ndvi_path = run_scene(path, monkeypatch, tmp_path)
    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1, masked=True).filled(np.nan)
        assert np.isclose(ndvi[0, 0], .055 / .205, rtol=1e-5)
        assert np.isfinite(ndvi).sum() == 1
        assert src.tags()["landsat_qa"] == "required"
    with rasterio.open(etf_path) as src:
        assert np.isfinite(src.read(1, masked=True).filled(np.nan)).sum() == 1


def test_missing_qa_requires_explicit_pre_masked_contract(tmp_path, monkeypatch):
    path = tmp_path / "Landsat_2017-01-01.tif"
    write_scene(path, include_qa=False)
    with pytest.raises(ValueError, match="Missing Landsat QA"):
        run_scene(path, monkeypatch, tmp_path)
    _, _, output = run_scene(path, monkeypatch, tmp_path, landsat_qa="pre_masked")
    with rasterio.open(output) as src:
        assert src.tags()["landsat_qa"] == "pre_masked"
        assert np.isfinite(src.read(1, masked=True).filled(np.nan)).sum() == 5


def test_zero_ndvi_is_retained_as_data(tmp_path, monkeypatch):
    path = tmp_path / "Landsat_2017-01-01.tif"
    write_scene(path)
    with rasterio.open(path, "r+") as dst:
        dst.write(dst.read(2), 3)
    _, _, output = run_scene(path, monkeypatch, tmp_path)
    with rasterio.open(output) as src:
        ndvi = src.read(1, masked=True)
        assert not ndvi.mask[0, 0]
        assert ndvi[0, 0] == 0


def test_default_preparation_retains_qa_without_mutating_defaults(tmp_path, monkeypatch):
    import yaml
    before = json.dumps(landsat._DEFAULT_LANDSAT_GEE_CFG, sort_keys=True)
    configs = []
    class Recorder:
        def __init__(self, path):
            configs.append(yaml.safe_load(Path(path).read_text()))
        def run(self):
            pass
    monkeypatch.setattr(landsat, "GEEDownloader", Recorder)
    landsat.prepare_landsat_inputs(date_range="2017-01-01 to 2017-01-02",
                                  extent=[146.2, -35.1, 146.4, -34.9],
                                  out_dir=str(tmp_path / "landsat"), gee_project="test-project")
    cfg = configs[0]
    assert {"QA_PIXEL", "QA_RADSAT"} <= set(cfg["bands"])
    assert cfg["daily_strategy"] == "first"
    assert cfg["cloud_mask"]["enabled"]
    assert cfg["postprocess"]["maskval_to_na"] is False
    assert json.dumps(landsat._DEFAULT_LANDSAT_GEE_CFG, sort_keys=True) == before
