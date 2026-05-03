#!/usr/bin/env python3
"""
Script: test_api_run.py
Objective: Verify the SSEBop package run API validates incomplete calls while forwarding supported workflow inputs.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-05-01
Inputs: Package API calls, temporary files, and monkeypatched package functions supplied by pytest.
Outputs: Test assertions.
Usage: pytest tests/ssebop/test_api_run.py
Dependencies: numpy, pytest, rioxarray, xarray
"""
from pathlib import Path
import sys

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pysweb.ssebop.api as ssebop_api

run = ssebop_api.run


def test_ssebop_run_requires_explicit_inputs():
    with pytest.raises(ValueError):
        run(
            date_range="2024-01-01 to 2024-01-03",
            landsat_dir="",
            met_dir="",
            dem="",
            output_dir="",
        )


def test_ssebop_run_rejects_incidental_kwargs_without_entry_inputs(monkeypatch):
    dispatched = False

    def fake_run_ssebop_workflow(**kwargs):
        nonlocal dispatched
        dispatched = True

    monkeypatch.setattr(ssebop_api, "run_ssebop_workflow", fake_run_ssebop_workflow)

    with pytest.raises(ValueError):
        run(workers=2)

    assert dispatched is False


def test_ssebop_run_dispatches_to_package_workflow(monkeypatch):
    recorded = {}

    def fake_run_ssebop_workflow(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(ssebop_api, "run_ssebop_workflow", fake_run_ssebop_workflow)

    run(
        date_range="2024-01-01 to 2024-01-03",
        landsat_dir="/tmp/landsat",
        met_dir="/tmp/met",
        dem="/tmp/dem.tif",
        output_dir="/tmp/out",
        workers=2,
    )

    assert recorded == {
        "date_range": "2024-01-01 to 2024-01-03",
        "landsat_dir": "/tmp/landsat",
        "met_dir": "/tmp/met",
        "dem": "/tmp/dem.tif",
        "output_dir": "/tmp/out",
        "workers": 2,
    }


def test_ssebop_run_allows_config_driven_invocation(monkeypatch):
    recorded = {}

    def fake_run_ssebop_workflow(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(ssebop_api, "run_ssebop_workflow", fake_run_ssebop_workflow)

    run(config="/tmp/ssebop.yaml")

    assert recorded == {"config": "/tmp/ssebop.yaml"}


def test_ssebop_run_forwards_explicit_direct_kwargs_intact(monkeypatch):
    recorded = {}

    def fake_run_ssebop_workflow(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(ssebop_api, "run_ssebop_workflow", fake_run_ssebop_workflow)

    run(
        date_range="2024-01-01 to 2024-01-03",
        landsat_dir="/tmp/landsat",
        dem="/tmp/dem.tif",
        output_dir="/tmp/out",
        et_short_crop="/tmp/eto.nc",
        tmax="/tmp/tmax.nc",
        tmin="/tmp/tmin.nc",
        rs="/tmp/rs.nc",
        ea="/tmp/ea.nc",
        workers=2,
    )

    assert recorded == {
        "date_range": "2024-01-01 to 2024-01-03",
        "landsat_dir": "/tmp/landsat",
        "dem": "/tmp/dem.tif",
        "output_dir": "/tmp/out",
        "et_short_crop": "/tmp/eto.nc",
        "tmax": "/tmp/tmax.nc",
        "tmin": "/tmp/tmin.nc",
        "rs": "/tmp/rs.nc",
        "ea": "/tmp/ea.nc",
        "workers": 2,
    }


def test_ssebop_run_forwards_tcold_fano_kwargs_intact(monkeypatch):
    recorded = {}

    def fake_run_ssebop_workflow(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(ssebop_api, "run_ssebop_workflow", fake_run_ssebop_workflow)

    run(
        date_range="2024-01-01 to 2024-01-03",
        landsat_dir="/tmp/landsat",
        dem="/tmp/dem.tif",
        output_dir="/tmp/out",
        et_short_crop="/tmp/eto.nc",
        tmax="/tmp/tmax.nc",
        tmin="/tmp/tmin.nc",
        rs="/tmp/rs.nc",
        ea="/tmp/ea.nc",
        tcold_dt_coeff=0.15,
        tcold_high_ndvi_threshold=0.85,
        tcold_anchor_ndvi_threshold=0.35,
        tcold_fine_scale_m=240.0,
        tcold_coarse_scale_m=4800.0,
        tcold_smooth_scale_m=240.0,
        workers=2,
    )

    assert recorded["tcold_dt_coeff"] == 0.15
    assert recorded["tcold_high_ndvi_threshold"] == 0.85
    assert recorded["tcold_anchor_ndvi_threshold"] == 0.35
    assert recorded["tcold_fine_scale_m"] == 240.0
    assert recorded["tcold_coarse_scale_m"] == 4800.0
    assert recorded["tcold_smooth_scale_m"] == 240.0


def test_open_meteorology_da_prefers_field_defaults_for_custom_files(tmp_path: Path):
    custom_path = tmp_path / "custom_tmax.nc"
    custom_ds = xr.Dataset(
        {"tmax": (("y", "x"), np.array([[1.0]], dtype=np.float32))},
        coords={"x": np.array([0.5]), "y": np.array([0.5])},
    )
    custom_ds.to_netcdf(custom_path)

    silo_path = tmp_path / "2024.max_temp.nc"
    silo_ds = xr.Dataset(
        {"max_temp": (("y", "x"), np.array([[2.0]], dtype=np.float32))},
        coords={"x": np.array([0.5]), "y": np.array([0.5])},
    )
    silo_ds.to_netcdf(silo_path)

    custom_da = ssebop_api.open_meteorology_da(str(custom_path), None, default_var="tmax")
    silo_da = ssebop_api.open_meteorology_da(str(silo_path), None, default_var="tmax")

    assert custom_da.name == "tmax"
    assert silo_da.name == "max_temp"


def test_process_landsat_scene_uses_nonprojected_tcold_fallback(monkeypatch, tmp_path: Path):
    coords = {"y": np.array([1.0, 0.0]), "x": np.array([0.0, 1.0])}

    def raster(name: str, value: float) -> xr.DataArray:
        data = xr.DataArray(
            np.full((2, 2), value, dtype=np.float32),
            dims=("y", "x"),
            coords=coords,
            name=name,
        )
        data.attrs["long_name"] = ("ST_B10", "SR_B4", "SR_B5")
        return data.rio.write_crs("EPSG:4326")

    bands = {
        "ST_B10": raster("ST_B10", 45000.0),
        "SR_B4": raster("SR_B4", 10000.0),
        "SR_B5": raster("SR_B5", 12000.0),
    }
    dt_clim = raster("dt", 8.0).expand_dims(dayofyear=[4])
    calls = []

    def fake_simple(lst, ndvi, dt, config=None):
        calls.append((lst.rio.crs.to_string(), config))
        return xr.zeros_like(lst).rename("tcold")

    def fail_strict_local(*args, **kwargs):
        raise AssertionError("strict local FANO should not be called for non-projected rasters")

    monkeypatch.setattr(ssebop_api, "read_geotiff_bands", lambda path: bands)
    monkeypatch.setattr(ssebop_api, "reproject_match", lambda source, match, resampling=None: source)
    monkeypatch.setattr(ssebop_api, "tcold_fano_simple_xr", fake_simple, raising=False)
    monkeypatch.setattr(ssebop_api, "tcold_fano_local_xr", fail_strict_local)

    ssebop_api.process_landsat_scene(
        tif_path = "Landsat_2019-01-04.tif",
        lst_band = "ST_B10",
        ndvi_band = "ndvi",
        red_band = "SR_B4",
        nir_band = "SR_B5",
        apply_water_mask = False,
        water_mask = None,
        dt_clim = dt_clim,
        tcold_config = ssebop_api.LocalFanoConfig(),
        template_crs = bands["ST_B10"].rio.crs,
        etf_dir = str(tmp_path),
        ndvi_dir = str(tmp_path),
    )

    assert calls
