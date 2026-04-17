#!/usr/bin/env python3
"""
Script: test_api_run.py
Objective: Verify the SSEBop package run API validates required inputs before dispatching workflow execution.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: Package API calls, temporary files, and monkeypatched package functions supplied by pytest.
Outputs: Test assertions.
Usage: pytest tests/ssebop/test_api_run.py
Dependencies: numpy, pytest, xarray
"""
from pathlib import Path
import sys

import numpy as np
import pytest
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
