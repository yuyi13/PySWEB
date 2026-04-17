#!/usr/bin/env python3
"""
Script: test_core.py
Objective: Verify extracted SSEBop package helpers preserve expected ET fraction, climatology, landcover, grid, and compatibility-shim behavior.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: In-memory xarray arrays, pandas timestamps, and temporary NetCDF files.
Outputs: Test assertions.
Usage: python -m pytest tests/ssebop/test_core.py -q
Dependencies: numpy, pandas, pytest, rioxarray, xarray
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from affine import Affine
import rioxarray  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = PROJECT_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from core.ssebop_au import open_silo_et_short_crop, open_silo_variable
from pysweb.ssebop.core import (
    build_doy_climatology,
    compute_dt_daily,
    daily_et_from_etf,
    et_fraction_xr,
)
from pysweb.ssebop.grid import reproject_match
from pysweb.ssebop.landcover import worldcover_masks


def test_et_fraction_xr_clamps_and_masks():
    lst = xr.DataArray(np.array([[305.0]], dtype=float), dims=("y", "x"))
    tcold = xr.DataArray(np.array([[300.0]], dtype=float), dims=("y", "x"))
    dt = xr.DataArray(np.array([[4.0]], dtype=float), dims=("y", "x"))

    result = et_fraction_xr(lst, tcold, dt, clamp_max=1.0, mask_max=2.0)

    np.testing.assert_allclose(result.values, [[0.0]])


def test_build_doy_climatology_groups_by_dayofyear():
    data = xr.DataArray(
        [1.0, 3.0],
        coords={"time": pd.to_datetime(["2024-01-01", "2025-01-01"])},
        dims=("time",),
    )

    result = build_doy_climatology(data)

    assert int(result["dayofyear"].values[0]) == 1
    np.testing.assert_allclose(result.values, [2.0])


def test_worldcover_masks_preserve_expected_classes():
    landcover = xr.DataArray(np.array([[30, 50, 80]], dtype=np.uint8), dims=("y", "x"))
    ag_mask, anomalous_mask, water_mask = worldcover_masks(landcover)

    np.testing.assert_array_equal(ag_mask.values, [[1, 0, 0]])
    np.testing.assert_array_equal(anomalous_mask.values, [[0, 1, 0]])
    np.testing.assert_array_equal(water_mask.values, [[0, 0, 1]])


def test_daily_et_from_etf_interpolates_and_multiplies_reference_et():
    etf_series = pd.Series(
        [0.5, 0.9],
        index = pd.to_datetime(["2024-01-01", "2024-01-03"]),
        name = "etf",
    )
    et_reference_series = pd.Series(
        [4.0, 4.0, 4.0],
        index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        name = "eto",
    )

    result = daily_et_from_etf(etf_series, et_reference_series)

    expected = pd.Series(
        [2.0, 2.8, 3.6],
        index = et_reference_series.index,
        name = "et_ssebop",
    )
    pd.testing.assert_series_equal(result, expected)


def test_compute_dt_daily_uses_time_dayofyear_coordinate():
    time = pd.to_datetime(["2024-01-01", "2024-01-02"])
    tmax = xr.DataArray(
        np.array([300.0, 301.0], dtype=float),
        coords = {"time": time},
        dims = ("time",),
    )
    tmin = xr.DataArray(
        np.array([290.0, 291.0], dtype=float),
        coords = {"time": time},
        dims = ("time",),
    )
    elev = xr.DataArray(100.0)
    lat = xr.DataArray(-35.0)
    rs = xr.DataArray(
        np.array([20.0, 21.0], dtype=float),
        coords = {"time": time},
        dims = ("time",),
    )
    ea = xr.DataArray(
        np.array([1.2, 1.1], dtype=float),
        coords = {"time": time},
        dims = ("time",),
    )

    result = compute_dt_daily(tmax, tmin, elev, lat, rs_mj_m2_day=rs, ea_kpa=ea)

    assert result.dims == ("time",)
    assert result.sizes["time"] == 2
    assert np.isfinite(result.values).all()
    assert not np.isclose(result.values[0], result.values[1])


def test_legacy_silo_helpers_open_requested_variable(tmp_path):
    data = xr.Dataset(
        {
            "et_short_crop": (("lat", "lon"), np.array([[1.5]], dtype=np.float32)),
            "max_temp": (("lat", "lon"), np.array([[32.0]], dtype=np.float32)),
        },
        coords = {"lat": np.array([-35.0]), "lon": np.array([149.0])},
    )
    path = tmp_path / "silo.nc"
    data.to_netcdf(path)

    eto = open_silo_et_short_crop(str(path))
    tmax = open_silo_variable(str(path), "max_temp")

    assert eto.dims == ("y", "x")
    assert tmax.dims == ("y", "x")
    np.testing.assert_allclose(eto.values, [[1.5]])
    np.testing.assert_allclose(tmax.values, [[32.0]])


def test_legacy_silo_helpers_raise_for_missing_variable(tmp_path):
    data = xr.Dataset(
        {"rain": (("lat", "lon"), np.array([[1.0]], dtype=np.float32))},
        coords = {"lat": np.array([-35.0]), "lon": np.array([149.0])},
    )
    path = tmp_path / "silo_missing.nc"
    data.to_netcdf(path)

    with pytest.raises(ValueError, match="Variable 'et_short_crop' not found"):
        open_silo_et_short_crop(str(path))

    with pytest.raises(ValueError, match="Variable 'max_temp' not found"):
        open_silo_variable(str(path), "max_temp")


def test_reproject_match_identity_smoke():
    x = np.array([0.5, 1.5], dtype=float)
    y = np.array([1.5, 0.5], dtype=float)
    data = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        coords = {"y": y, "x": x},
        dims = ("y", "x"),
        name = "sample",
    )
    data = data.rio.write_crs("EPSG:4326")
    data = data.rio.write_transform(Affine.translation(0.0, 2.0) * Affine.scale(1.0, -1.0))

    result = reproject_match(data, data, resampling="nearest")

    np.testing.assert_allclose(result.values, data.values)
    assert result.rio.crs == data.rio.crs
