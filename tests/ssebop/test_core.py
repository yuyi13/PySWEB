#!/usr/bin/env python3
"""
Script: test_core.py
Objective: Verify extracted SSEBop package helpers preserve expected ET fraction, climatology, and landcover behavior.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: In-memory xarray arrays and pandas timestamps.
Outputs: Test assertions.
Usage: python -m pytest tests/ssebop/test_core.py -q
Dependencies: numpy, pandas, pytest, xarray
"""
import numpy as np
import pandas as pd
import xarray as xr

from pysweb.ssebop.core import build_doy_climatology, et_fraction_xr
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
