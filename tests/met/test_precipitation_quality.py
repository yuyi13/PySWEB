#!/usr/bin/env python3
"""
Script: test_precipitation_quality.py
Objective: Verify bounded precipitation-noise correction cannot create rain from missing or substantially negative inputs.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Signed, missing and nonfinite synthetic precipitation values in millimetres.
Outputs: Quality/immutability and explicit-tolerance assertions.
Usage: python -m pytest tests/met/test_precipitation_quality.py
Dependencies: numpy, pytest, PySWEB.
"""
import numpy as np
import pytest
from pysweb.met.era5land.quality import qc_precipitation_mm


def test_signed_noise_missingness_and_input_immutability():
    source=np.array([[0., .00001, 12., -.00003],[-.001, -.001001, np.nan, np.inf]])
    original=source.copy()
    values,flags=qc_precipitation_mm(source,small_negative_tolerance_mm=.001)
    np.testing.assert_allclose(values,[[0.,.00001,12.,0.],[0.,np.nan,np.nan,np.nan]],equal_nan=True)
    np.testing.assert_array_equal(flags,[[0,0,0,1],[1,2,3,3]])
    np.testing.assert_equal(source,original)


def test_zero_tolerance_retains_no_negative_values():
    values,flags=qc_precipitation_mm([-.00001,0.,.00001],small_negative_tolerance_mm=0)
    assert np.isnan(values[0]) and flags[0]==2
    np.testing.assert_equal(values[1:],[0.,.00001])


@pytest.mark.parametrize('tolerance',[-1.,np.nan,np.inf])
def test_invalid_tolerance_refused(tolerance):
    with pytest.raises(ValueError,match='finite and nonnegative'):
        qc_precipitation_mm([0.],small_negative_tolerance_mm=tolerance)
