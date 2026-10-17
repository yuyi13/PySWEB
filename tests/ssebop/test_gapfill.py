#!/usr/bin/env python3
"""
Script: test_gapfill.py
Objective: Verify adaptive matching, missing-data restrictions and fill provenance.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Deterministic synthetic reflectance and temperature fixtures.
Outputs: Pytest assertions.
Usage: pytest tests/ssebop/test_gapfill.py
Dependencies: numpy, pytest, PySWEB.
"""
import numpy as np
import pytest
from pysweb.ssebop.gapfill import local_linear_gapfill


@pytest.mark.parametrize('offset,scale', [(0.05, 1.3), (280, 1.1)])
def test_affine_stripes_preserve_primary_and_do_not_fill_clouds(offset, scale):
    y, x = np.indices((51, 61))
    donor = (x + y * 0.2) / 100
    expected = donor * scale + offset
    primary = expected.copy()
    gaps = (x % 11) < 3
    primary[gaps] = np.nan
    primary[23:26, 23:26] = np.nan  # cloud outside eligible stripes
    result = local_linear_gapfill(primary, [donor], gaps)
    filled = result['provenance'] == 2
    assert filled.sum() > 500
    np.testing.assert_allclose(result['values'][filled], expected[filled], atol=1e-10)
    np.testing.assert_array_equal(result['values'][np.isfinite(primary)], primary[np.isfinite(primary)])
    assert np.isnan(result['values'][24, 25])
    assert np.all(result['common_count'][filled] >= 144)
    assert np.all(result['window'][filled] <= 31)
    assert np.all(result['method'][filled] == 1)


def test_constant_band_and_ordered_donor_provenance():
    primary = np.full((41, 41), 300.)
    gaps = np.zeros(primary.shape, bool)
    gaps[20, 19:22] = True
    primary[gaps] = np.nan
    first = np.full(primary.shape, 290.)
    first[20, 20:] = np.nan
    second = np.full(primary.shape, 310.)
    result = local_linear_gapfill(primary, [first, second], gaps)
    np.testing.assert_allclose(result['values'], 300.)
    np.testing.assert_array_equal(result['provenance'][20, 19:22], [2, 3, 3])
    assert np.all(result['method'][gaps] == 3)


def test_insufficient_support_and_rejected_physical_values_stay_missing():
    primary = np.full((5, 5), np.nan)
    primary[0, 0] = 300
    result = local_linear_gapfill(primary, [np.ones((5, 5))], np.ones((5, 5), bool))
    assert np.isfinite(result['values']).sum() == 1
    primary = np.full((41, 41), 400.)
    primary[20, 20] = np.nan
    result = local_linear_gapfill(primary, [np.full((41, 41), 400.)], np.isnan(primary), value_limits=(200, 370))
    assert np.isnan(result['values'][20, 20])


def test_excessive_gain_uses_mean_offset():
    yy, xx = np.indices((41, 41))
    donor = xx.astype(float)
    primary = donor * 10 + 3
    primary[20, 20] = np.nan
    result = local_linear_gapfill(primary, [donor], np.isnan(primary))
    assert result['method'][20, 20] == 3
    assert result['values'][20, 20] == pytest.approx(203.)


@pytest.mark.parametrize('kwargs', [dict(max_window=30), dict(min_common=1), dict(gain_limits=(-1, 3))])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        local_linear_gapfill(np.ones((3, 3)), [], np.zeros((3, 3), bool), **kwargs)
