"""
Script: test_extraction_thresholds.py
Objective: Verify depth-dependent extraction, continuous relaxation and conserved water independently.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-21
Last updated: 2026-10-21
Inputs: Analytical drydowns and synthetic layered columns with known hydraulic properties.
Outputs: Assertions for threshold transfer, extraction, physical bounds and both numerical backends.
Usage: python -m pytest tests/swb/test_extraction_thresholds.py
Dependencies: numpy, pandas, pytest; pysweb and optional numba.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(params=["reference", "numba"])
def solve(request):
    if request.param == "numba":
        pytest.importorskip("numba")
        from pysweb.swb.accelerated import soil_water_balance_1d
    else:
        from pysweb.swb.solver import soil_water_balance_1d
    return soil_water_balance_1d


def properties():
    return dict(layer_depth=np.array([50., 150., 300., 600., 1000.]),
        layer_thickness=np.array([50., 100., 150., 300., 400.]),
        porosity=np.full(5, .5), wilting_point=np.array([.15, .18, .20, .25, .30]),
        available_water_capacity=np.full(5, .05), b_coefficient=np.ones(5),
        conductivity_sat=np.zeros(5), root_beta=.96, dry_reference_factor=.4)


def test_all_layers_follow_analytical_three_day_extraction_relaxation(solve):
    p = properties()
    initial = np.full(5, .35)
    thresholds = np.array([.06, .072, .14, .25, .30])
    result = solve(np.zeros(60), np.full(60, 1e5), p,
        pd.date_range("2020-01-01", periods=60), transpiration_data=np.full(60, 1e5),
        initial_soil_moisture=initial, state_timing="end")
    expected = thresholds + (initial-thresholds)*np.exp(-np.arange(1, 61)[:, None]/3.)
    np.testing.assert_allclose(result, expected, rtol=0, atol=3e-16)
    np.testing.assert_allclose(result.attrs["extraction_threshold"], thresholds, rtol=0, atol=1e-16)
    np.testing.assert_array_equal(result.attrs["hydraulic_wilting_point"], p["wilting_point"])
    assert np.all(result.iloc[-1, :3] < p["wilting_point"][:3])
    assert np.all(result.values >= thresholds-1e-16)
    b = result.attrs["water_balance"]
    assert not np.any(b["storage_adjustment"]) and not np.any(b["actual_evaporation"])
    changes = np.diff(np.vstack([initial, result.values]), axis=0)*p["layer_thickness"]
    uptake = np.column_stack([b[f"transpiration_layer_{i}"] for i in range(1, 6)])
    np.testing.assert_allclose(changes, -uptake, atol=3e-14, rtol=0)
    np.testing.assert_allclose(changes.sum(axis=1), -b["actual_et"], atol=5e-14, rtol=0)


def test_dry_initial_state_has_no_extraction_or_restoration(solve):
    initial = np.array([.03, .036, .07, .125, .15])
    result = solve([0.], [1000.], properties(), ["2020-01-01"],
        transpiration_data=[1000.], initial_soil_moisture=initial, state_timing="end")
    np.testing.assert_allclose(result.iloc[0], initial, rtol=0, atol=3e-17)
    b = result.attrs["water_balance"]
    assert b["actual_et"][0] == b["storage_adjustment"][0] == 0.
    for layer in range(1, 6):
        assert b[f"below_reference_days_layer_{layer}"][0] == 1.
        assert b[f"below_wp_days_layer_{layer}"][0] == 1.


def test_thresholds_do_not_block_redistribution_below_them(solve):
    p = properties()
    for key in ["layer_depth", "layer_thickness", "porosity", "wilting_point",
                "available_water_capacity", "b_coefficient", "conductivity_sat"]:
        p[key] = p[key][:2]
    p.update(conductivity_sat=np.full(2, 100.), drainage_slope=0.)
    initial = np.array([.05, .06])  # Both states are below their extraction thresholds.
    result = solve([0.], [1000.], p, ["2020-01-01"], diff_factor=1000.,
        transpiration_data=[1000.], initial_soil_moisture=initial, state_timing="end")
    b = result.attrs["water_balance"]
    assert b["interface_flux_1"][0] < 0.
    assert result.iloc[0, 1] < initial[1]
    assert b["actual_transpiration"][0] == 0.
    assert abs(np.dot(result.iloc[0]-initial, p["layer_thickness"])) < 1e-14


def test_unadjusted_reference_retains_wp_extraction_thresholds(solve):
    p = properties()
    p.pop("dry_reference_factor")
    result = solve([0.], [1000.], p, ["2020-01-01"],
        transpiration_data=[1000.], initial_soil_moisture=p["wilting_point"], state_timing="end")
    np.testing.assert_array_equal(result.attrs["extraction_threshold"], p["wilting_point"])
    assert result.attrs["water_balance"]["actual_transpiration"][0] == 0.


def test_depth_rule_is_preserved_when_layers_cross_reference_bands(solve):
    p = properties()
    p.update(layer_depth=np.array([100., 200., 400., 1000.]),
        layer_thickness=np.array([100., 100., 200., 600.]),
        wilting_point=np.full(4, .2))
    for key in ["porosity", "available_water_capacity", "b_coefficient", "conductivity_sat"]:
        p[key] = p[key][:4]
    result = solve([0.], [0.], p, ["2020-01-01"])
    np.testing.assert_allclose(result.attrs["extraction_threshold"], [.08, .11, .17, .2], atol=1e-16)


def test_zero_dry_reference_retains_the_third_layer_midpoint(solve):
    p = properties()
    p["dry_reference_factor"] = 0.
    result = solve([0.], [0.], p, ["2020-01-01"])
    np.testing.assert_array_equal(result.attrs["extraction_threshold"], [0., 0., .1, .25, .3])


def test_repeated_and_disabled_initialization_preserve_revised_physics(solve):
    """Combine the two branches against explicitly expanded daily forcing."""
    p = properties()
    p['conductivity_sat'] = np.full(5, 10.)
    dates = pd.date_range('2014-01-01', '2016-12-31')
    t = np.arange(len(dates))
    rain = np.where(t % 11 == 0, 20., 0.)
    et = 3. + np.sin(t / 50.)
    tr = .8 * et
    settings = dict(enabled=True, source_years=[2014, 2015], repeats=10)
    compact = solve(rain, et, p, dates, transpiration_data=tr, spinup=settings)
    expanded = [np.r_[np.tile(v[:730], 9), v] for v in (rain, et, tr)]
    explicit = solve(expanded[0], expanded[1], p,
        pd.date_range(end=dates[-1], periods=len(expanded[0])), transpiration_data=expanded[2])
    np.testing.assert_allclose(compact, explicit.iloc[6570:], atol=1e-12, rtol=0)
    np.testing.assert_allclose(compact.attrs['extraction_threshold'], [.06, .072, .14, .25, .30], atol=1e-16)
    ordinary = solve(rain, et, p, dates, transpiration_data=tr)
    disabled = solve(rain, et, p, dates, transpiration_data=tr, spinup={**settings, 'enabled': False})
    np.testing.assert_array_equal(ordinary, disabled)
    for key, values in compact.attrs['water_balance'].items():
        np.testing.assert_allclose(values, explicit.attrs['water_balance'][key][6570:], atol=1e-10, rtol=0)
        np.testing.assert_array_equal(ordinary.attrs['water_balance'][key], disabled.attrs['water_balance'][key])
