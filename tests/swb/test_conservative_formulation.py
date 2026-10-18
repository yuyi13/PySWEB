"""
Script: test_conservative_formulation.py
Objective: Independently verify physical storage, analytical limits, conservative budgets and migration.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-18
Last updated: 2026-10-18
Inputs: Synthetic columns, extreme forcing and fitted float64 coefficient fixtures.
Outputs: Numerical assertions for the revised SWB contract.
Usage: python -m pytest tests/swb/test_conservative_formulation.py
Dependencies: numpy, pandas, pytest; pysweb
"""
import numpy as np
import pandas as pd
import pytest
from pysweb.swb.solver import soil_water_balance_1d, _compute_root_distribution_beta
from pysweb.swb.formulation import load_calibration, settings_identity


def props(n=5, **overrides):
    bottom = np.array([50., 150., 300., 600., 1000.])[:n]
    result = dict(layer_depth=bottom, layer_thickness=np.diff(np.r_[0., bottom]),
        porosity=np.full(n, .45), wilting_point=np.full(n, .15),
        available_water_capacity=np.full(n, .15), b_coefficient=np.full(n, 4.),
        conductivity_sat=np.zeros(n), root_beta=.96, dry_reference_factor=.4)
    result.update(overrides)
    return result


@pytest.mark.parametrize("backend", ["reference", "numba"])
def test_drydown_zero_reference_has_exact_exponential_limit(backend):
    solver = soil_water_balance_1d
    if backend == "numba":
        pytest.importorskip("numba")
        from pysweb.swb.accelerated import soil_water_balance_1d as solver
    # With d=0, no hydraulics and ample demand, storage decays exactly as exp(-t/tau).
    output = solver(np.zeros(15), np.full(15, 100.), props(1, dry_reference_factor=0.),
        pd.date_range("2020-01-01", periods=15), transpiration_data=np.zeros(15),
        initial_soil_moisture=[.12], state_timing="end")
    np.testing.assert_allclose(output.values[:, 0], .12*np.exp(-np.arange(1, 16)/3.), atol=2e-15)
    assert output.attrs["water_balance"]["storage_adjustment"].sum() == 0


def test_evaporation_crosses_wp_and_reference_without_adding_water():
    p = props(1)
    result = soil_water_balance_1d(np.zeros(120), np.full(120, 20.), p,
        pd.date_range("2020-01-01", periods=120), transpiration_data=np.zeros(120),
        initial_soil_moisture=[.16], state_timing="end")
    trace = result.values[:, 0]
    assert np.all(np.diff(trace) < 0) and 0 < trace[-1] < .01
    assert trace[0] < .15 and np.any(trace < .06)
    assert np.all(np.diff(result.attrs["water_balance"]["actual_evaporation"]) < 0)
    for initial in [0., .02, .06, .14]:
        no_sink = soil_water_balance_1d([0.], [100.], p, ["2020-01-01"],
            transpiration_data=[100.], initial_soil_moisture=[initial])
        np.testing.assert_array_equal(no_sink.attrs["final_state"], [initial])


def test_physical_fc_initialisation_and_partial_root_geometry():
    p = props(max_root_depth=230.)
    result = soil_water_balance_1d([0.], [0.], p, ["2020-01-01"])
    np.testing.assert_array_equal(result.attrs["initial_state"], np.full(5, .3))
    roots = _compute_root_distribution_beta(p)
    expected = np.diff(1-.96**np.array([0., 5., 15., 23., 23., 23.]))/(1-.96**23.)
    np.testing.assert_allclose(roots, expected, atol=1e-15)
    assert roots[-1] == roots[-2] == 0


@pytest.mark.parametrize("backend", ["reference", "numba"])
def test_daily_transfer_matches_one_independent_balance(backend):
    solver = soil_water_balance_1d
    if backend == "numba":
        pytest.importorskip("numba")
        from pysweb.swb.accelerated import soil_water_balance_1d as solver
    p = props(2, porosity=np.full(2, .5), b_coefficient=np.ones(2),
              conductivity_sat=np.ones(2), drainage_slope=0.)
    result = solver([0.], [0.], p, ["2020-01-01"],
                    initial_soil_moisture=[.25, .1], diff_factor=10., state_timing="end")
    # S=1/2, K=(1/2)^5, D=10*(1/2)^3; centres are 75 mm apart.
    transfer_mm = .03125 + 1.25 * .15 / 75.
    np.testing.assert_allclose(result.values[0],
        [.25-transfer_mm/50., .1+transfer_mm/100.], rtol=0, atol=1e-15)
    assert result.attrs["water_balance"]["interface_flux_1"][0] == pytest.approx(transfer_mm)
    np.testing.assert_array_equal(result.attrs["water_balance"]["substeps"], [1.])
    assert result.attrs["time_step_days"] == 1.


def test_subcycling_settings_and_prior_calibrations_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="max_substep_days"):
        soil_water_balance_1d([0.], [0.], props(max_substep_days=1/768), ["2020-01-01"])
    path = tmp_path / "subcycled.csv"
    identity = settings_identity()
    identity["formulation_version"] = "swb-conservative-2p-v1"
    pd.DataFrame([dict(diff_factor=1000., root_beta=.96, **identity)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="formulation_version"):
        load_calibration(path)


@pytest.mark.parametrize("initial", [0., .01, .149, .449, .45])
@pytest.mark.parametrize("rain,et", [(0., 0.), (0., 10000.), (1000., 10000.), (1000., 0.)])
def test_extremes_joint_donors_and_independent_layer_profile_balances(initial, rain, et):
    p = props(conductivity_sat=np.full(5, 300.))
    result = soil_water_balance_1d([rain]*2, [et]*2, p,
        pd.date_range("2020-01-01", periods=2), initial_soil_moisture=np.full(5, initial),
        transpiration_data=[et*.7]*2, state_timing="end", diff_factor=10000.)
    b = result.attrs["water_balance"]
    endpoints = np.vstack([result.attrs["initial_state"], result.values])
    changes = np.diff(endpoints, axis=0)*p["layer_thickness"]
    flux = np.column_stack([b["infiltration"], *[b[f"interface_flux_{i}"] for i in range(1, 5)], b["drainage"]+b["overflow"]])
    sinks = np.column_stack([b[f"transpiration_layer_{i}"] for i in range(1, 6)])
    sinks[:, 0] += b["actual_evaporation"]
    np.testing.assert_allclose(changes, flux[:, :-1]-flux[:, 1:]-sinks, atol=1e-9)
    np.testing.assert_allclose(changes.sum(axis=1), b["infiltration"]-b["actual_et"]-b["drainage"]-b["overflow"], atol=1e-9)
    for component in ("evaporation", "transpiration"):
        np.testing.assert_allclose(b[f"requested_{component}"], b[f"actual_{component}"]+b[f"unmet_{component}"], atol=1e-9)
        assert np.min(b[f"unmet_{component}"]) >= -1e-9
    assert np.min(result.values) >= 0 and np.max(result.values) <= .45
    assert not np.any(b["storage_adjustment"])


def test_upward_downward_and_no_inflow_conservation():
    p = props(2, conductivity_sat=np.full(2, 50.), drainage_slope=0.)
    for initial, sign in [([.1, .4], -1), ([.4, .1], 1)]:
        result = soil_water_balance_1d([0.], [0.], p, ["2020-01-01"],
            initial_soil_moisture=initial, diff_factor=10000.)
        b = result.attrs["water_balance"]
        assert b["interface_flux_1"][0]*sign > 0
        assert abs(np.dot(result.attrs["final_state"]-initial, p["layer_thickness"])) < 1e-10


def test_wet_reference_is_diagnostic_and_legacy_inputs_fail(tmp_path):
    baseline = soil_water_balance_1d([50.], [2.], props(), ["2020-01-01"])
    altered = soil_water_balance_1d([50.], [2.], props(wet_reference_factor=4.), ["2020-01-01"])
    np.testing.assert_array_equal(baseline.attrs["final_state"], altered.attrs["final_state"])
    with pytest.raises(ValueError, match="legacy"):
        soil_water_balance_1d([0.], [0.], props(sm_min_factor=1.), ["2020-01-01"])
    path = tmp_path/"legacy.csv"
    pd.DataFrame([dict(diff_factor=1000, root_beta=.96, sm_min_factor=.75)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="legacy"):
        load_calibration(path)
    pd.DataFrame([dict(diff_factor=1000, root_beta=.96)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Incompatible"):
        load_calibration(path)
    pd.DataFrame([dict(diff_factor=1000, root_beta=.96, **settings_identity())]).to_csv(path, index=False)
    assert load_calibration(path) == dict(diff_factor=1000., root_beta=.96)


@pytest.mark.parametrize("diff_factor,root_beta", [
    (9609.49560311022, 0.9352476318927986),
    (108.24431434733798, 0.9006237327166058),
])
def test_calibration_csv_preserves_exact_optimized_coefficients(tmp_path, diff_factor, root_beta):
    # These fitted values exposed a few-ulp shift with pandas' default parser.
    # Exact restoration matters when a rejected polish must retain the DE fit.
    expected = dict(diff_factor=diff_factor, root_beta=root_beta)
    path = tmp_path/"calibration.csv"
    pd.DataFrame([{**expected, **settings_identity()}]).to_csv(
        path, index=False, float_format="%.17g")
    restored = load_calibration(path)
    assert {key:value.hex() for key,value in restored.items()} == {
        key:value.hex() for key,value in expected.items()}
