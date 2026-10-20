"""
Script: test_multiyear_spinup.py
Objective: Verify compact repeated-year initialization against explicitly expanded histories.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-20
Last updated: 2026-10-20
Inputs: Deterministic daily forcing and physical five-layer columns.
Outputs: Regression assertions for states, budgets, calendars and rejected configurations.
Usage: pytest tests/swb/test_multiyear_spinup.py
Dependencies: numpy, pandas, pytest, pysweb; numba for accelerated cases.
"""
import numpy as np
import pandas as pd
import pytest

from pysweb.swb.spinup import source_block
from pysweb.swb.solver import soil_water_balance_1d


def properties():
    bottoms = np.array([50., 150., 300., 600., 1000.])
    return dict(layer_depth=bottoms, layer_thickness=np.diff(np.r_[0., bottoms]),
                porosity=np.full(5, .45), wilting_point=np.full(5, .15),
                available_water_capacity=np.full(5, .15), b_coefficient=np.full(5, 4.),
                conductivity_sat=np.full(5, 10.), root_beta=.96)


@pytest.mark.parametrize("backend", ["reference", "numba"])
@pytest.mark.parametrize("years,repeats,timing", [
    ([2014, 2015], 1, "start"), ([2014, 2015], 5, "start"),
    ([2015, 2016], 3, "end"),
])
def test_compact_equals_explicit_history(backend, years, repeats, timing):
    solve = soil_water_balance_1d
    if backend == "numba":
        pytest.importorskip("numba")
        from pysweb.swb.accelerated import soil_water_balance_1d as solve
    dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]+1}-02-17")
    config = dict(source_years=years, repeats=repeats)
    count, metadata = source_block(dates, config)
    t = np.arange(len(dates))
    rain = np.where(t % 9 == 0, 19., .1)
    et = 2. + np.sin(t / 50.)
    tr = .7 * et
    compact = solve(rain, et, properties(), dates, transpiration_data=tr,
                    spinup=config, state_timing=timing)
    expanded = [np.r_[np.tile(v[:count], repeats-1), v] for v in (rain, et, tr)]
    explicit = solve(expanded[0], expanded[1], properties(),
                     pd.date_range(end=dates[-1], periods=len(expanded[0])),
                     transpiration_data=expanded[2], state_timing=timing)
    offset = count * (repeats-1)
    np.testing.assert_allclose(compact, explicit.iloc[offset:], atol=1e-12, rtol=0)
    np.testing.assert_allclose(compact.attrs["final_state"], explicit.attrs["final_state"], atol=1e-12)
    for key, values in compact.attrs["water_balance"].items():
        np.testing.assert_allclose(values, explicit.attrs["water_balance"][key][offset:], atol=1e-10)
    assert compact.index.equals(dates)
    assert metadata["initialization_days"] == count * repeats
    assert compact.attrs["spinup"] == metadata


@pytest.mark.parametrize("config", [
    {"source_years": [2014, 2016], "repeats": 5},
    {"source_years": [2014, 2015], "repeats": 0},
    {"source_years": [2014, 2015], "repeats": True},
    {"source_years": [2015, 2016], "repeats": 5},
])
def test_reject_incomplete_or_invalid_block(config):
    with pytest.raises(ValueError):
        source_block(pd.date_range("2014-01-01", "2016-12-31"), config)


def test_missing_spinup_forcing_is_not_silently_skipped():
    dates = pd.date_range("2014-01-01", "2016-01-02")
    rain, et = np.ones(len(dates)), np.ones(len(dates))
    rain[42] = np.nan
    with pytest.raises(ValueError, match="invalid"):
        soil_water_balance_1d(rain, et, properties(), dates,
                             spinup=dict(source_years=[2014, 2015], repeats=5))


def test_native_plan_excludes_source_years_from_scoring():
    from pysweb.workflow_native import native_plan
    config = dict(run=dict(start="2014-01-01", end="2025-12-31", state_timing="start",
                           root="/tmp", name="fixture", holdout_start="2021-01-01"),
                  calibration=dict(enabled=True, start="2016-01-01", end="2020-12-31"),
                  spinup=dict(source_years=[2014, 2015], repeats=5,
                              tolerance=.005, minimum_fraction=.99))
    plan = native_plan(config, ["calibrate"])
    assert plan["initialization"]["artificial_warmup_days"] == 4 * 730
    assert plan["initialization"]["actual_warmup_days"] == 730
    config["calibration"]["start"] = "2015-01-01"
    with pytest.raises(ValueError, match="Scoring"):
        native_plan(config, ["calibrate"])


def test_failed_convergence_requires_explicit_record_action(tmp_path, monkeypatch):
    import xarray as xr
    import pysweb.workflow_protocol as protocol
    import pysweb.swb.spinup as spinup_module
    dates = pd.date_range("2014-01-01", "2015-12-31")
    forcing = xr.Dataset({name: (("time", "lat", "lon"), np.ones((len(dates), 1, 1)))
                          for name in ["effective_precip", "et", "E_t", "ndvi"]},
                         coords={"time": dates, "lat": [0.], "lon": [0.]})
    path = tmp_path / "forcing.nc"
    forcing.to_netcdf(path)
    soils = {"porosity": xr.DataArray(np.full((5, 1, 1), .45),
             dims=("layer", "lat", "lon"), coords={"layer": np.arange(1, 6), "lat": [0.], "lon": [0.]})}
    monkeypatch.setattr(protocol, "_load_soil_arrays", lambda *_: soils)
    monkeypatch.setattr(protocol, "infer_layer_bottoms", lambda *_: [50, 150, 300, 600, 1000])
    mask = soils["porosity"].isel(layer=0, drop=True).astype(bool)
    monkeypatch.setattr(spinup_module, "verify_convergence", lambda *a, **k:
                        (dict(passed=False, mode="wet_dry"), mask.astype(float)))
    monkeypatch.setattr(protocol, "atomic_netcdf", lambda *a: None)
    records = []
    monkeypatch.setattr(protocol, "atomic_json", lambda p, r: records.append(dict(r)))
    record = dict(scoring_start="2016-01-01", development_end="2020-12-31",
                  actual_start="2014-01-01", simulation_start="2014-01-01",
                  files={name: [str(path), name] for name in forcing})
    config = dict(tolerance=.005, minimum_fraction=.99)
    with pytest.raises(ValueError, match="convergence failed"):
        protocol.convergence_gate(tmp_path, record, mask, {}, config, "default")
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = protocol.convergence_gate(tmp_path, record, mask, {},
                    {**config, "convergence_action": "record"}, "default")
    assert not result["passed"]
    assert not records[-1]["passed"]


@pytest.mark.parametrize("enabled", [True, False])
def test_toml_to_native_plan_and_api_arguments(tmp_path, enabled):
    from pysweb.workflow import load_config
    from pysweb.workflow_native import native_plan
    from pysweb.workflow_protocol import initialized_forcing
    from pysweb.swb.run import _build_run_args
    path = tmp_path / "experiment.toml"
    calibration = '\n[calibration]\nenabled = true\nstart = "2016-01-01"\nend = "2020-12-31"\n' if enabled else ""
    path.write_text('''[run]
name = "fixture"
root = "."
extent = [146.2, -35.1, 146.3, -35.0]
start = "2014-01-01"
end = "2025-12-31"
holdout_start = "2021-01-01"
[spinup]
source_years = [2014, 2015]
repeats = 10
tolerance = 0.005
minimum_fraction = 0.99
convergence_action = "record"
''' + calibration)
    config = load_config(path)
    plan = native_plan(config, ["run"])
    assert plan["initialization"]["analysis_start"] == "2016-01-01"
    assert plan["initialization"]["actual_warmup_days"] == 730
    record = dict(simulation_start="2014-01-01", end="2025-12-31",
                  initialization_mode="repeated_source_block", spinup=config["spinup"],
                  files={key: [f"{key}.nc", key] for key in ["effective_precip", "et", "E_t", "ndvi"]})
    arguments = initialized_forcing(record, tmp_path, workers=1, state_timing="start")
    arguments.update(precip="precip.nc", output_dir=str(tmp_path))
    args = _build_run_args(arguments)
    assert args.spinup_source_years == [2014, 2015] and args.spinup_repeats == 10
