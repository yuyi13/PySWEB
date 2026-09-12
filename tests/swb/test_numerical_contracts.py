"""
Script: test_numerical_contracts.py
Objective: Verify conservation, restart states, forcing calendars, calibration support and output identity.
Author: Yi Yu (with assistance from Codex)
Created: 2026-09-12
Last updated: 2026-09-12
Inputs: Synthetic soil profiles, daily forcing arrays, demo files and provenance manifests.
Outputs: Regression assertions.
Usage: python -m pytest tests/swb/test_numerical_contracts.py
Dependencies: numpy, pandas, pytest, xarray; pysweb
"""

from importlib import import_module
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pysweb.contracts import daily_data, geographic_weights, layer_bottoms
from pysweb.demo import run_demo, prepare_demo
from pysweb.swb.solver import soil_water_balance_1d
from pysweb.swb.preprocess import compute_effective_precipitation_smith
from pysweb.io.provenance import input_manifest, validate_existing_output


def properties(n=2):
    return {
        "layer_depth": np.cumsum(np.full(n, 100.0)),
        "layer_thickness": np.full(n, 100.0),
        "porosity": np.full(n, 0.45),
        "wilting_point": np.full(n, 0.12),
        "available_water_capacity": np.full(n, 0.18),
        "b_coefficient": np.full(n, 4.0),
        "conductivity_sat": np.full(n, 8.0),
        "root_beta": 0.96,
        "drainage_slope": 0.5,
        "drainage_upper_limit": 25.0,
        "drainage_lower_limit": 0.0,
        "sm_max_factor": 1.0,
        "sm_min_factor": 1.0,
    }


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize(
    "rain,et", [(0.0, 0.0), (12.0, 3.0), (200.0, 2.0), (0.0, 200.0)]
)
def test_water_budget_closes_for_dry_wet_and_saturated_profiles(n, rain, et):
    out = soil_water_balance_1d(
        np.full(10, rain),
        np.full(10, et),
        properties(n),
        pd.date_range("2024-01-01", periods=10),
    )
    budget = out.attrs["water_balance"]
    np.testing.assert_allclose(budget["balance_residual"], 0, atol=1e-8)
    total_change = np.dot(
        out.attrs["final_state"] - out.attrs["initial_state"],
        properties(n)["layer_thickness"],
    )
    np.testing.assert_allclose(
        total_change, np.sum(budget["storage_change"]), atol=1e-8
    )
    assert np.isfinite(out.values).all()
    assert np.min(out.values) >= 0
    if rain == 200:
        assert np.sum(budget["overflow"]) > 0


def test_state_timing_and_restart_are_consistent():
    time = pd.date_range("2024-01-01", periods=4)
    start = soil_water_balance_1d([10, 0, 0, 3], [2] * 4, properties(), time)
    end = soil_water_balance_1d(
        [10, 0, 0, 3], [2] * 4, properties(), time, state_timing="end"
    )
    np.testing.assert_allclose(start.values[1:], end.values[:-1])
    np.testing.assert_allclose(end.values[-1], start.attrs["final_state"])
    first = soil_water_balance_1d([10, 0], [2] * 2, properties(), time[:2])
    second = soil_water_balance_1d(
        [0, 3],
        [2] * 2,
        properties(),
        time[2:],
        initial_soil_moisture=first.attrs["final_state"],
    )
    np.testing.assert_allclose(second.values, start.values[2:])


def test_missing_forcing_freezes_state_with_explicit_qc():
    out = soil_water_balance_1d(
        [5, np.nan, 1], [2] * 3, properties(), pd.date_range("2024-01-01", periods=3)
    )
    np.testing.assert_array_equal(out.attrs["forcing_valid"], [True, False, True])
    np.testing.assert_allclose(out.values[1], out.values[2])
    assert np.isnan(out.attrs["water_balance"]["actual_et"][1])


@pytest.mark.parametrize(
    "dates",
    [
        ["2024-01-01", "2024-01-03"],
        ["2024-01-01", "2024-01-01"],
        ["2024-01-02", "2024-01-01"],
    ],
)
def test_bad_daily_calendars_are_rejected(dates):
    da = xr.DataArray([1.0, 2.0], dims="time", coords={"time": pd.to_datetime(dates)})
    with pytest.raises(ValueError):
        daily_data(da)


@pytest.mark.parametrize(
    "depths", [[100, 100], [100, 50], [0, 100], [100, float("nan")]]
)
def test_invalid_layer_geometry_rejected(depths):
    with pytest.raises(ValueError):
        layer_bottoms(depths)


def test_smith_requires_whole_month_and_matches_overlapping_windows():
    rain = xr.DataArray(
        np.full((31, 1, 1), 5.0),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2024-01-01", periods=31),
            "lat": [0.0],
            "lon": [0.0],
        },
    )
    whole = compute_effective_precipitation_smith(rain, "float64")
    with pytest.raises(ValueError, match="missing"):
        compute_effective_precipitation_smith(rain.isel(time=slice(0, 14)), "float64")
    # Identical month context must yield identical values in any reported subwindow.
    np.testing.assert_allclose(
        whole.sel(time=slice("2024-01-04", "2024-01-10")), whole.values[3:10]
    )
    missing = rain.copy()
    missing[5] = np.nan
    assert np.isnan(compute_effective_precipitation_smith(missing, "float64")).all()


def test_area_weights_preserve_geographic_support():
    da = xr.DataArray(
        [[1.0, 1.0], [3.0, 3.0]],
        dims=("lat", "lon"),
        coords={"lat": [0.0, 60.0], "lon": [0.0, 1.0]},
    )
    assert float(da.weighted(geographic_weights(da)).mean()) == pytest.approx(5 / 3)


def test_solver_failure_penalises_calibration_candidate(monkeypatch):
    cal = import_module("pysweb.swb.calibrate")

    def fail(*args, **kwargs):
        raise ValueError("unstable candidate")

    monkeypatch.setattr(cal, "soil_water_balance_1d", fail)
    soil = {
        key: np.full((2, 1, 2), value)
        for key, value in [
            ("porosity", 0.45),
            ("wilting_point", 0.12),
            ("available_water_capacity", 0.18),
            ("b_coefficient", 4.0),
            ("conductivity_sat", 8.0),
        ]
    }
    data = np.ones((3, 1, 2))
    score, count = cal._compute_rmse(
        [1000, 1, 1, 0.96],
        data,
        data,
        data,
        data * 0.3,
        pd.date_range("2024-01-01", periods=3),
        soil,
        np.ones((1, 2), bool),
        None,
        [100, 200],
        0,
        0.5,
        25.0,
        0.0,
        False,
    )
    assert np.isinf(score) and count == 0


def test_offline_demo_and_provenance_integrity(tmp_path):
    result = run_demo(tmp_path)
    with xr.open_dataset(result) as ds:
        assert ds.sizes["time"] == 14
        assert bool((ds.valid_forcing_fraction == 1).all())
        assert "final_sm_layer_2" in ds
        assert float(abs(ds.balance_residual).max()) < 1e-6
    manifest = json.loads(result.with_name(result.name + ".manifest.json").read_text())
    validate_existing_output(result, manifest)
    changed = {**manifest, "fingerprint": "different-input"}
    with pytest.raises(ValueError, match="does not match"):
        validate_existing_output(result, changed)
    with result.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        validate_existing_output(result, manifest)
