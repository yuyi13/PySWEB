#!/usr/bin/env python3
"""
Script: test_calibrate.py
Objective: Verify calibration arguments, optimizer seeds, daily time alignment and rejection of invalid fit metrics.
Author: Yi Yu (with assistance from Codex)
Created: 2026-04-19
Last updated: 2026-10-20
Inputs: Package calibration parser construction and CLI argument parsing under pytest.
Outputs: Test assertions.
Usage: python -m pytest tests/swb/test_calibrate.py
Dependencies: numpy, pandas, pytest, xarray; pysweb
"""
import builtins
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysweb.swb import calibrate as calibrate_module
from pysweb.swb.calibrate import build_parser


def _forcing_cube(name: str, dates: list[str], value: float = 1.0) -> xr.DataArray:
    return xr.DataArray(
        np.full((len(dates), 1, 1), value, dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.to_datetime(dates),
            "lat": np.array([-35.0], dtype=float),
            "lon": np.array([148.0], dtype=float),
        },
        name=name,
    )


def _soil_cube(name: str, value: float) -> xr.DataArray:
    return xr.DataArray(
        np.full((5, 1, 1), value, dtype=np.float32),
        dims=("layer", "lat", "lon"),
        coords={
            "layer": np.array([1, 2, 3, 4, 5], dtype=int),
            "lat": np.array([-35.0], dtype=float),
            "lon": np.array([148.0], dtype=float),
        },
        name=name,
    )


def test_calibration_parser_uses_reference_ssm_names():
    parser = build_parser()
    help_text = parser.format_help()

    assert "--reference-ssm" in help_text
    assert "--reference-var" in help_text
    assert "--smap-ssm" not in help_text
    assert "--smap-var" not in help_text


def test_calibration_parser_defaults_reference_var_to_reference_ssm():
    parser = build_parser()
    args = parser.parse_args([
        "--effective-precip", "/tmp/effective.nc",
        "--et", "/tmp/et.nc",
        "--t", "/tmp/t.nc",
        "--soil-dir", "/tmp/soil",
        "--reference-ssm", "/tmp/reference.nc",
        "--output", "/tmp/calibration.csv",
    ])

    assert args.reference_var == "reference_ssm"


def test_get_differential_evolution_raises_clear_error_when_scipy_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals = None, locals = None, fromlist = (), level = 0):
        if name == "scipy.optimize":
            raise ModuleNotFoundError("No module named 'scipy'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match = "scipy is required for SWB calibration"):
        calibrate_module._get_differential_evolution()


def test_namespace_from_kwargs_rejects_unknown_arguments():
    with pytest.raises(TypeError, match = "smap_ssm"):
        calibrate_module._namespace_from_kwargs(
            {
                "effective_precip": "/tmp/effective.nc",
                "et": "/tmp/et.nc",
                "t": "/tmp/t.nc",
                "soil_dir": "/tmp/soil",
                "reference_ssm": "/tmp/reference.nc",
                "output": "/tmp/calibration.csv",
                "date_range": ["2024-01-01", "2024-01-02"],
                "smap_ssm": "/tmp/legacy.nc",
            }
        )


def test_calibrate_domain_rejects_requested_window_with_no_overlapping_timesteps(monkeypatch, tmp_path: Path):
    data_by_path = {
        "effective.nc": _forcing_cube("effective_precipitation", ["2024-01-01", "2024-01-02"]),
        "et.nc": _forcing_cube("et", ["2024-01-01", "2024-01-02"]),
        "t.nc": _forcing_cube("t", ["2024-01-01", "2024-01-02"]),
        "reference.nc": _forcing_cube("reference_ssm", ["2024-01-01", "2024-01-02"], value = 0.3),
    }
    soil_arrays = {
        "porosity": _soil_cube("porosity", 0.45),
        "wilting_point": _soil_cube("wilting_point", 0.15),
        "available_water_capacity": _soil_cube("available_water_capacity", 0.20),
        "b_coefficient": _soil_cube("b_coefficient", 4.5),
        "conductivity_sat": _soil_cube("conductivity_sat", 8.0),
    }

    monkeypatch.setattr(calibrate_module, "_load_single_variable", lambda path, var = None, **kwargs: data_by_path[path.name])
    monkeypatch.setattr(calibrate_module, "_load_soil_arrays", lambda soil_dir: soil_arrays)
    monkeypatch.setattr(
        calibrate_module,
        "_get_differential_evolution",
        lambda: (
            lambda *args, **kwargs: type("FakeResult", (), {"x": np.array([1000.0, 0.96], dtype=float), "success": True})()
        ),
    )

    with pytest.raises(ValueError, match = "No effective precipitation timesteps overlap"):
        calibrate_module.calibrate_domain(
            effective_precip = str(tmp_path / "effective.nc"),
            et            = str(tmp_path / "et.nc"),
            t             = str(tmp_path / "t.nc"),
            soil_dir      = str(tmp_path / "soil"),
            reference_ssm = str(tmp_path / "reference.nc"),
            date_range    = ["2024-02-01", "2024-02-02"],
            output        = str(tmp_path / "calibration.csv"),
            sm_res        = None,
        )


@pytest.mark.parametrize("root_beta", [None, 0.93])
def test_calibrate_domain_accepts_noon_stamped_daily_inputs(monkeypatch, tmp_path: Path, root_beta):
    data_by_path = {
        "effective.nc": _forcing_cube("effective_precipitation", ["2024-01-01 12:00:00", "2024-01-02 12:00:00"]),
        "et.nc": _forcing_cube("et", ["2024-01-01 12:00:00", "2024-01-02 12:00:00"]),
        "t.nc": _forcing_cube("t", ["2024-01-01 12:00:00", "2024-01-02 12:00:00"]),
        "reference.nc": _forcing_cube("reference_ssm", ["2024-01-01 12:00:00", "2024-01-02 12:00:00"], value = 0.3),
    }
    soil_arrays = {
        "porosity": _soil_cube("porosity", 0.45),
        "wilting_point": _soil_cube("wilting_point", 0.15),
        "available_water_capacity": _soil_cube("available_water_capacity", 0.20),
        "b_coefficient": _soil_cube("b_coefficient", 4.5),
        "conductivity_sat": _soil_cube("conductivity_sat", 8.0),
    }

    monkeypatch.setattr(calibrate_module, "_load_single_variable", lambda path, var = None, **kwargs: data_by_path[path.name])
    monkeypatch.setattr(calibrate_module, "_load_soil_arrays", lambda soil_dir: soil_arrays)
    optimizer_seeds = []

    def fake_de(*args, x0, **kwargs):
        optimizer_seeds.append(x0.copy())
        return type("FakeResult", (), {"x": x0.copy(), "success": True})()

    monkeypatch.setattr(calibrate_module, "_get_differential_evolution", lambda: fake_de)
    monkeypatch.setattr(calibrate_module, "_compute_rmse", lambda *args, **kwargs: (0.42, 2))

    # Keep real input files available to the provenance writer; only optimization is mocked.
    for filename, da in data_by_path.items():
        da.to_netcdf(tmp_path / filename)
    (tmp_path / "soil").mkdir()
    for name, da in soil_arrays.items():
        da.to_netcdf(tmp_path / "soil" / f"soil_{name}.nc")
    output_path = tmp_path / "calibration.csv"
    parameter_options = {} if root_beta is None else {"root_beta": root_beta}
    calibrate_module.calibrate_domain(
        effective_precip = str(tmp_path / "effective.nc"),
        et            = str(tmp_path / "et.nc"),
        t             = str(tmp_path / "t.nc"),
        soil_dir      = str(tmp_path / "soil"),
        reference_ssm = str(tmp_path / "reference.nc"),
        date_range    = ["2024-01-01", "2024-01-02"],
        output        = str(output_path),
        sm_res        = None,
        **parameter_options,
    )

    assert output_path.exists()
    np.testing.assert_array_equal(optimizer_seeds, [[1000., .96 if root_beta is None else root_beta]])


def test_calibrate_domain_rejects_invalid_rmse_outputs(monkeypatch, tmp_path: Path):
    data_by_path = {
        "effective.nc": _forcing_cube("effective_precipitation", ["2024-01-01", "2024-01-02"]),
        "et.nc": _forcing_cube("et", ["2024-01-01", "2024-01-02"]),
        "t.nc": _forcing_cube("t", ["2024-01-01", "2024-01-02"]),
        "reference.nc": _forcing_cube("reference_ssm", ["2024-01-01", "2024-01-02"], value = 0.3),
    }
    soil_arrays = {
        "porosity": _soil_cube("porosity", 0.45),
        "wilting_point": _soil_cube("wilting_point", 0.15),
        "available_water_capacity": _soil_cube("available_water_capacity", 0.20),
        "b_coefficient": _soil_cube("b_coefficient", 4.5),
        "conductivity_sat": _soil_cube("conductivity_sat", 8.0),
    }

    monkeypatch.setattr(calibrate_module, "_load_single_variable", lambda path, var = None, **kwargs: data_by_path[path.name])
    monkeypatch.setattr(calibrate_module, "_load_soil_arrays", lambda soil_dir: soil_arrays)
    monkeypatch.setattr(
        calibrate_module,
        "_get_differential_evolution",
        lambda: (
            lambda *args, **kwargs: type("FakeResult", (), {"x": np.array([1000.0, 0.96], dtype=float), "success": True})()
        ),
    )
    monkeypatch.setattr(calibrate_module, "_compute_rmse", lambda *args, **kwargs: (float("inf"), 0))

    with pytest.raises(ValueError, match = "no valid observations"):
        calibrate_module.calibrate_domain(
            effective_precip = str(tmp_path / "effective.nc"),
            et            = str(tmp_path / "et.nc"),
            t             = str(tmp_path / "t.nc"),
            soil_dir      = str(tmp_path / "soil"),
            reference_ssm = str(tmp_path / "reference.nc"),
            date_range    = ["2024-01-01", "2024-01-02"],
            output        = str(tmp_path / "calibration.csv"),
            sm_res        = None,
        )


def test_cached_objective_and_numba_preserve_missing_reference_support():
    pytest.importorskip("numba")
    rng = np.random.default_rng(18)
    times = pd.date_range("2010-01-01", periods=120)
    rain = rng.exponential(5, (120, 1, 3))
    et = rng.uniform(0, 6, rain.shape)
    tr = et * .6
    reference = rng.uniform(.1, .4, rain.shape)
    reference[::7, 0, 0] = np.nan
    reference[::9, 0, 1] = np.nan
    rain[4, 0, 2] = np.nan
    valid = np.ones((1,3), dtype=bool)
    weights = np.array([[1.,2.,3.]])
    soil = {name:np.full((5,1,3), value) for name,value in
            [("porosity",.45),("wilting_point",.15),("available_water_capacity",.2),
             ("b_coefficient",4.5),("conductivity_sat",8.)]}
    args = (rain,et,tr,reference,times,soil,valid,None,[50,150,300,600,1000],0,.5,25.,0.,False,weights,"start",30)
    params = [1000.,.96]
    expected = calibrate_module._compute_rmse(params,*args)
    ctx = calibrate_module._prepare_observation_context(rain,et,tr,reference,valid,weights,30,"numba")
    actual = calibrate_module._compute_rmse(params,*args,ctx)
    assert actual[0] == pytest.approx(expected[0],abs=1e-12)
    assert actual[1] == expected[1]
    assert not ctx["obs_mask"][:30].any()
    assert not ctx["obs_mask"][:,:,2].any()
    assert not ctx["obs_mask"].flags.writeable
    assert weights.flags.writeable
