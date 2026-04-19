#!/usr/bin/env python3
"""
Script: test_calibrate.py
Objective: Verify the package-owned SWB calibration parser exposes the neutral reference SSM interface.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Package calibration parser construction and CLI argument parsing under pytest.
Outputs: Test assertions.
Usage: pytest tests/swb/test_calibrate.py
Dependencies: numpy, pandas, pytest, xarray
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

    monkeypatch.setattr(calibrate_module, "_load_single_variable", lambda path, var = None: data_by_path[path.name])
    monkeypatch.setattr(calibrate_module, "_load_soil_arrays", lambda soil_dir: soil_arrays)
    monkeypatch.setattr(
        calibrate_module,
        "_get_differential_evolution",
        lambda: (
            lambda *args, **kwargs: type("FakeResult", (), {"x": np.array([1000.0, 1.0, 1.0, 0.96], dtype=float)})()
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


def test_calibrate_domain_accepts_noon_stamped_daily_inputs(monkeypatch, tmp_path: Path):
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

    monkeypatch.setattr(calibrate_module, "_load_single_variable", lambda path, var = None: data_by_path[path.name])
    monkeypatch.setattr(calibrate_module, "_load_soil_arrays", lambda soil_dir: soil_arrays)
    monkeypatch.setattr(
        calibrate_module,
        "_get_differential_evolution",
        lambda: (
            lambda *args, **kwargs: type("FakeResult", (), {"x": np.array([1000.0, 1.0, 1.0, 0.96], dtype=float)})()
        ),
    )
    monkeypatch.setattr(calibrate_module, "_compute_rmse", lambda *args, **kwargs: (0.42, 2))

    output_path = tmp_path / "calibration.csv"
    calibrate_module.calibrate_domain(
        effective_precip = str(tmp_path / "effective.nc"),
        et            = str(tmp_path / "et.nc"),
        t             = str(tmp_path / "t.nc"),
        soil_dir      = str(tmp_path / "soil"),
        reference_ssm = str(tmp_path / "reference.nc"),
        date_range    = ["2024-01-01", "2024-01-02"],
        output        = str(output_path),
        sm_res        = None,
    )

    assert output_path.exists()


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

    monkeypatch.setattr(calibrate_module, "_load_single_variable", lambda path, var = None: data_by_path[path.name])
    monkeypatch.setattr(calibrate_module, "_load_soil_arrays", lambda soil_dir: soil_arrays)
    monkeypatch.setattr(
        calibrate_module,
        "_get_differential_evolution",
        lambda: (
            lambda *args, **kwargs: type("FakeResult", (), {"x": np.array([1000.0, 1.0, 1.0, 0.96], dtype=float)})()
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
