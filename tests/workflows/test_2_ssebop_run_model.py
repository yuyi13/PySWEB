#!/usr/bin/env python3
"""
Script: test_2_ssebop_run_model.py
Objective: Verify meteorology path resolution and workflow bootstrap behavior for the SSEBop runner.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-04-16
Inputs: Temporary paths, helper-module imports, and workflow CLI invocations.
Outputs: Test assertions.
Usage: pytest tests/workflows/test_2_ssebop_run_model.py
Dependencies: pathlib, subprocess, sys, pytest
"""
from importlib import util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import xarray as xr

from core.met_input_paths import infer_met_var_from_path, resolve_met_input_paths


def test_explicit_missing_meteorology_file_raises_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError, match="custom_tmax\\.nc"):
        resolve_met_input_paths(
            field="tmax",
            explicit_path=str(tmp_path / "custom_tmax.nc"),
            met_dir=None,
            silo_dir=None,
            date_range=None,
        )


def test_missing_met_dir_product_raises_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError, match="tmax_daily_2024-01-02_2024-01-03\\.nc"):
        resolve_met_input_paths(
            field="tmax",
            explicit_path=None,
            met_dir=str(tmp_path / "met"),
            silo_dir=None,
            date_range="2024-01-02 to 2024-01-03",
        )


def test_explicit_meteorology_paths_override_inference(tmp_path):
    explicit = tmp_path / "custom_tmax.nc"
    explicit.write_text("", encoding="utf-8")

    resolved = resolve_met_input_paths(
        field="tmax",
        explicit_path=str(explicit),
        met_dir=str(tmp_path / "met"),
        silo_dir=str(tmp_path / "silo"),
        date_range="2024-01-02 to 2024-01-03",
    )

    assert resolved == str(explicit)


def test_met_dir_with_date_range_resolves_era5land_stack_names(tmp_path):
    met_dir = tmp_path / "met"
    met_dir.mkdir()
    (met_dir / "et_short_crop_daily_2024-01-02_2024-01-03.nc").write_text("", encoding="utf-8")

    resolved = resolve_met_input_paths(
        field="et_short_crop",
        explicit_path=None,
        met_dir=str(met_dir),
        silo_dir=None,
        date_range="2024-01-02 to 2024-01-03",
    )

    assert resolved == str(met_dir / "et_short_crop_daily_2024-01-02_2024-01-03.nc")


def test_silo_dir_with_date_range_resolves_legacy_year_files(tmp_path):
    silo_dir = tmp_path / "silo"
    silo_dir.mkdir()
    (silo_dir / "2024.max_temp.nc").write_text("", encoding="utf-8")
    (silo_dir / "2025.max_temp.nc").write_text("", encoding="utf-8")

    resolved = resolve_met_input_paths(
        field="tmax",
        explicit_path=None,
        met_dir=None,
        silo_dir=str(silo_dir),
        date_range="2024-12-30 to 2025-01-02",
    )

    assert resolved == [
        str(silo_dir / "2024.max_temp.nc"),
        str(silo_dir / "2025.max_temp.nc"),
    ]


def test_default_variable_inference_handles_era5land_and_silo_names():
    assert infer_met_var_from_path("tmax_daily_2024-01-02_2024-01-03.nc") == "tmax"
    assert infer_met_var_from_path("2024.max_temp.nc") == "max_temp"
    assert infer_met_var_from_path("custom_tmax.nc", default_var="tmax") == "tmax"
    assert infer_met_var_from_path("custom_tmin.nc", default_var="tmin") == "tmin"


def test_workflow_open_meteorology_da_prefers_field_defaults_for_custom_files(tmp_path, monkeypatch):
    workflow_module = _load_workflow_module(monkeypatch)

    custom_path = tmp_path / "custom_tmax.nc"
    custom_ds = xr.Dataset(
        {"tmax": (("y", "x"), np.array([[1.0]], dtype=np.float32))},
        coords={"x": np.array([0.5]), "y": np.array([0.5])},
    )
    custom_ds.to_netcdf(custom_path)

    silo_path = tmp_path / "2024.max_temp.nc"
    silo_ds = xr.Dataset(
        {"max_temp": (("y", "x"), np.array([[2.0]], dtype=np.float32))},
        coords={"x": np.array([0.5]), "y": np.array([0.5])},
    )
    silo_ds.to_netcdf(silo_path)

    custom_da = workflow_module.open_meteorology_da(str(custom_path), None, default_var="tmax")
    silo_da = workflow_module.open_meteorology_da(str(silo_path), None, default_var="tmax")

    assert custom_da.name == "tmax"
    assert silo_da.name == "max_temp"


def _load_workflow_module(monkeypatch):
    original_yaml = sys.modules.get("yaml")
    original_scipy = sys.modules.get("scipy")
    original_scipy_signal = sys.modules.get("scipy.signal")

    fake_yaml = types.ModuleType("yaml")
    fake_yaml.safe_load = lambda _: {}
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)

    fake_scipy = types.ModuleType("scipy")
    fake_scipy_signal = types.ModuleType("scipy.signal")
    fake_scipy_signal.savgol_filter = lambda *args, **kwargs: None
    fake_scipy.__spec__ = util.spec_from_loader("scipy", loader=None)
    fake_scipy_signal.__spec__ = util.spec_from_loader("scipy.signal", loader=None)
    fake_scipy.signal = fake_scipy_signal
    monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
    monkeypatch.setitem(sys.modules, "scipy.signal", fake_scipy_signal)

    workflow_path = Path(__file__).resolve().parents[2] / "workflows" / "2_ssebop_run_model.py"
    spec = util.spec_from_file_location("ssebop_run_model_workflow", workflow_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    if original_yaml is None:
        monkeypatch.delitem(sys.modules, "yaml", raising=False)
    else:
        monkeypatch.setitem(sys.modules, "yaml", original_yaml)
    if original_scipy is None:
        monkeypatch.delitem(sys.modules, "scipy", raising=False)
    else:
        monkeypatch.setitem(sys.modules, "scipy", original_scipy)
    if original_scipy_signal is None:
        monkeypatch.delitem(sys.modules, "scipy.signal", raising=False)
    else:
        monkeypatch.setitem(sys.modules, "scipy.signal", original_scipy_signal)
    return module


def test_workflow_help_includes_met_dir_and_bootstraps_project_root(monkeypatch, capsys):
    workflow_module = _load_workflow_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["2_ssebop_run_model.py", "--help"])

    with pytest.raises(SystemExit) as exc:
        workflow_module.main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--met-temp-units" in captured.out
    assert "meteorology" in captured.out.lower()
