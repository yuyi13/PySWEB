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

import pytest

from core.met_input_paths import infer_met_var_from_path, resolve_met_input_paths


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
    resolved = resolve_met_input_paths(
        field="et_short_crop",
        explicit_path=None,
        met_dir=str(tmp_path / "met"),
        silo_dir=None,
        date_range="2024-01-02 to 2024-01-03",
    )

    assert resolved == str(tmp_path / "met" / "et_short_crop_daily_2024-01-02_2024-01-03.nc")


def test_silo_dir_with_date_range_resolves_legacy_year_files(tmp_path):
    resolved = resolve_met_input_paths(
        field="tmax",
        explicit_path=None,
        met_dir=None,
        silo_dir=str(tmp_path / "silo"),
        date_range="2024-12-30 to 2025-01-02",
    )

    assert resolved == [
        str(tmp_path / "silo" / "2024.max_temp.nc"),
        str(tmp_path / "silo" / "2025.max_temp.nc"),
    ]


def test_default_variable_inference_handles_era5land_and_silo_names():
    assert infer_met_var_from_path("tmax_daily_2024-01-02_2024-01-03.nc") == "tmax"
    assert infer_met_var_from_path("2024.max_temp.nc") == "max_temp"


def _load_workflow_module(monkeypatch):
    fake_yaml = types.ModuleType("yaml")
    fake_yaml.safe_load = lambda _: {}
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)

    fake_scipy = types.ModuleType("scipy")
    fake_scipy_signal = types.ModuleType("scipy.signal")
    fake_scipy_signal.savgol_filter = lambda *args, **kwargs: None
    fake_scipy.signal = fake_scipy_signal
    monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
    monkeypatch.setitem(sys.modules, "scipy.signal", fake_scipy_signal)

    workflow_path = Path(__file__).resolve().parents[2] / "workflows" / "2_ssebop_run_model.py"
    spec = util.spec_from_file_location("ssebop_run_model_workflow", workflow_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_workflow_help_includes_met_dir_and_bootstraps_project_root(monkeypatch, capsys):
    workflow_module = _load_workflow_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["2_ssebop_run_model.py", "--help"])

    with pytest.raises(SystemExit) as exc:
        workflow_module.main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--met-dir" in captured.out
    assert "meteorology" in captured.out.lower()
