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
Dependencies: pytest
"""
import builtins
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysweb.swb import calibrate as calibrate_module
from pysweb.swb.calibrate import build_parser


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
