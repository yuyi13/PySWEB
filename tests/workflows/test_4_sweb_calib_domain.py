#!/usr/bin/env python3
"""
Script: test_4_sweb_calib_domain.py
Objective: Verify Workflow 4 remains a thin wrapper around the package-owned SWB calibration entry points.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Workflow module loading and monkeypatched calibration dispatch under pytest.
Outputs: Test assertions.
Usage: pytest tests/workflows/test_4_sweb_calib_domain.py
Dependencies: pytest
"""
from importlib import util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_workflow_module():
    workflow_path = ROOT / "workflows" / "4_sweb_calib_domain.py"
    spec = util.spec_from_file_location("sweb_calibrate_workflow", workflow_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_workflow_main_forwards_reference_ssm_args(monkeypatch):
    workflow_module = _load_workflow_module()
    recorded = {}

    def fake_calibrate_domain(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(workflow_module, "calibrate_domain", fake_calibrate_domain)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "4_sweb_calib_domain.py",
            "--effective-precip", "/tmp/effective.nc",
            "--et", "/tmp/et.nc",
            "--t", "/tmp/t.nc",
            "--soil-dir", "/tmp/soil",
            "--reference-ssm", "/tmp/reference.nc",
            "--output", "/tmp/calibration.csv",
        ],
    )

    workflow_module.main()

    assert recorded["reference_ssm"] == "/tmp/reference.nc"
    assert recorded["reference_var"] == "reference_ssm"


def test_workflow_main_rejects_legacy_smap_flag(monkeypatch):
    workflow_module = _load_workflow_module()

    with pytest.raises(SystemExit) as exc_info:
        workflow_module.main([
            "--effective-precip", "/tmp/effective.nc",
            "--et", "/tmp/et.nc",
            "--t", "/tmp/t.nc",
            "--soil-dir", "/tmp/soil",
            "--smap-ssm", "/tmp/reference.nc",
            "--output", "/tmp/calibration.csv",
        ])

    assert exc_info.value.code == 2


def test_shell_runner_uses_reference_ssm_flag_for_calibration():
    runner_path = ROOT / "workflows" / "sweb_domain_runner.sh"
    runner_text = runner_path.read_text(encoding = "utf-8")

    assert '--reference-ssm "${CALIB_SMAP_SSM_FILE}"' in runner_text
    assert '--smap-ssm "${CALIB_SMAP_SSM_FILE}"' not in runner_text
