#!/usr/bin/env python3
"""
Script: test_3_sweb_preprocess_inputs.py
Objective: Verify Workflow 3 delegates to the package-owned SWB preprocess CLI surface.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Pytest execution against the workflow wrapper module.
Outputs: Regression coverage for help text and forwarded preprocess arguments.
Usage: python -m pytest tests/workflows/test_3_sweb_preprocess_inputs.py -q
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
    workflow_path = ROOT / "workflows" / "3_sweb_preprocess_inputs.py"
    spec = util.spec_from_file_location("sweb_preprocess_workflow", workflow_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_workflow_help_uses_reference_ssm_terms(monkeypatch, capsys):
    workflow_module = _load_workflow_module()
    monkeypatch.setattr(sys, "argv", ["3_sweb_preprocess_inputs.py", "--help"])

    with pytest.raises(SystemExit) as exc:
        workflow_module.main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--reference-source" in captured.out
    assert "--reference-ssm-asset" in captured.out
    assert "--skip-reference-ssm" in captured.out
    assert "--skip-smap" not in captured.out


def test_workflow_main_forwards_to_package_preprocess(monkeypatch):
    workflow_module = _load_workflow_module()
    recorded = {}

    def fake_preprocess_inputs(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(workflow_module, "preprocess_inputs", fake_preprocess_inputs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "3_sweb_preprocess_inputs.py",
            "--date-range", "2024-01-01", "2024-01-03",
            "--extent", "148.0", "-35.5", "148.1", "-35.4",
            "--sm-res", "0.01",
            "--output-dir", "/tmp/prepped",
        ],
    )

    workflow_module.main()

    assert recorded["output_dir"] == "/tmp/prepped"
    assert recorded["reference_source"] == "gssm1km"
    assert recorded["gee_project"] == "yiyu-research"


def test_workflow_main_accepts_runner_preprocess_contract(monkeypatch):
    workflow_module = _load_workflow_module()
    recorded = {}

    def fake_preprocess_inputs(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(workflow_module, "preprocess_inputs", fake_preprocess_inputs)

    workflow_module.main(
        [
            "--date-range", "2024-01-01", "2024-01-02",
            "--extent", "147.2", "-35.1", "147.3", "-35.0",
            "--sm-res", "0.00025",
            "--workers", "4",
            "--rain-file", "/tmp/rain.nc",
            "--rain-var", "precipitation",
            "--et-file", "/tmp/et.nc",
            "--e-var", "E",
            "--et-var", "ET",
            "--t-var", "T",
            "--output-dir", "/tmp/preprocessed",
        ]
    )

    assert recorded["rain_file"] == "/tmp/rain.nc"
    assert recorded["et_file"] == "/tmp/et.nc"
    assert recorded["workers"] == 4
    assert recorded["reference_source"] == "gssm1km"
