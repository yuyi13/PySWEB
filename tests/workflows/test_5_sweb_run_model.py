#!/usr/bin/env python3
"""
Script: test_5_sweb_run_model.py
Objective: Verify the Workflow 5 CLI wrapper stays thin while exposing the package-owned SWB run parser and dispatcher.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: Workflow module imports and CLI-like argv lists supplied by pytest.
Outputs: Test assertions.
Usage: pytest tests/workflows/test_5_sweb_run_model.py
Dependencies: importlib, pathlib, sys
"""
from importlib import util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_workflow_module():
    workflow_path = Path(__file__).resolve().parents[2] / "workflows" / "5_sweb_run_model.py"
    spec = util.spec_from_file_location("sweb_run_model_workflow", workflow_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_workflow_build_parser_exposes_core_run_arguments():
    module = _load_workflow_module()
    parser = module.build_parser()

    args = parser.parse_args(
        [
            "--precip",
            "/tmp/precip.nc",
            "--effective-precip",
            "/tmp/effective_precip.nc",
            "--et",
            "/tmp/et.nc",
            "--t",
            "/tmp/t.nc",
            "--output-dir",
            "/tmp/out",
        ]
    )

    assert args.precip == "/tmp/precip.nc"
    assert args.effective_precip == "/tmp/effective_precip.nc"
    assert args.output_dir == "/tmp/out"
    assert args.workers == 1


def test_workflow_main_forwards_parsed_args_to_package_run(monkeypatch):
    module = _load_workflow_module()
    recorded = {}

    def fake_run_swb_workflow(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(module, "run_swb_workflow", fake_run_swb_workflow)

    module.main(
        [
            "--precip",
            "/tmp/precip.nc",
            "--effective-precip",
            "/tmp/effective_precip.nc",
            "--et",
            "/tmp/et.nc",
            "--t",
            "/tmp/t.nc",
            "--soil-dir",
            "/tmp/soil",
            "--output-dir",
            "/tmp/out",
            "--workers",
            "3",
        ]
    )

    assert recorded["precip"] == "/tmp/precip.nc"
    assert recorded["effective_precip"] == "/tmp/effective_precip.nc"
    assert recorded["et"] == "/tmp/et.nc"
    assert recorded["t"] == "/tmp/t.nc"
    assert recorded["soil_dir"] == "/tmp/soil"
    assert recorded["output_dir"] == "/tmp/out"
    assert recorded["workers"] == 3
