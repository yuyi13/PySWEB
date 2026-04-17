#!/usr/bin/env python3
"""
Script: test_1_ssebop_prepare_inputs.py
Objective: Verify the unified SSEBop prepare-inputs workflow exposes the package-backed CLI contract.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: Workflow module imports and CLI arguments supplied by pytest.
Outputs: Test assertions.
Usage: pytest tests/workflows/test_1_ssebop_prepare_inputs.py
Dependencies: importlib, pytest
"""
from importlib import util
from pathlib import Path


def test_unified_first_step_cli_exposes_met_source():
    workflow_path = Path(__file__).resolve().parents[2] / "workflows" / "1_ssebop_prepare_inputs.py"
    spec = util.spec_from_file_location("ssebop_prepare_inputs_workflow", workflow_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--date-range", "2024-01-01 to 2024-01-03",
            "--extent", "147.2,-35.1,147.3,-35.0",
            "--met-source", "era5land",
            "--gee-config", "/tmp/gee.yaml",
            "--out-dir", "/tmp/out",
            "--dem", "/tmp/dem.tif",
        ]
    )

    assert args.met_source == "era5land"


def test_unified_first_step_cli_calls_package_api(monkeypatch, tmp_path: Path):
    workflow_path = Path(__file__).resolve().parents[2] / "workflows" / "1_ssebop_prepare_inputs.py"
    spec = util.spec_from_file_location("ssebop_prepare_inputs_workflow", workflow_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    recorded = {}
    monkeypatch.setattr(module, "prepare_inputs", lambda **kwargs: recorded.update(kwargs))

    module.main(
        [
            "--date-range", "2024-01-01 to 2024-01-03",
            "--extent", "147.2,-35.1,147.3,-35.0",
            "--met-source", "era5land",
            "--gee-config", "/tmp/gee.yaml",
            "--out-dir", str(tmp_path / "out"),
            "--dem", str(tmp_path / "dem.tif"),
        ]
    )

    assert recorded["date_range"] == "2024-01-01 to 2024-01-03"
    assert recorded["extent"] == [147.2, -35.1, 147.3, -35.0]
    assert recorded["met_source"] == "era5land"
    assert recorded["gee_config"] == "/tmp/gee.yaml"
    assert recorded["dem"] == str(tmp_path / "dem.tif")
    assert recorded["landsat_dir"] == str(tmp_path / "out" / "landsat")
    assert recorded["met_raw_dir"] == str(tmp_path / "out" / "met" / "era5land" / "raw")
    assert recorded["met_stack_dir"] == str(tmp_path / "out" / "met" / "era5land" / "stack")
