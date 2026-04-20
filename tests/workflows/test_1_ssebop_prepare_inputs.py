#!/usr/bin/env python3
"""
Script: test_1_ssebop_prepare_inputs.py
Objective: Verify the unified SSEBop prepare-inputs workflow exposes the package-backed CLI contract.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-20
Inputs: Workflow module imports and CLI arguments supplied by pytest.
Outputs: Test assertions.
Usage: pytest tests/workflows/test_1_ssebop_prepare_inputs.py
Dependencies: importlib, pytest
"""
from importlib import util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
            "--gee-project", "workflow-project",
            "--out-dir", "/tmp/out",
        ]
    )

    assert args.met_source == "era5land"
    assert args.gee_project == "workflow-project"
    assert not hasattr(args, "dem")
    assert not hasattr(args, "gee_config")
    assert module.parse_extent.__module__ == "pysweb.ssebop.landsat"


def test_unified_first_step_cli_rejects_unwired_met_source():
    workflow_path = Path(__file__).resolve().parents[2] / "workflows" / "1_ssebop_prepare_inputs.py"
    spec = util.spec_from_file_location("ssebop_prepare_inputs_workflow", workflow_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--date-range", "2024-01-01 to 2024-01-03",
                "--extent", "147.2,-35.1,147.3,-35.0",
                "--met-source", "silo",
                "--gee-project", "workflow-project",
                "--out-dir", "/tmp/out",
            ]
        )


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
            "--gee-project", "workflow-project",
            "--out-dir", str(tmp_path / "out"),
        ]
    )

    assert recorded["date_range"] == "2024-01-01 to 2024-01-03"
    assert recorded["extent"] == [147.2, -35.1, 147.3, -35.0]
    assert recorded["met_source"] == "era5land"
    assert recorded["gee_project"] == "workflow-project"
    assert recorded["gee_config_template"] is None
    assert recorded["dem_source"] == "nasadem"
    assert recorded["dem_dir"] == str(tmp_path / "out" / "dem")
    assert recorded["landsat_dir"] == str(tmp_path / "out" / "landsat")
    assert recorded["met_raw_dir"] == str(tmp_path / "out" / "met" / "era5land" / "raw")
    assert recorded["met_stack_dir"] == str(tmp_path / "out" / "met" / "era5land" / "stack")


def test_unified_first_step_cli_requires_gee_project():
    workflow_path = Path(__file__).resolve().parents[2] / "workflows" / "1_ssebop_prepare_inputs.py"
    spec = util.spec_from_file_location("ssebop_prepare_inputs_workflow", workflow_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--date-range", "2024-01-01 to 2024-01-03",
                "--extent", "147.2,-35.1,147.3,-35.0",
                "--met-source", "era5land",
                "--out-dir", "/tmp/out",
            ]
        )


def test_unified_first_step_cli_rejects_blank_gee_project(monkeypatch, tmp_path: Path):
    workflow_path = Path(__file__).resolve().parents[2] / "workflows" / "1_ssebop_prepare_inputs.py"
    spec = util.spec_from_file_location("ssebop_prepare_inputs_workflow", workflow_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    recorded = {}
    monkeypatch.setattr(module, "prepare_inputs", lambda **kwargs: recorded.update(kwargs))

    try:
        module.main(
            [
                "--date-range", "2024-01-01 to 2024-01-03",
                "--extent", "147.2,-35.1,147.3,-35.0",
                "--met-source", "era5land",
                "--gee-project", "   ",
                "--out-dir", str(tmp_path / "out"),
            ]
        )
    except ValueError as exc:
        assert "gee_project" in str(exc)
    else:
        raise AssertionError("Expected workflow main to reject blank gee_project values")

    assert recorded == {}
