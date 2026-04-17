#!/usr/bin/env python3
"""
Script: test_runners.py
Objective: Verify the workflow shell wrappers stay thin and point at the package-backed entrypoints.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: Workflow shell scripts and subprocess invocations supplied by pytest.
Outputs: Test assertions.
Usage: pytest tests/workflows/test_runners.py
Dependencies: pathlib, subprocess
"""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / "workflows"


def _run_help(script_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WORKFLOWS_DIR / script_name), "--help"],
        cwd = ROOT,
        capture_output = True,
        text = True,
        check = False,
    )


def test_ssebop_runner_help_returns_zero():
    result = _run_help("ssebop_runner_landsat.sh")

    assert result.returncode == 0, result.stderr


def test_sweb_runner_help_returns_zero():
    result = _run_help("sweb_domain_runner.sh")

    assert result.returncode == 0, result.stderr


def test_ssebop_runner_uses_package_backed_workflow_scripts():
    script_text = (WORKFLOWS_DIR / "ssebop_runner_landsat.sh").read_text(encoding = "utf-8")

    assert '${SCRIPT_DIR}/1_ssebop_prepare_inputs.py' in script_text
    assert '${SCRIPT_DIR}/2_ssebop_run_model.py' in script_text
    assert "1b_download_era5land_daily.py" not in script_text
    assert "1c_stack_era5land_daily.py" not in script_text
    assert '${RUN_PREPARED_DIR}/landsat' in script_text
    assert '${RUN_PREPARED_DIR}/met/era5land/stack' in script_text


def test_sweb_runner_uses_script_dir_relative_workflow_entrypoints():
    script_text = (WORKFLOWS_DIR / "sweb_domain_runner.sh").read_text(encoding = "utf-8")

    assert '${SCRIPT_DIR}/3_sweb_preprocess_inputs.py' in script_text
    assert '${SCRIPT_DIR}/4_sweb_calib_domain.py' in script_text
    assert '${SCRIPT_DIR}/5_sweb_run_model.py' in script_text
    assert "CODE_DIR=" not in script_text
    assert "/code/workflows/" not in script_text
