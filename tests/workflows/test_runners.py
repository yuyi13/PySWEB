#!/usr/bin/env python3
"""
Script: test_runners.py
Objective: Verify the workflow shell wrappers stay thin and point at the package-backed entrypoints.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-19
Inputs: Workflow shell scripts and subprocess invocations supplied by pytest.
Outputs: Test assertions.
Usage: pytest tests/workflows/test_runners.py
Dependencies: pathlib, re, subprocess
"""
from pathlib import Path
import re
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


def _extract_assignment(script_text: str, variable_name: str) -> str:
    match = re.search(rf'^{variable_name}="([^"]+)"', script_text, flags = re.MULTILINE)
    assert match is not None, f"Missing assignment for {variable_name}"
    return match.group(1)


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


def test_wrapper_handoff_prefers_prepared_precip_stack_contract():
    ssebop_text = (WORKFLOWS_DIR / "ssebop_runner_landsat.sh").read_text(encoding = "utf-8")
    sweb_text = (WORKFLOWS_DIR / "sweb_domain_runner.sh").read_text(encoding = "utf-8")

    prepared_dir_base = _extract_assignment(ssebop_text, "PREPARED_DIR_BASE")
    assert prepared_dir_base == "${PROJECT_DIR}/1_ssebop_inputs"
    assert 'RUN_MET_STACK_DIR="${RUN_PREPARED_DIR}/met/era5land/stack"' in ssebop_text

    assert "1_ssebop_inputs" in sweb_text
    assert "met/era5land/stack" in sweb_text
    assert "1_era5land_stacks" in sweb_text
    assert sweb_text.index("1_ssebop_inputs") < sweb_text.index("1_era5land_stacks")


def test_sweb_runner_uses_reference_ssm_filename_contract():
    sweb_text = (WORKFLOWS_DIR / "sweb_domain_runner.sh").read_text(encoding = "utf-8")

    assert "reference_ssm_daily_" in sweb_text
    assert "smap_ssm_daily_" not in sweb_text
    assert '--reference-ssm "${CALIB_REFERENCE_SSM_FILE}"' in sweb_text
    assert '--reference-ssm "${CALIB_SMAP_SSM_FILE}"' not in sweb_text


def test_sweb_runner_does_not_pass_legacy_soil_cli_flags():
    sweb_text = (WORKFLOWS_DIR / "sweb_domain_runner.sh").read_text(encoding = "utf-8")

    assert "--soil-texture-dir" not in sweb_text
    assert "--soil-soc-dir" not in sweb_text
