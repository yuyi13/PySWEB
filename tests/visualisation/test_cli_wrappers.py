#!/usr/bin/env python3
"""
Script: test_cli_wrappers.py
Objective: Verify the migrated visualisation modules and legacy CLI wrappers import and expose help cleanly.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Module imports, wrapper source text, and subprocess CLI help invocations.
Outputs: Test assertions.
Usage: pytest tests/visualisation/test_cli_wrappers.py
Dependencies: pytest, subprocess
"""
from importlib import import_module
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "module_name",
    [
        "pysweb.visualisation.plot_time_series",
        "pysweb.visualisation.plot_heatmap",
    ],
)
def test_package_visualisation_modules_import_cleanly(module_name):
    assert import_module(module_name).__name__ == module_name


@pytest.mark.parametrize(
    ("wrapper_path", "expected_import"),
    [
        (
            ROOT / "visualisation" / "plot_time_series.py",
            "from pysweb.visualisation.plot_time_series import main",
        ),
        (
            ROOT / "visualisation" / "plot_heatmap.py",
            "from pysweb.visualisation.plot_heatmap import main",
        ),
    ],
)
def test_legacy_wrappers_import_main_from_package(wrapper_path, expected_import):
    assert expected_import in wrapper_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, "-m", "pysweb.visualisation.plot_time_series", "--help"],
        [sys.executable, "-m", "pysweb.visualisation.plot_heatmap", "--help"],
        [sys.executable, str(ROOT / "visualisation" / "plot_time_series.py"), "--help"],
        [sys.executable, str(ROOT / "visualisation" / "plot_heatmap.py"), "--help"],
    ],
)
def test_visualisation_entrypoints_exit_cleanly_for_help(command):
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not python_path else f"{ROOT}:{python_path}"

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert result.stderr == ""
