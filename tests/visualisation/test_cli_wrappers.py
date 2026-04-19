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
import importlib.util
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
    ("wrapper_name", "wrapper_path", "package_module_name"),
    [
        (
            "legacy_plot_time_series",
            ROOT / "visualisation" / "plot_time_series.py",
            "pysweb.visualisation.plot_time_series",
        ),
        (
            "legacy_plot_heatmap",
            ROOT / "visualisation" / "plot_heatmap.py",
            "pysweb.visualisation.plot_heatmap",
        ),
    ],
)
def test_legacy_wrappers_load_and_expose_package_main(wrapper_name, wrapper_path, package_module_name):
    spec = importlib.util.spec_from_file_location(wrapper_name, wrapper_path)
    module = importlib.util.module_from_spec(spec)

    assert spec is not None
    assert spec.loader is not None

    spec.loader.exec_module(module)

    package_module = import_module(package_module_name)
    assert module.main is package_module.main


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
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert result.stderr == ""
