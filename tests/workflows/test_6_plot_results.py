#!/usr/bin/env python3
"""
Script: test_6_plot_results.py
Objective: Verify the Step 6 plotting workflow dispatches to package visualisation entrypoints.
Author: Yi Yu
Created: 2026-05-03
Last updated: 2026-05-03
Inputs: Workflow script imports and subprocess CLI help invocations.
Outputs: Test assertions.
Usage: pytest tests/workflows/test_6_plot_results.py
Dependencies: pytest, subprocess
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "workflows" / "6_plot_results.py"


def load_workflow_module():
    spec = importlib.util.spec_from_file_location("workflow_6_plot_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)

    assert spec is not None
    assert spec.loader is not None

    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, str(SCRIPT), "--help"],
        [sys.executable, str(SCRIPT), "heatmap", "--help"],
        [sys.executable, str(SCRIPT), "time-series", "--help"],
    ],
)
def test_plot_results_help_exits_cleanly(command):
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


def test_plot_results_forwards_heatmap_args(monkeypatch):
    module = load_workflow_module()
    calls = []

    monkeypatch.setattr(module.plot_heatmap, "main", lambda argv=None: calls.append(("heatmap", argv)))

    assert module.main(["heatmap", "--domain-mean", "--output", "plot.png"]) == 0
    assert calls == [("heatmap", ["--domain-mean", "--output", "plot.png"])]


def test_plot_results_forwards_time_series_args(monkeypatch):
    module = load_workflow_module()
    calls = []

    monkeypatch.setattr(module.plot_time_series, "main", lambda argv=None: calls.append(("time-series", argv)))

    assert module.main(["time-series", "--output", "plot.png"]) == 0
    assert calls == [("time-series", ["--output", "plot.png"])]

