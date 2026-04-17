#!/usr/bin/env python3
"""
Script: test_api_run.py
Objective: Verify the SWB package run facade rejects empty calls and forwards meaningful inputs to the package workflow.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: Facade API calls and monkeypatched workflow functions supplied by pytest.
Outputs: Test assertions.
Usage: pytest tests/swb/test_api_run.py
Dependencies: pytest
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pysweb.swb.api as swb_api

run = swb_api.run


def test_swb_run_requires_meaningful_inputs():
    with pytest.raises(ValueError, match="Missing required inputs for SWB run"):
        run()


def test_swb_run_rejects_incidental_kwargs_without_entry_inputs(monkeypatch):
    dispatched = False

    def fake_run_swb_workflow(**kwargs):
        nonlocal dispatched
        dispatched = True

    monkeypatch.setattr(swb_api, "run_swb_workflow", fake_run_swb_workflow, raising=False)

    with pytest.raises(ValueError, match="Missing required inputs for SWB run"):
        run(workers = 2)

    assert dispatched is False


def test_swb_run_dispatches_to_package_workflow(monkeypatch):
    recorded = {}

    def fake_run_swb_workflow(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(swb_api, "run_swb_workflow", fake_run_swb_workflow, raising=False)

    run(
        precip = "/tmp/precip.nc",
        effective_precip = "/tmp/effective_precip.nc",
        et = "/tmp/et.nc",
        t = "/tmp/t.nc",
        soil_dir = "/tmp/soil",
        output_dir = "/tmp/out",
        workers = 2,
    )

    assert recorded == {
        "precip": "/tmp/precip.nc",
        "effective_precip": "/tmp/effective_precip.nc",
        "et": "/tmp/et.nc",
        "t": "/tmp/t.nc",
        "soil_dir": "/tmp/soil",
        "output_dir": "/tmp/out",
        "workers": 2,
    }
