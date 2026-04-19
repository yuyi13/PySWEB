#!/usr/bin/env python3
"""
Script: test_api.py
Objective: Verify SWB package facade functions dispatch to the package-owned preprocess and calibration callables.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Package facade calls and monkeypatched dispatch targets exercised under pytest.
Outputs: Test assertions.
Usage: pytest tests/swb/test_api.py
Dependencies: pytest
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pysweb.swb.api as swb_api


def test_swb_preprocess_dispatches_to_package_module(monkeypatch):
    recorded = {}

    def fake_preprocess_inputs(**kwargs):
        recorded.update(kwargs)
        return "preprocess-ok"

    monkeypatch.setattr(swb_api, "preprocess_inputs", fake_preprocess_inputs, raising = False)

    result = swb_api.preprocess(output_dir = "/tmp/out", reference_source = "gssm1km")

    assert result == "preprocess-ok"
    assert recorded == {
        "output_dir": "/tmp/out",
        "reference_source": "gssm1km",
    }


def test_swb_calibrate_dispatches_to_package_module(monkeypatch):
    recorded = {}

    def fake_calibrate_domain(**kwargs):
        recorded.update(kwargs)
        return "calibrate-ok"

    monkeypatch.setattr(swb_api, "calibrate_domain", fake_calibrate_domain, raising = False)

    result = swb_api.calibrate(reference_ssm = "/tmp/reference.nc", output = "/tmp/params.csv")

    assert result == "calibrate-ok"
    assert recorded == {
        "reference_ssm": "/tmp/reference.nc",
        "output": "/tmp/params.csv",
    }
