#!/usr/bin/env python3
"""
Script: test_api_prepare_inputs.py
Objective: Verify the SSEBop package API orchestrates Landsat and meteorological input preparation steps.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: Temporary paths and monkeypatched package functions supplied by pytest.
Outputs: Test assertions.
Usage: pytest tests/ssebop/test_api_prepare_inputs.py
Dependencies: pytest
"""
from pathlib import Path

from pysweb.ssebop.api import prepare_inputs


def test_prepare_inputs_calls_landsat_and_era5land_steps(monkeypatch, tmp_path: Path):
    recorded = []

    monkeypatch.setattr(
        "pysweb.ssebop.inputs.landsat.prepare_landsat_inputs",
        lambda **kwargs: recorded.append(("landsat", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.met.era5land.download.download_era5land_daily",
        lambda **kwargs: recorded.append(("era5land_download", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.met.era5land.stack.stack_era5land_daily_inputs",
        lambda **kwargs: recorded.append(("era5land_stack", kwargs)),
    )

    prepare_inputs(
        date_range="2024-01-01 to 2024-01-03",
        extent=[147.2, -35.1, 147.3, -35.0],
        met_source="era5land",
        landsat_dir=str(tmp_path / "landsat"),
        met_raw_dir=str(tmp_path / "raw"),
        met_stack_dir=str(tmp_path / "stack"),
        dem=str(tmp_path / "dem.tif"),
        gee_config="/tmp/gee.yaml",
    )

    assert [name for name, _ in recorded] == ["landsat", "era5land_download", "era5land_stack"]
