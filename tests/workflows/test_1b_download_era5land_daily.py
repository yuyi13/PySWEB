#!/usr/bin/env python3
"""
Script: test_1b_download_era5land_daily.py
Objective: Verify the ERA5-Land daily download config builder produces the expected GEE settings.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-04-16
Inputs: Temporary paths and pure config-builder inputs supplied by pytest.
Outputs: Test assertions.
Usage: pytest tests/workflows/test_1b_download_era5land_daily.py
Dependencies: pytest
"""
from core.era5land_download_config import build_era5land_cfg


def test_build_era5land_cfg_sets_expected_collection_and_bands(tmp_path):
    cfg = build_era5land_cfg(
        start_date="2024-01-01",
        end_date="2024-01-03",
        extent=[147.2, -35.1, 147.3, -35.0],
        out_dir=str(tmp_path / "raw"),
    )

    assert cfg["collection"] == "ECMWF/ERA5_LAND/DAILY_AGGR"
    assert cfg["daily_strategy"] == "first"
    assert cfg["bands"] == [
        "temperature_2m_min",
        "temperature_2m_max",
        "dewpoint_temperature_2m",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "surface_solar_radiation_downwards_sum",
        "total_precipitation_sum",
    ]
