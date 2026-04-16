#!/usr/bin/env python3
"""
Script: test_era5land_refet.py
Objective: Verify ERA5-Land reference ET math utilities and daily file discovery behave as expected.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-04-16
Inputs: Pytest fixtures and synthetic temporary files.
Outputs: Test assertions.
Usage: pytest tests/core/test_era5land_refet.py
Dependencies: numpy, pytest
"""
from pathlib import Path

import numpy as np
import pytest

from core.era5land_refet import (
    actual_vapor_pressure_from_dewpoint_c,
    compute_daily_eto_short,
    j_per_m2_to_mj_per_m2_day,
    kelvin_to_celsius,
    meters_to_mm_day,
    wind_speed_from_uv,
)
from core.era5land_stack import discover_daily_files


def test_kelvin_to_celsius_converts_expected_values():
    result = kelvin_to_celsius([273.15, 300.15])
    np.testing.assert_allclose(result, [0.0, 27.0])


def test_j_per_m2_to_mj_per_m2_day_converts_expected_value():
    result = j_per_m2_to_mj_per_m2_day(8640000.0)
    assert result == pytest.approx(8.64)


def test_meters_to_mm_day_converts_expected_value():
    result = meters_to_mm_day(0.012)
    assert result == pytest.approx(12.0)


def test_actual_vapor_pressure_from_dewpoint_c_matches_reference_value():
    result = actual_vapor_pressure_from_dewpoint_c([20.0])
    np.testing.assert_allclose(result, [2.338], atol=0.002)


def test_wind_speed_from_uv_returns_vector_magnitude():
    result = wind_speed_from_uv([3.0], [4.0])
    np.testing.assert_allclose(result, [5.0])


def test_compute_daily_eto_short_matches_reference_case():
    result = compute_daily_eto_short(
        tmax_c=[31.0],
        tmin_c=[16.0],
        ea_kpa=[1.90],
        rs_mj_m2_day=[24.0],
        uz_m_s=[3.2],
        zw_m=10.0,
        elev_m=[180.0],
        lat_deg=[-35.0],
        doy=[15],
    )

    np.testing.assert_allclose(result, [6.0], atol=0.5)


def test_discover_daily_files_sorts_by_embedded_date(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name in [
        "ERA5LandDaily_2024-01-03.tif",
        "ERA5LandDaily_2024-01-01.tif",
        "ERA5LandDaily_2024-01-02.tif",
    ]:
        (raw_dir / name).write_bytes(b"")

    files = discover_daily_files(raw_dir)

    assert files == [
        Path(raw_dir / "ERA5LandDaily_2024-01-01.tif"),
        Path(raw_dir / "ERA5LandDaily_2024-01-02.tif"),
        Path(raw_dir / "ERA5LandDaily_2024-01-03.tif"),
    ]
