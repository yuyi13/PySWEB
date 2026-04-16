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
from importlib import util
from pathlib import Path
import sys
import types

from core.era5land_download_config import build_era5land_cfg


EXPECTED_BANDS = [
    "temperature_2m_min",
    "temperature_2m_max",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_solar_radiation_downwards_sum",
    "total_precipitation_sum",
]


def _load_workflow_module(monkeypatch):
    fake_downloader_mod = types.ModuleType("core.gee_downloader")
    fake_downloader_mod.GEEDownloader = object
    monkeypatch.setitem(sys.modules, "core.gee_downloader", fake_downloader_mod)

    workflow_path = Path(__file__).resolve().parents[2] / "workflows" / "1b_download_era5land_daily.py"
    spec = util.spec_from_file_location("era5land_daily_workflow", workflow_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_era5land_cfg_sets_expected_contract(tmp_path):
    cfg = build_era5land_cfg(
        start_date="2024-01-01",
        end_date="2024-01-03",
        extent=[147.2, -35.1, 147.3, -35.0],
        out_dir=str(tmp_path / "raw"),
    )

    assert cfg["collection"] == "ECMWF/ERA5_LAND/DAILY_AGGR"
    assert cfg["coords"] == [147.2, -35.1, 147.3, -35.0]
    assert cfg["download_dir"] == str(tmp_path / "raw")
    assert cfg["start_year"] == 2024
    assert cfg["start_month"] == 1
    assert cfg["start_day"] == 1
    assert cfg["end_year"] == 2024
    assert cfg["end_month"] == 1
    assert cfg["end_day"] == 3
    assert cfg["bands"] == EXPECTED_BANDS
    assert cfg["scale"] == 11132
    assert cfg["crs"] == "EPSG:4326"
    assert cfg["out_format"] == "tif"
    assert cfg["auth_mode"] == "browser"
    assert cfg["filename_prefix"] == "ERA5LandDaily"
    assert cfg["daily_strategy"] == "first"
    assert cfg["postprocess"] == {
        "maskval_to_na": False,
        "enforce_float32": False,
    }


def test_build_era5land_cfg_rejects_reversed_date_range(tmp_path):
    try:
        build_era5land_cfg(
            start_date="2024-01-03",
            end_date="2024-01-01",
            extent=[147.2, -35.1, 147.3, -35.0],
            out_dir=str(tmp_path / "raw"),
        )
    except ValueError as exc:
        assert "start_date must be on or before end_date" in str(exc)
    else:
        raise AssertionError("Expected build_era5land_cfg to reject a reversed date range")


def test_workflow_writes_config_and_invokes_downloader(tmp_path, monkeypatch):
    recorded = {
        "init_paths": [],
        "run_calls": 0,
    }

    class FakeDownloader:
        def __init__(self, config_path):
            recorded["init_paths"].append(config_path)

        def run(self):
            recorded["run_calls"] += 1

    workflow_module = _load_workflow_module(monkeypatch)
    monkeypatch.setattr(workflow_module, "GEEDownloader", FakeDownloader)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "1b_download_era5land_daily.py",
            "--date-range",
            "2024-01-01 to 2024-01-03",
            "--extent",
            "147.2,-35.1,147.3,-35.0",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    workflow_module.main()

    cfg_path = tmp_path / "out" / "gee_config_era5land_2024-01-01_2024-01-03.yaml"
    assert cfg_path.exists()
    assert recorded["init_paths"] == [str(cfg_path)]
    assert recorded["run_calls"] == 1
