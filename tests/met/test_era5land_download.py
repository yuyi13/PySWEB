#!/usr/bin/env python3
"""
Script: test_era5land_download.py
Objective: Verify the package ERA5-Land daily download config builder produces the expected GEE settings.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-05-03
Inputs: Temporary paths and pure config-builder inputs supplied by pytest.
Outputs: Test assertions.
Usage: pytest tests/met/test_era5land_download.py
Dependencies: pytest
"""
from importlib import reload
from pathlib import Path
import sys
import json
import types

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysweb.met.era5land.download import (
    build_era5land_cfg,
    download_era5land_daily,
    write_era5land_config,
)


EXPECTED_BANDS = [
    "temperature_2m_min",
    "temperature_2m_max",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_solar_radiation_downwards_sum",
    "total_precipitation_sum",
]


def test_build_era5land_cfg_sets_expected_contract(tmp_path):
    cfg = build_era5land_cfg(
        start_date="2024-01-01",
        end_date="2024-01-03",
        extent=[147.2, -35.1, 147.3, -35.0],
        out_dir=str(tmp_path / "raw"),
        gee_project="workflow-project",
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
    assert cfg["gee_project"] == "workflow-project"
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
            gee_project="workflow-project",
        )
    except ValueError as exc:
        assert "start_date must be on or before end_date" in str(exc)
    else:
        raise AssertionError("Expected build_era5land_cfg to reject a reversed date range")


def test_build_era5land_cfg_requires_explicit_gee_project(tmp_path):
    try:
        build_era5land_cfg(
            start_date="2024-01-01",
            end_date="2024-01-03",
            extent=[147.2, -35.1, 147.3, -35.0],
            out_dir=str(tmp_path / "raw"),
        )
    except TypeError as exc:
        assert "gee_project" in str(exc)
    else:
        raise AssertionError("Expected build_era5land_cfg to require gee_project explicitly")


def test_build_era5land_cfg_rejects_blank_gee_project(tmp_path):
    try:
        build_era5land_cfg(
            start_date="2024-01-01",
            end_date="2024-01-03",
            extent=[147.2, -35.1, 147.3, -35.0],
            out_dir=str(tmp_path / "raw"),
            gee_project="   ",
        )
    except ValueError as exc:
        assert "gee_project" in str(exc)
    else:
        raise AssertionError("Expected build_era5land_cfg to reject blank gee_project values")


def test_write_era5land_config_writes_expected_config_file(tmp_path):
    cfg_path = write_era5land_config(
        start_date="2024-01-01",
        end_date="2024-01-03",
        extent=[147.2, -35.1, 147.3, -35.0],
        output_dir=str(tmp_path / "raw"),
        gee_project="workflow-project",
    )

    assert cfg_path.exists()
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert payload["collection"] == "ECMWF/ERA5_LAND/DAILY_AGGR"
    assert payload["download_dir"] == str(tmp_path / "raw")
    assert payload["gee_project"] == "workflow-project"
    assert payload["start_day"] == 1
    assert payload["end_day"] == 3


def test_download_era5land_daily_uses_injected_downloader(tmp_path):
    recorded = {
        "init_paths": [],
        "run_calls": 0,
    }

    class FakeDownloader:
        def __init__(self, config_path):
            recorded["init_paths"].append(config_path)

        def run(self):
            recorded["run_calls"] += 1

    cfg_path = download_era5land_daily(
        start_date="2024-01-01",
        end_date="2024-01-03",
        extent=[147.2, -35.1, 147.3, -35.0],
        output_dir=str(tmp_path / "out"),
        gee_project="workflow-project",
        downloader_cls=FakeDownloader,
    )

    assert cfg_path.exists()
    assert recorded["init_paths"] == [str(cfg_path)]
    assert recorded["run_calls"] == 1
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert payload["gee_project"] == "workflow-project"


def test_pysweb_io_gee_exports_package_downloader():
    import pysweb.io.gee as gee_adapter
    import pysweb.io.gee_downloader as gee_downloader_module

    gee_adapter = reload(gee_adapter)

    assert gee_adapter.GEEDownloader is gee_downloader_module.GEEDownloader
    assert gee_adapter._safe_mkdir is gee_downloader_module._safe_mkdir


def test_package_downloader_browser_mode_uses_configured_gee_project(monkeypatch, tmp_path):
    fake_yaml = types.SimpleNamespace(
        safe_load=lambda payload: json.loads(payload.read() if hasattr(payload, "read") else payload)
    )
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)
    import pysweb.io.gee_downloader as gee_downloader_module

    gee_downloader_module = reload(gee_downloader_module)

    cfg_path = tmp_path / "gee_config.yaml"
    cfg_path.write_text(
        json.dumps(
            {
                "collection": "ECMWF/ERA5_LAND/DAILY_AGGR",
                "coords": [147.2, -35.1, 147.3, -35.0],
                "download_dir": str(tmp_path / "out"),
                "start_year": 2024,
                "start_month": 1,
                "start_day": 1,
                "end_year": 2024,
                "end_month": 1,
                "end_day": 3,
                "bands": EXPECTED_BANDS,
                "scale": 11132,
                "out_format": "tif",
                "auth_mode": "browser",
                "gee_project": "configured-ee-project",
                "filename_prefix": "ERA5LandDaily",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    calls = []

    class FakeEE:
        @staticmethod
        def Initialize(*args, **kwargs):
            calls.append(("Initialize", args, kwargs))

        @staticmethod
        def Authenticate():
            calls.append(("Authenticate", (), {}))

    monkeypatch.setattr(gee_downloader_module, "ee", FakeEE)

    downloader = gee_downloader_module.GEEDownloader(str(cfg_path))
    downloader.initialize()

    assert calls == [("Initialize", (), {"project": "configured-ee-project"})]


def test_package_downloader_browser_mode_requires_configured_gee_project(monkeypatch, tmp_path):
    fake_yaml = types.SimpleNamespace(
        safe_load=lambda payload: json.loads(payload.read() if hasattr(payload, "read") else payload)
    )
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)
    import pysweb.io.gee_downloader as gee_downloader_module

    gee_downloader_module = reload(gee_downloader_module)

    cfg_path = tmp_path / "gee_config.yaml"
    cfg_path.write_text(
        json.dumps(
            {
                "collection": "ECMWF/ERA5_LAND/DAILY_AGGR",
                "coords": [147.2, -35.1, 147.3, -35.0],
                "download_dir": str(tmp_path / "out"),
                "start_year": 2024,
                "start_month": 1,
                "start_day": 1,
                "end_year": 2024,
                "end_month": 1,
                "end_day": 3,
                "bands": EXPECTED_BANDS,
                "scale": 11132,
                "out_format": "tif",
                "auth_mode": "browser",
                "filename_prefix": "ERA5LandDaily",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    downloader = gee_downloader_module.GEEDownloader(str(cfg_path))

    try:
        downloader.initialize()
    except ValueError as exc:
        assert "gee_project" in str(exc)
    else:
        raise AssertionError("Expected browser mode to require an explicit gee_project")
