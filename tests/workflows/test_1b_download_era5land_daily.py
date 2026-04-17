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
from importlib import reload, util
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


def _load_workflow_module(monkeypatch):
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


def test_write_era5land_config_writes_expected_config_file(tmp_path):
    cfg_path = write_era5land_config(
        start_date="2024-01-01",
        end_date="2024-01-03",
        extent=[147.2, -35.1, 147.3, -35.0],
        output_dir=str(tmp_path / "raw"),
    )

    assert cfg_path.exists()
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert payload["collection"] == "ECMWF/ERA5_LAND/DAILY_AGGR"
    assert payload["download_dir"] == str(tmp_path / "raw")
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
        downloader_cls=FakeDownloader,
    )

    assert cfg_path.exists()
    assert recorded["init_paths"] == [str(cfg_path)]
    assert recorded["run_calls"] == 1


def test_pysweb_io_gee_wraps_legacy_downloader(monkeypatch):
    fake_core_module = types.ModuleType("core.gee_downloader")
    calls = {
        "init_paths": [],
        "mkdir_paths": [],
        "run_calls": 0,
    }

    class FakeLegacyDownloader:
        def __init__(self, config_path):
            calls["init_paths"].append(config_path)
            self.config_path = config_path

        def run(self):
            calls["run_calls"] += 1
            return "ok"

    def fake_safe_mkdir(path):
        calls["mkdir_paths"].append(path)

    fake_core_module.GEEDownloader = FakeLegacyDownloader
    fake_core_module._safe_mkdir = fake_safe_mkdir
    monkeypatch.setitem(sys.modules, "core.gee_downloader", fake_core_module)

    import pysweb.io.gee as gee_adapter

    gee_adapter = reload(gee_adapter)
    downloader = gee_adapter.GEEDownloader("config.yaml")

    assert downloader.config_path == "config.yaml"
    assert downloader.run() == "ok"
    gee_adapter._safe_mkdir("out")

    assert calls["init_paths"] == ["config.yaml"]
    assert calls["run_calls"] == 1
    assert calls["mkdir_paths"] == ["out"]


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
    monkeypatch.setattr(workflow_module, "_resolve_downloader_cls", lambda: FakeDownloader)
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
