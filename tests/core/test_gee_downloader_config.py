#!/usr/bin/env python3
"""
Script: test_gee_downloader_config.py
Objective: Verify GEEDownloader normalizes daily_strategy during config validation.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-04-16
Inputs: Temporary YAML configs written during pytest execution.
Outputs: Test assertions.
Usage: pytest tests/core/test_gee_downloader_config.py
Dependencies: pytest
"""
from pathlib import Path
import sys
import types

import pytest


def _safe_load_config(text):
    if hasattr(text, "read"):
        text = text.read()

    def _parse_scalar(value):
        value = value.strip()
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    result = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            result[key] = None
        elif value.startswith("[") and value.endswith("]"):
            items = [item.strip() for item in value[1:-1].split(",") if item.strip()]
            result[key] = [_parse_scalar(item) for item in items]
        else:
            result[key] = _parse_scalar(value)
    return result


def _install_dependency_shims():
    ee = types.ModuleType("ee")
    ee.Image = type("Image", (), {})
    ee.ImageCollection = type("ImageCollection", (), {})
    ee.Geometry = type("Geometry", (), {})
    ee.Reducer = type("Reducer", (), {})
    ee.Algorithms = type("Algorithms", (), {})
    ee.Number = type("Number", (), {})
    ee.String = type("String", (), {})
    ee.Initialize = lambda *args, **kwargs: None
    ee.Authenticate = lambda *args, **kwargs: None
    ee.ServiceAccountCredentials = type("ServiceAccountCredentials", (), {})
    sys.modules.setdefault("ee", ee)

    numpy_mod = types.ModuleType("numpy")
    numpy_mod.linspace = lambda start, stop, num: [start, stop]
    numpy_mod.cos = lambda value: 1.0
    numpy_mod.radians = lambda value: value
    numpy_mod.ceil = lambda value: value
    numpy_mod.floor = lambda value: value
    numpy_mod.float32 = float
    numpy_mod.nan = float("nan")
    numpy_mod.array = lambda value, **kwargs: value
    sys.modules.setdefault("numpy", numpy_mod)

    rasterio_mod = types.ModuleType("rasterio")
    rasterio_mod.Env = type("Env", (), {"__enter__": lambda self: self, "__exit__": lambda self, exc_type, exc, tb: False})
    rasterio_mod.open = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("rasterio open shim should not be used"))
    sys.modules.setdefault("rasterio", rasterio_mod)

    rasterio_merge_mod = types.ModuleType("rasterio.merge")
    rasterio_merge_mod.merge = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("rasterio merge shim should not be used"))
    sys.modules.setdefault("rasterio.merge", rasterio_merge_mod)

    requests_mod = types.ModuleType("requests")
    requests_mod.get = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("requests shim should not be used"))
    sys.modules.setdefault("requests", requests_mod)

    yaml_mod = types.ModuleType("yaml")
    yaml_mod.safe_load = _safe_load_config
    sys.modules.setdefault("yaml", yaml_mod)

    dateutil_mod = types.ModuleType("dateutil")
    relativedelta_mod = types.ModuleType("dateutil.relativedelta")
    relativedelta_mod.relativedelta = type("relativedelta", (), {})
    sys.modules.setdefault("dateutil", dateutil_mod)
    sys.modules.setdefault("dateutil.relativedelta", relativedelta_mod)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


_install_dependency_shims()

from core.gee_downloader import GEEDownloader


def _write_config(tmp_path: Path, daily_strategy=None):
    config_lines = [
        "coords: [0.0, 0.0, 1.0, 1.0]",
        f"download_dir: {tmp_path / 'downloads'}",
        "start_year: 2020",
        "start_month: 1",
        "start_day: 1",
        "end_year: 2020",
        "end_month: 1",
        "end_day: 2",
        "bands: [band_1]",
        "scale: 1000",
        "out_format: tif",
        "auth_mode: browser",
        "filename_prefix: test",
        "collection: ECMWF/ERA5_LAND/DAILY_AGGR",
    ]
    if daily_strategy is not None:
        config_lines.append(f"daily_strategy: {daily_strategy}")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    return config_path


def test_daily_strategy_defaults_to_median(tmp_path):
    downloader = GEEDownloader(_write_config(tmp_path))
    assert downloader.cfg["daily_strategy"] == "median"


@pytest.mark.parametrize("daily_strategy", ["FIRST"])
def test_daily_strategy_accepts_first(tmp_path, daily_strategy):
    downloader = GEEDownloader(_write_config(tmp_path, daily_strategy=daily_strategy))
    assert downloader.cfg["daily_strategy"] == "first"
