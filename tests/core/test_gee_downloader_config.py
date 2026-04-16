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
import re
import sys
import types
from datetime import timedelta

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
    relativedelta_mod.relativedelta = lambda **kwargs: timedelta(days=kwargs.get("days", 0))
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


class _FakeBandNames:
    def __init__(self, names):
        self.names = list(names)

    def map(self, func):
        return [func(name) for name in self.names]


class _FakeImage:
    def __init__(self, band_names=None):
        self.band_names = list(band_names or [])
        self.rename_args = None
        self.get_download_url_args = None

    def bandNames(self):
        return _FakeBandNames(self.band_names)

    def rename(self, new_names):
        self.rename_args = list(new_names)
        return self

    def getDownloadURL(self, params):
        self.get_download_url_args = params
        return "fake-url"


class _FakeImageCollection:
    def __init__(self, images):
        self.images = list(images)
        self.calls = []
        self.reduced_reducer = None

    def filterDate(self, start, end):
        self.calls.append(("filterDate", start, end))
        return self

    def sort(self, key):
        self.calls.append(("sort", key))
        return self

    def filterBounds(self, region):
        self.calls.append(("filterBounds", region))
        return self

    def map(self, func):
        self.calls.append(("map",))
        self.images = [func(image) for image in self.images]
        return self

    def first(self):
        self.calls.append(("first",))
        return self.images[0]

    def reduce(self, reducer):
        self.calls.append(("reduce", reducer))
        self.reduced_reducer = reducer
        return self.images[0]

    def size(self):
        return types.SimpleNamespace(getInfo=lambda: len(self.images))

    def aggregate_array(self, name):
        return types.SimpleNamespace(getInfo=lambda: [])


class _FakeReducer:
    @staticmethod
    def median():
        return "median-reducer"


class _FakeEeString:
    def __init__(self, value):
        self.value = str(value)

    def replace(self, pattern, replacement):
        if pattern == "_median$" and replacement == "":
            return re.sub(r"_median$", "", self.value)
        return self.value.replace(pattern, replacement)


def test_daily_strategy_defaults_to_median(tmp_path):
    downloader = GEEDownloader(_write_config(tmp_path))
    assert downloader.cfg["daily_strategy"] == "median"


def test_daily_strategy_explicit_median(tmp_path):
    downloader = GEEDownloader(_write_config(tmp_path, daily_strategy="median"))
    assert downloader.cfg["daily_strategy"] == "median"


@pytest.mark.parametrize("daily_strategy", ["FIRST"])
def test_daily_strategy_accepts_first(tmp_path, daily_strategy):
    downloader = GEEDownloader(_write_config(tmp_path, daily_strategy=daily_strategy))
    assert downloader.cfg["daily_strategy"] == "first"


def test_daily_strategy_rejects_invalid_value(tmp_path):
    with pytest.raises(ValueError, match="daily_strategy"):
        GEEDownloader(_write_config(tmp_path, daily_strategy="bogus"))


def test_composite_for_day_median_reduces_and_renames(tmp_path, monkeypatch):
    downloader = GEEDownloader(_write_config(tmp_path, daily_strategy="median"))
    fake_image = _FakeImage(["band_1_median", "band_2_median"])
    fake_collection = _FakeImageCollection([fake_image])

    monkeypatch.setattr(downloader, "_is_globalish_extent", lambda: True)
    monkeypatch.setattr(sys.modules["core.gee_downloader"].ee, "ImageCollection", lambda collection: fake_collection)
    monkeypatch.setattr(sys.modules["core.gee_downloader"].ee, "Reducer", _FakeReducer)
    monkeypatch.setattr(sys.modules["core.gee_downloader"].ee, "Image", lambda image: image)
    monkeypatch.setattr(sys.modules["core.gee_downloader"].ee, "String", _FakeEeString)

    result = downloader._composite_for_day("2020-01-01", downloader.cfg["collection"])

    assert result is fake_image
    assert ("sort", "system:time_start") in fake_collection.calls
    assert ("reduce", "median-reducer") in fake_collection.calls
    assert result.rename_args == ["band_1", "band_2"]
    assert ("first",) not in fake_collection.calls


def test_composite_for_day_first_sorts_and_returns_first(tmp_path, monkeypatch):
    downloader = GEEDownloader(_write_config(tmp_path, daily_strategy="first"))
    first_image = _FakeImage(["band_1"])
    fake_collection = _FakeImageCollection([first_image, _FakeImage(["band_1"])])

    monkeypatch.setattr(downloader, "_is_globalish_extent", lambda: True)
    monkeypatch.setattr(sys.modules["core.gee_downloader"].ee, "ImageCollection", lambda collection: fake_collection)
    monkeypatch.setattr(sys.modules["core.gee_downloader"].ee, "Image", lambda image: image)

    result = downloader._composite_for_day("2020-01-01", downloader.cfg["collection"])

    assert result is first_image
    assert ("sort", "system:time_start") in fake_collection.calls
    assert ("first",) in fake_collection.calls
    assert not any(call[0] == "reduce" for call in fake_collection.calls)
