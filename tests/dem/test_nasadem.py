#!/usr/bin/env python3
"""
Script: test_nasadem.py
Objective: Verify NASADEM DEM preparation dispatches through the package API and downloads a non-empty GeoTIFF via Earth Engine.
Author: Yi Yu
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: Pytest monkeypatch fixtures, temporary output paths, and backend stubs.
Outputs: Test assertions.
Usage: python -m pytest tests/dem/test_nasadem.py -q
Dependencies: pathlib, sys, pytest
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysweb.dem import api as dem_api
from pysweb.dem import nasadem


def test_prepare_dem_dispatches_to_nasadem_backend(monkeypatch):
    recorded = {}

    def fake_backend_prepare_dem(*, gee_project: str, extent: list[float], output_path: str) -> str:
        recorded["gee_project"] = gee_project
        recorded["extent"] = extent
        recorded["output_path"] = output_path
        return output_path

    monkeypatch.setattr(nasadem, "prepare_dem", fake_backend_prepare_dem)

    result = dem_api.prepare_dem(
        dem_source = "nasadem",
        gee_project = "demo-project",
        extent = [147.2, -35.1, 147.3, -35.0],
        output_path = "/tmp/nasadem.tif",
    )

    assert result == "/tmp/nasadem.tif"
    assert recorded == {
        "gee_project": "demo-project",
        "extent": [147.2, -35.1, 147.3, -35.0],
        "output_path": "/tmp/nasadem.tif",
    }


def test_prepare_dem_downloads_clipped_nasadem_geotiff(monkeypatch, tmp_path: Path):
    class FakeImage:
        def __init__(self):
            self.selected_band = None
            self.clip_region = None
            self.download_params = None

        def select(self, band_name: str):
            self.selected_band = band_name
            return self

        def clip(self, region):
            self.clip_region = region
            return self

        def getDownloadURL(self, params):
            self.download_params = params
            return "https://example.invalid/nasadem.tif"

    fake_image = FakeImage()
    ee_calls = {}

    class FakeGeometry:
        @staticmethod
        def Rectangle(coords, proj = None, geodesic = None):
            ee_calls["rectangle"] = {
                "coords": coords,
                "proj": proj,
                "geodesic": geodesic,
            }
            return {"type": "Rectangle", "coords": coords}

    class FakeEE:
        Geometry = FakeGeometry

        @staticmethod
        def Initialize(project = None):
            ee_calls["initialize_project"] = project

        @staticmethod
        def Image(image_id: str):
            ee_calls["image_id"] = image_id
            return fake_image

    response_calls = {}

    class FakeResponse:
        def __init__(self, content: bytes):
            self.content = content

        def raise_for_status(self):
            response_calls["raise_for_status"] = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_requests_get(url: str, timeout: int):
        response_calls["url"] = url
        response_calls["timeout"] = timeout
        return FakeResponse(b"fake-geotiff-bytes")

    monkeypatch.setattr(nasadem, "ee", FakeEE)
    monkeypatch.setattr(nasadem.requests, "get", fake_requests_get)

    output_path = tmp_path / "nested" / "nasadem.tif"
    result = nasadem.prepare_dem(
        gee_project = "demo-project",
        extent = [147.2, -35.1, 147.3, -35.0],
        output_path = str(output_path),
    )

    assert result == str(output_path)
    assert output_path.read_bytes() == b"fake-geotiff-bytes"
    assert ee_calls == {
        "initialize_project": "demo-project",
        "image_id": "NASA/NASADEM_HGT/001",
        "rectangle": {
            "coords": [147.2, -35.1, 147.3, -35.0],
            "proj": "EPSG:4326",
            "geodesic": False,
        },
    }
    assert fake_image.selected_band == "elevation"
    assert fake_image.clip_region == {"type": "Rectangle", "coords": [147.2, -35.1, 147.3, -35.0]}
    assert fake_image.download_params == {
        "name": "nasadem",
        "region": {"type": "Rectangle", "coords": [147.2, -35.1, 147.3, -35.0]},
        "crs": "EPSG:4326",
        "format": "GEO_TIFF",
        "filePerBand": False,
    }
    assert response_calls == {
        "url": "https://example.invalid/nasadem.tif",
        "timeout": 300,
        "raise_for_status": True,
    }


def test_prepare_dem_rejects_empty_download(monkeypatch, tmp_path: Path):
    class FakeImage:
        def select(self, band_name: str):
            return self

        def clip(self, region):
            return self

        def getDownloadURL(self, params):
            return "https://example.invalid/nasadem-empty.tif"

    class FakeGeometry:
        @staticmethod
        def Rectangle(coords, proj = None, geodesic = None):
            return {"type": "Rectangle", "coords": coords}

    class FakeEE:
        Geometry = FakeGeometry

        @staticmethod
        def Initialize(project = None):
            return None

        @staticmethod
        def Image(image_id: str):
            return FakeImage()

    class FakeResponse:
        content = b""

        def raise_for_status(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(nasadem, "ee", FakeEE)
    monkeypatch.setattr(nasadem.requests, "get", lambda url, timeout: FakeResponse())

    with pytest.raises(RuntimeError, match = "empty"):
        nasadem.prepare_dem(
            gee_project = "demo-project",
            extent = [147.2, -35.1, 147.3, -35.0],
            output_path = str(tmp_path / "nasadem.tif"),
        )
