#!/usr/bin/env python3
"""
Script: test_api.py
Objective: Verify the DEM API validates its public contract and dispatches current NASADEM requests correctly.
Author: Yi Yu
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: DEM API imports and dispatch behavior.
Outputs: Test assertions.
Usage: pytest tests/dem/test_api.py
Dependencies: pytest
"""

from importlib import import_module

import pytest


def test_supported_dem_sources_contract():
    api = import_module("pysweb.dem.api")

    assert api.SUPPORTED_DEM_SOURCES == ("nasadem",)


def test_prepare_dem_requires_gee_project():
    api = import_module("pysweb.dem.api")

    with pytest.raises(TypeError, match="gee_project"):
        api.prepare_dem(dem_source="nasadem")


def test_unknown_dem_source_fails_early():
    api = import_module("pysweb.dem.api")

    with pytest.raises(ValueError, match=r"Unsupported dem_source 'bogus'"):
        api.prepare_dem(dem_source="bogus", gee_project="test-project")


def test_public_prepare_dem_dispatches_to_backend_with_current_contract(monkeypatch):
    api = import_module("pysweb.dem.api")
    recorded = {}

    class FakeBackendModule:
        @staticmethod
        def prepare_dem(*, gee_project, extent, output_path):
            recorded["gee_project"] = gee_project
            recorded["extent"] = extent
            recorded["output_path"] = output_path
            return output_path

    monkeypatch.setattr(api, "import_module", lambda module_name: FakeBackendModule)

    result = api.prepare_dem(
        dem_source = "nasadem",
        gee_project = "test-project",
        extent = [147.2, -35.1, 147.3, -35.0],
        output_path = "/tmp/nasadem.tif",
    )

    assert result == "/tmp/nasadem.tif"
    assert recorded == {
        "gee_project": "test-project",
        "extent": [147.2, -35.1, 147.3, -35.0],
        "output_path": "/tmp/nasadem.tif",
    }


def test_prepare_dem_rejects_empty_gee_project():
    api = import_module("pysweb.dem.api")

    with pytest.raises(ValueError, match="gee_project must be a non-empty string"):
        api.prepare_dem(dem_source="nasadem", gee_project="")


def test_prepare_dem_rejects_non_string_gee_project():
    api = import_module("pysweb.dem.api")

    with pytest.raises(ValueError, match="gee_project must be a non-empty string"):
        api.prepare_dem(dem_source="nasadem", gee_project=123)


def test_prepare_dem_rejects_whitespace_only_gee_project():
    api = import_module("pysweb.dem.api")

    with pytest.raises(ValueError, match="gee_project must be a non-empty string"):
        api.prepare_dem(dem_source="nasadem", gee_project="   ")
