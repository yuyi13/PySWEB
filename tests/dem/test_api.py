#!/usr/bin/env python3
"""
Script: test_api.py
Objective: Verify the DEM API contract and placeholder backend dispatch behavior.
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


def test_public_prepare_dem_dispatches_to_placeholder_backend():
    api = import_module("pysweb.dem.api")

    with pytest.raises(NotImplementedError, match="NASADEM"):
        api.prepare_dem(
            dem_source="nasadem",
            gee_project="test-project",
            output_dir="/tmp/dem",
        )


def test_prepare_dem_rejects_empty_gee_project():
    api = import_module("pysweb.dem.api")

    with pytest.raises(ValueError, match="gee_project must be a non-empty string"):
        api.prepare_dem(dem_source="nasadem", gee_project="")


def test_nasadem_backend_is_a_stub_for_now():
    nasadem = import_module("pysweb.dem.nasadem")

    with pytest.raises(NotImplementedError, match="NASADEM"):
        nasadem.prepare_dem(gee_project="test-project")
