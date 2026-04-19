#!/usr/bin/env python3
"""
Script: test_api.py
Objective: Verify the soil API contract and public backend dispatch behavior.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Soil API imports and backend dispatch behavior.
Outputs: Test assertions.
Usage: pytest tests/soil/test_api.py
Dependencies: pytest, xarray
"""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr


def test_supported_soil_sources_contract():
    api = import_module("pysweb.soil.api")

    assert api.SUPPORTED_SOIL_SOURCES == ("openlandmap", "mlcons", "slga", "custom")


def test_unknown_soil_source_fails_early():
    api = import_module("pysweb.soil.api")

    with pytest.raises(ValueError, match=r"Unsupported soil_source 'bogus'"):
        api.load_soil_properties(soil_source="bogus", args=None, grid=None)


def test_validate_soil_source_accepts_openlandmap():
    api = import_module("pysweb.soil.api")

    assert api.validate_soil_source("openlandmap") is None


def test_validate_soil_source_rejects_unknown_value():
    api = import_module("pysweb.soil.api")

    with pytest.raises(ValueError, match=r"Unsupported soil_source 'bogus'"):
        api.validate_soil_source("bogus")


@pytest.mark.parametrize("soil_source", ["mlcons", "slga", "custom"])
def test_validate_soil_source_rejects_placeholder_backends(soil_source):
    api = import_module("pysweb.soil.api")

    with pytest.raises(NotImplementedError, match=rf"'{soil_source}'"):
        api.validate_soil_source(soil_source)


def test_openlandmap_dispatches_via_public_api(monkeypatch):
    api = import_module("pysweb.soil.api")
    openlandmap = import_module("pysweb.soil.openlandmap")
    expected = api.SoilOutputs(
        arrays={
            "porosity": xr.DataArray(
                np.ones((1, 1, 1), dtype=np.float32),
                dims=("layer", "lat", "lon"),
                coords={"layer": [1], "lat": [-35.0], "lon": [148.0]},
                name="porosity",
            )
        },
        layer_bottoms_mm=np.array([50.0], dtype=float),
    )
    recorded = {}

    def fake_load_soil_properties(*, args, grid, **kwargs):
        recorded["args"] = args
        recorded["grid"] = grid
        recorded["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(openlandmap, "load_soil_properties", fake_load_soil_properties)

    args = SimpleNamespace(extent=[148.0, -35.1, 148.1, -35.0], gee_project="yiyu-research")
    grid = SimpleNamespace(lat_dim="lat", lon_dim="lon")

    actual = api.load_soil_properties(
        soil_source="openlandmap",
        args=args,
        grid=grid,
        reproject_to_template="callback",
    )

    assert actual is expected
    assert recorded == {
        "args": args,
        "grid": grid,
        "kwargs": {"reproject_to_template": "callback"},
    }


@pytest.mark.parametrize("soil_source", ["mlcons", "slga", "custom"])
def test_placeholder_backends_fail_via_public_api(soil_source):
    api = import_module("pysweb.soil.api")

    with pytest.raises(NotImplementedError, match=rf"'{soil_source}'"):
        api.load_soil_properties(soil_source=soil_source, args=None, grid=None)
