#!/usr/bin/env python3
"""
Script: test_api.py
Objective: Verify the soil API contract and placeholder backend failures.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Soil API imports and backend dispatch behavior.
Outputs: Test assertions.
Usage: pytest tests/soil/test_api.py
Dependencies: pytest
"""

from importlib import import_module

import pytest


def test_supported_soil_sources_contract():
    api = import_module("pysweb.soil.api")

    assert api.SUPPORTED_SOIL_SOURCES == ("openlandmap", "mlcons", "slga", "custom")


def test_unknown_soil_source_fails_early():
    api = import_module("pysweb.soil.api")

    with pytest.raises(ValueError, match=r"Unsupported soil_source 'bogus'"):
        api.load_soil_properties(soil_source="bogus", args=None, grid=None)


@pytest.mark.parametrize("soil_source", ["mlcons", "slga", "custom"])
def test_placeholder_backends_fail_with_named_not_implemented_error(soil_source):
    backend = import_module(f"pysweb.soil.{soil_source}")

    with pytest.raises(NotImplementedError, match=soil_source):
        backend.load_soil_properties(args=None, grid=None)
