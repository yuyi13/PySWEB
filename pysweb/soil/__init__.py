#!/usr/bin/env python3
"""
Script: __init__.py
Objective: Expose lazy soil package entry points and the package-level soil loader.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Package attribute access and submodule imports.
Outputs: Lazy access to soil backends and the soil loader facade.
Usage: Imported as `pysweb.soil`
Dependencies: importlib
"""

from importlib import import_module

__all__ = [
    "api",
    "custom",
    "load_soil_properties",
    "mlcons",
    "openlandmap",
    "slga",
]

_SUBMODULES = {"api", "custom", "mlcons", "openlandmap", "slga"}


def __getattr__(name):
    if name in _SUBMODULES:
        module = import_module(f"pysweb.soil.{name}")
        globals()[name] = module
        return module
    if name == "load_soil_properties":
        api = import_module("pysweb.soil.api")
        value = api.load_soil_properties
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pysweb.soil' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
