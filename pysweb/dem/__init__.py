#!/usr/bin/env python3
"""
Script: __init__.py
Objective: Expose lazy DEM package entry points and the package-level DEM preparer.
Author: Yi Yu
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: Package attribute access and submodule imports.
Outputs: Lazy access to DEM backends and the DEM preparer facade.
Usage: Imported as `pysweb.dem`
Dependencies: importlib
"""

from importlib import import_module

__all__ = ["api", "nasadem", "prepare_dem"]


def __getattr__(name):
    if name in {"api", "nasadem"}:
        module = import_module(f"pysweb.dem.{name}")
        globals()[name] = module
        return module
    if name == "prepare_dem":
        api = import_module("pysweb.dem.api")
        value = api.prepare_dem
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pysweb.dem' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
