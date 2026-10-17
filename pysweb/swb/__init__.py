"""
Script: __init__.py
Objective: Expose SWB package entry points while preserving callable facade attributes across import order.
Author: Yi Yu (with assistance from Codex)
Created: 2026-04-19
Last updated: 2026-10-17
Inputs: Package attribute access and submodule imports.
Outputs: Callable package facade entry points.
Usage: import pysweb.swb
Dependencies: Python standard library
"""

from importlib import import_module

__all__ = ["calibrate", "preprocess", "run"]


def __getattr__(name):
    if name == "preprocess":
        return import_module("pysweb.swb.preprocess")
    if name == "calibrate":
        return import_module("pysweb.swb.calibrate")
    if name == "run":
        return import_module("pysweb.swb.run")
    raise AttributeError(f"module 'pysweb.swb' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
