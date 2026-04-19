"""
Script: __init__.py
Objective: Expose SWB package entry points while preserving callable facade attributes across import order.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Package attribute access and submodule imports.
Outputs: Callable package facade entry points.
Usage: Imported as `pysweb.swb`
Dependencies: importlib, sys, types
"""

from importlib import import_module
import sys
import types

__all__ = ["calibrate", "preprocess", "run"]


class _SWBPackageModule(types.ModuleType):
    def __setattr__(self, name, value):
        if name in __all__ and isinstance(value, types.ModuleType):
            api = import_module("pysweb.swb.api")
            value = getattr(api, name)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _SWBPackageModule


def __getattr__(name):
    if name in __all__:
        api = import_module("pysweb.swb.api")
        value = getattr(api, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pysweb.swb' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
