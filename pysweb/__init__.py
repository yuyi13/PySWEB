#!/usr/bin/env python3
"""
Script: __init__.py
Objective: Expose the top-level pysweb package namespaces lazily.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Package attribute access and submodule imports.
Outputs: Lazy access to pysweb subpackages.
Usage: Imported as `pysweb`
Dependencies: importlib
"""

from importlib import import_module

__all__ = ["io", "met", "ssebop", "soil", "swb", "visualisation"]

_SUBMODULES = {name: f"pysweb.{name}" for name in __all__}


def __getattr__(name):
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'pysweb' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
