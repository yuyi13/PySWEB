#!/usr/bin/env python3
"""
Script: __init__.py
Objective: Expose lazy visualisation submodules for plotting helpers.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Package attribute access and submodule imports.
Outputs: Lazy access to plotting submodules.
Usage: Imported as `pysweb.visualisation`
Dependencies: importlib
"""

from importlib import import_module

__all__ = ["plot_heatmap", "plot_time_series"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"pysweb.visualisation.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'pysweb.visualisation' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
