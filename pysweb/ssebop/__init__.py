"""
Script: __init__.py
Objective: Expose SSEBop input preparation and model execution through lazy API imports.
Author: Yi Yu (with assistance from Codex)
Created: 2026-04-17
Last updated: Unknown
Inputs: Package attribute access and submodule imports.
Outputs: The prepare_inputs and run API functions.
Usage: import pysweb.ssebop
Dependencies: Python standard library
"""

from importlib import import_module

__all__ = ["prepare_inputs", "run"]


def __getattr__(name):
    if name in __all__:
        api = import_module("pysweb.ssebop.api")
        value = getattr(api, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pysweb.ssebop' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
