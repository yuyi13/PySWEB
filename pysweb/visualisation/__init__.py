#!/usr/bin/env python3
"""
Script: __init__.py
Objective: Expose lazy visualisation entry points for upcoming plotting helpers.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Package attribute access.
Outputs: Lazy access to plotting helpers.
Usage: Imported as `pysweb.visualisation`
Dependencies: none
"""

__all__ = ["plot_heatmap", "plot_time_series"]


def _placeholder_plot(name):
    def _plot(*args, **kwargs):
        raise NotImplementedError(f"Plot helper '{name}' has not been implemented yet.")

    _plot.__name__ = name
    return _plot


def __getattr__(name):
    if name in __all__:
        value = _placeholder_plot(name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'pysweb.visualisation' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
