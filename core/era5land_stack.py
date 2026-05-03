#!/usr/bin/env python3
"""
Script: era5land_stack.py
Objective: Provide a deprecated compatibility wrapper for package-owned ERA5-Land stacking helpers.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-05-03
Inputs: Arguments forwarded to `pysweb.met.era5land.stack`.
Outputs: Delegated daily file discovery and NetCDF stacking behavior.
Usage: import helpers from pysweb.met.era5land.stack instead.
Dependencies: pysweb.met.era5land.stack
"""
from __future__ import annotations

from pysweb.met.era5land.stack import *  # noqa: F401,F403
