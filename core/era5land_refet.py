#!/usr/bin/env python3
"""
Script: era5land_refet.py
Objective: Provide a deprecated compatibility wrapper for package-owned ERA5-Land reference ET helpers.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-05-03
Inputs: Numeric arguments forwarded to `pysweb.met.era5land.refet`.
Outputs: Delegated meteorology conversions and short-reference ET calculations.
Usage: import helpers from pysweb.met.era5land.refet instead.
Dependencies: pysweb.met.era5land.refet
"""
from __future__ import annotations

from pysweb.met.era5land.refet import *  # noqa: F401,F403
