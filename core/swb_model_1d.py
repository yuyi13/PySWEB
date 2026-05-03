#!/usr/bin/env python3
"""
Script: swb_model_1d.py
Objective: Provide a deprecated compatibility wrapper for the package-owned 1-D SWB solver.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-05-03
Inputs: Arguments forwarded to `pysweb.swb.solver`.
Outputs: Delegated layer soil-moisture states, runoff/drainage components, and diagnostics.
Usage: import soil_water_balance_1d from pysweb.swb.solver instead.
Dependencies: pysweb.swb.solver
"""
from __future__ import annotations

from pysweb.swb.solver import *  # noqa: F401,F403
