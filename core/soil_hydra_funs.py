#!/usr/bin/env python3
"""
Script: soil_hydra_funs.py
Objective: Provide a deprecated compatibility wrapper for package-owned SWB hydraulic helpers.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-05-03
Inputs: Arguments forwarded to `pysweb.swb.solver`.
Outputs: Delegated hydraulic coefficients, tridiagonal system terms, and soil-moisture updates.
Usage: import helpers from pysweb.swb.solver instead.
Dependencies: pysweb.swb.solver
"""
from __future__ import annotations

from pysweb.swb.solver import *  # noqa: F401,F403
