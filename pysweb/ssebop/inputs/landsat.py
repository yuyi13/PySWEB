#!/usr/bin/env python3
"""
Script: landsat.py
Objective: Preserve the legacy SSEBop Landsat import path as a compatibility shim over the canonical module.
Author: Yi Yu
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: Legacy imports from `pysweb.ssebop.inputs.landsat`.
Outputs: Re-exported canonical Landsat helper symbols.
Usage: Imported as `pysweb.ssebop.inputs.landsat`
Dependencies: pysweb.ssebop.landsat
"""

from pysweb.ssebop.landsat import *  # noqa: F401,F403
