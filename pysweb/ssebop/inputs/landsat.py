#!/usr/bin/env python3
"""
Script: landsat.py
Objective: Preserve the legacy SSEBop Landsat import path as a compatibility shim over the canonical module.
Author: Yi Yu (with assistance from Codex)
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: Legacy imports from `pysweb.ssebop.inputs.landsat`.
Outputs: Re-exported canonical Landsat helper symbols.
Usage: import pysweb.ssebop.inputs.landsat
Dependencies: Python standard library; pysweb
"""

from pysweb.ssebop.landsat import *  # noqa: F401,F403
