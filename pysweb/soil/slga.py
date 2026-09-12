#!/usr/bin/env python3
"""
Script: slga.py
Objective: Expose the reserved SLGA soil-backend interface with an explicit unsupported-operation error.
Author: Yi Yu (with assistance from Codex)
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Backend arguments, target grid and keyword options.
Outputs: NotImplementedError; this backend currently produces no soil data.
Usage: import pysweb.soil.slga
Dependencies: Python standard library
"""


def load_soil_properties(*, args, grid, **kwargs):
    raise NotImplementedError("Soil backend 'slga' has not been implemented yet.")
