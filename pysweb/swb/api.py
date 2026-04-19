"""
Script: api.py
Objective: Provide the package facade for SWB preprocess, calibration, and run workflows.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-19
Inputs: Facade API keyword arguments for SWB preprocess, calibration, or run operations.
Outputs: Forwarded workflow execution or explicit placeholder and validation errors.
Usage: Imported as `pysweb.swb.api`
Dependencies: pysweb
"""
from __future__ import annotations

import sys

from pysweb.swb.calibrate import calibrate_domain
from pysweb.swb.preprocess import preprocess_inputs
from pysweb.swb.run import run_swb_workflow

_RUN_ENTRY_INPUT_KEYS = {
    "precip",
    "effective_precip",
    "et",
    "t",
    "ndvi",
    "soil_dir",
    "soil_porosity",
    "soil_wilting_point",
    "soil_available_water_capacity",
    "soil_b_coefficient",
    "soil_conductivity_sat",
    "param_grid",
    "output_dir",
    "output_file",
}


def preprocess(**kwargs):
    return preprocess_inputs(**kwargs)


def calibrate(**kwargs):
    return calibrate_domain(**kwargs)


def run(**kwargs):
    meaningful_inputs = {
        key: value
        for key, value in kwargs.items()
        if key in _RUN_ENTRY_INPUT_KEYS and value not in (None, "")
    }
    if not meaningful_inputs:
        raise ValueError("Missing required inputs for SWB run")

    return run_swb_workflow(**kwargs)


_PACKAGE = sys.modules.get("pysweb.swb")
if _PACKAGE is not None:
    _PACKAGE.preprocess = preprocess
    _PACKAGE.calibrate = calibrate
    _PACKAGE.run = run
