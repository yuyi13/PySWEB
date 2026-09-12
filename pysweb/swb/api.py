"""
Script: api.py
Objective: Dispatch public SWB preprocessing, calibration and simulation calls to package workflows.
Author: Yi Yu (with assistance from Codex)
Created: 2026-04-17
Last updated: 2026-04-19
Inputs: Keyword arguments describing workflow inputs, outputs and model settings.
Outputs: Preprocessing results, calibration parameters or the simulation output path.
Usage: import pysweb.swb.api
Dependencies: Python standard library; pysweb
"""
from __future__ import annotations

from importlib import import_module
import sys

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
    return import_module("pysweb.swb.preprocess").preprocess_inputs(**kwargs)


def calibrate(**kwargs):
    return import_module("pysweb.swb.calibrate").calibrate_domain(**kwargs)


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
    _PACKAGE.run = run
