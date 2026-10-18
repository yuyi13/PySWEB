#!/usr/bin/env python3
"""
Script: accelerated.py
Objective: Provide an optional numerically equivalent JIT backend for the default column solver.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-16
Last updated: 2026-10-18
Inputs: Daily forcing and column properties accepted by the maintained solver.
Outputs: Identical state/budget schema, without silent reference fallback.
Usage: from pysweb.swb.accelerated import soil_water_balance_1d
Dependencies: numpy, pandas, numba; PySWEB at the recorded source revision.
"""
from numba import njit
from pysweb.swb.conservative import column_kernel
from pysweb.swb.solver import _solve

_kernel = njit(cache=True, fastmath=False)(column_kernel)


def soil_water_balance_1d(precip_data, et_data, soil_properties, time, time_step=1.,
                          initial_soil_moisture=None, evap_fraction=None, diff_factor=1e3,
                          transpiration_data=None, state_timing="start"):
    """Execute the compiled conservative kernel after common input validation."""
    result = _solve(precip_data, et_data, soil_properties, time, time_step,
        initial_soil_moisture, evap_fraction, diff_factor, transpiration_data,
        state_timing, kernel=_kernel)
    result.attrs["solver_backend"] = "numba"
    return result
