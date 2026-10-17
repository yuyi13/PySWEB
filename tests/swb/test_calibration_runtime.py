#!/usr/bin/env python3
"""
Script: test_calibration_runtime.py
Objective: Verify shared-worker differential evolution including polishing matches serial deferred evaluation.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-16
Last updated: 2026-10-17
Inputs: Deterministic two-parameter fixture with fixed array arguments.
Outputs: Optimiser result and evaluation-count equivalence assertions.
Usage: pytest tests/swb/test_calibration_runtime.py (PBS)
Dependencies: pytest, numpy, scipy, calibration_runtime.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution


def objective(x,target,weights):
    return float(np.sum(weights*(x-target)**2))


def test_shared_objective_with_polishing():
    # Public-API import tests reload package modules; resolve the worker module
    # at execution time so its pickle identity matches sys.modules.
    from pysweb.swb.calibration_runtime import shared_differential_evolution
    kwargs=dict(args=(np.array([.3,-.2]),np.array([2.,3.])),seed=0,maxiter=7,popsize=5,updating='deferred',polish=True,x0=[.5,.5])
    baseline=differential_evolution(objective,[(-1,1),(-1,1)],workers=1,**kwargs)
    shared=shared_differential_evolution(objective,[(-1,1),(-1,1)],workers=2,**kwargs)
    np.testing.assert_allclose(shared.x,baseline.x,rtol=0,atol=1e-12)
    assert shared.nfev==baseline.nfev and shared.nit==baseline.nit
    assert abs(shared.fun-baseline.fun)<1e-20
