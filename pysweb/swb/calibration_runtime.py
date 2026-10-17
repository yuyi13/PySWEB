#!/usr/bin/env python3
"""
Script: calibration_runtime.py
Objective: Share immutable calibration arrays across local fork workers while retaining the maintained optimiser settings.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-16
Last updated: 2026-10-17
Inputs: Maintained differential-evolution objective, bounds and fixed argument arrays.
Outputs: SciPy optimiser result and optional per-generation progress ledger.
Usage: Select shared_workers=True in calibrate_domain on a local fork-capable host.
Dependencies: scipy, multiprocessing, numpy.
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
_OBJECTIVE=None
_ARGS=None


def _evaluate(parameters):
    return _OBJECTIVE(parameters,*_ARGS)


class _SharedMap:
    def __init__(self,pool):
        self.pool=pool

    def __call__(self,function,values):
        # SciPy uses the same wrapped objective for population and polishing.
        # Refuse an unexpected callable instead of silently changing semantics.
        if getattr(function,'f',None) is not _OBJECTIVE:
            raise ValueError('Unexpected objective passed to shared calibration map')
        return self.pool.map(_evaluate,values,chunksize=1)


def shared_differential_evolution(function,bounds,args=(),workers=1,x0=None,**kwargs):
    global _OBJECTIVE,_ARGS
    if workers<=1:
        return differential_evolution(function,bounds,args=args,workers=workers,x0=x0,**kwargs)
    _OBJECTIVE=function;_ARGS=args
    path=os.environ.get('SWEB_CALIB_PROGRESS')
    generations=[]
    if path:
        def callback(intermediate_result):
            generations.append({'generation':int(intermediate_result.nit),'objective':float(intermediate_result.fun),'parameters':np.asarray(intermediate_result.x).tolist()})
            Path(path).write_text(json.dumps(generations,indent=2)+'\n')
        kwargs['callback']=callback
    try:
        with mp.get_context('fork').Pool(workers) as pool:
            return differential_evolution(function,bounds,args=args,workers=_SharedMap(pool),x0=x0,**kwargs)
    finally:
        _OBJECTIVE=None;_ARGS=None
