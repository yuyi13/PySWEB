"""
Script: test_cell_objective.py
Objective: Verify cell-day errors cannot cancel spatially and use fixed area weights.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-18
Inputs: Opposing synthetic residuals and a deterministic solver stub.
Outputs: Exact weighted-loss and fixed-support assertions.
Usage: python -m pytest tests/swb/test_cell_objective.py
Dependencies: numpy, pandas, pytest, pysweb
"""
import numpy as np
import pandas as pd
import pytest
from pysweb.swb import calibrate as module


def test_cell_day_loss_retains_opposing_errors_and_area_weights(monkeypatch):
    dates = pd.date_range('2020-01-01',periods=2)
    monkeypatch.setattr(module,'_extract_props_from_arrays',lambda *args:{})
    monkeypatch.setattr(module,'soil_water_balance_1d',lambda *args,**kwargs:pd.DataFrame({'layer_1':[.2,.2]},index=dates))
    forcing = np.ones((2,1,2))
    obs = np.asarray([[[.1,.3]],[[.1,.3]]])
    valid = np.ones((1,2),dtype=bool)
    args = ([1000,.96],forcing,forcing,forcing,obs,dates,{},valid,None,[50],0,.5,25,0,False)
    contexts = {kind:module._prepare_observation_context(forcing,forcing,forcing,obs,valid,objective=kind)
                for kind in ('cell','domain_mean')}
    assert module._compute_rmse(*args,observation_context=contexts['cell'])[0] == pytest.approx(.1)
    assert module._compute_rmse(*args,observation_context=contexts['domain_mean'])[0] == pytest.approx(0,abs=1e-15)
    obs[:,:,1] = .4
    weights = np.asarray([[1,3]])
    context = module._prepare_observation_context(forcing,forcing,forcing,obs,valid,weights,1,objective='cell')
    loss,count = module._compute_rmse(*args,observation_context=context)
    assert loss == pytest.approx(np.sqrt((.1**2+3*.2**2)/4))
    assert count == 2  # The first day was simulated but excluded from scoring.
