#!/usr/bin/env python3
"""
Script: test_accelerated_solver.py
Objective: Compare accelerated state trajectories and every water-budget term to the maintained solver.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-16
Last updated: 2026-10-18
Inputs: Seeded wet/dry/missing/negative forcing, layer geometries and parameter extremes.
Outputs: Pytest assertions on states, budgets, validity and final-state metadata.
Usage: python -m pytest tests/swb/test_accelerated_solver.py (PBS)
Dependencies: numpy, pandas, pytest, numba, PySWEB.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
pytest.importorskip('numba')
from pysweb.swb.accelerated import soil_water_balance_1d as fast
from pysweb.swb.solver import soil_water_balance_1d as slow

@pytest.mark.parametrize('layers',[1,2,5,7])
@pytest.mark.parametrize('timing',['start','end'])
@pytest.mark.parametrize('case',[0,1,2])
def test_full_trajectory_and_budgets(layers,timing,case):
    # Resolve together after package API reload tests, then inspect the executed kernel.
    from pysweb.swb.accelerated import soil_water_balance_1d as fast, _kernel
    from pysweb.swb.solver import soil_water_balance_1d as slow
    rng=np.random.default_rng(layers*100+case)
    n=220; thick=rng.uniform(30,250,layers); depths=np.cumsum(thick)
    por=rng.uniform(.35,.6,layers); wp=rng.uniform(.06,.2,layers)
    props=dict(layer_depth=depths,layer_thickness=thick,porosity=por,wilting_point=wp,
               available_water_capacity=np.minimum(.15, por-wp-.01),b_coefficient=rng.uniform(2,9,layers),
               conductivity_sat=rng.uniform(1,100,layers),root_beta=[.9,.96,.995][case],
               dry_reference_factor=[0.,.3,1.][case],wet_reference_factor=[1.,1.1,1.25][case],
               drainage_slope=.5,drainage_lower_limit=0.,drainage_upper_limit=25.)
    rain=rng.exponential(10,n)*(rng.random(n)<.1); et=rng.uniform(0,9,n); tr=et*rng.uniform(0,1,n)
    rain[10]=300.; et[30]=-2.; tr[40]=15.; tr[41]=-2.; rain[45]=-1.
    rain[50:53]=np.nan; et[150]=np.nan; tr[200]=np.nan
    dt=.5 if case==2 else 1.
    times=pd.date_range('2015-01-01',periods=n,freq=pd.Timedelta(days=dt))
    kwargs=dict(time_step=dt,diff_factor=[10.,1000.,10000.][case],transpiration_data=tr,state_timing=timing)
    if case==1: kwargs['initial_soil_moisture']=wp*.5
    a=slow(rain,et,props,times,**kwargs); b=fast(rain,et,props,times,**kwargs)
    np.testing.assert_allclose(a.values,b.values,rtol=1e-10,atol=1e-10)
    for name,expected in a.attrs['water_balance'].items():
        np.testing.assert_allclose(expected,b.attrs['water_balance'][name],rtol=1e-9,atol=1e-8,equal_nan=True)
    np.testing.assert_array_equal(a.attrs['forcing_valid'],b.attrs['forcing_valid'])
    np.testing.assert_allclose(a.attrs['final_state'],b.attrs['final_state'],rtol=1e-10,atol=1e-10)
    assert a.attrs['final_state_time']==b.attrs['final_state_time']
    assert _kernel.nopython_signatures
