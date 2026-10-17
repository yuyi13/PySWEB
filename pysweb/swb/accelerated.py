#!/usr/bin/env python3
"""
Script: accelerated.py
Objective: Provide an optional numerically equivalent JIT backend for the default column solver.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-16
Last updated: 2026-10-17
Inputs: Daily forcing and column properties accepted by the maintained solver.
Outputs: Identical state/budget schema, with explicit fallback for unsupported options.
Usage: from pysweb.swb.accelerated import soil_water_balance_1d
Dependencies: numpy, pandas, numba; PySWEB at the recorded source revision.
"""
import numpy as np
import pandas as pd
try:
    from numba import njit
except ImportError as exc:
    raise ImportError("The optional accelerated backend requires numba (PySWEB speed extra)") from exc
from pysweb.swb import solver as reference

BUDGET_NAMES=('infiltration','actual_et','drainage','overflow','storage_adjustment','storage_change','balance_residual')

@njit(cache=True, fastmath=False)
def _kernel(rain, et, tr, theta, por, bcoef, ks, thick, low, high, roots, diff, dt, slope, dlow, dhigh, end):
    nt=len(rain); nl=len(theta)
    states=np.zeros((nt,nl)); budget=np.full((nt,7),np.nan); valid=np.zeros(nt,dtype=np.bool_)
    decay=np.exp(-dt/3.)
    for t in range(nt):
        states[t]=theta
        if not (np.isfinite(rain[t]) and np.isfinite(et[t]) and np.isfinite(tr[t])):
            continue
        valid[t]=True
        before=np.sum(theta*thick)
        p=max(0.,rain[t]); e=max(0.,et[t]); transp=min(max(tr[t],0.),e)
        uptake=transp*roots
        uptake[0]+=e-transp
        cond=np.zeros(nl); diffus=np.zeros(nl)
        for j in range(nl):
            prev=max(theta[j],low[j])
            floor=low[j]+(prev-low[j])*decay if prev<=low[j]*1.25 else low[j]
            uptake[j]=min(uptake[j],max(0.,(theta[j]-floor)*thick[j])/dt)
            ratio=max(.01,theta[j]/por[j])
            cond[j]=ks[j]*ratio**(2.*bcoef[j]+3.)
            diffus[j]=diff*cond[j]*(1./ratio**(bcoef[j]+1.))
        flux=np.zeros(max(nl-1,0)); distance=np.zeros(max(nl-1,0))
        for j in range(nl-1):
            distance[j]=.5*(thick[j]+thick[j+1])
            raw=diffus[j]*(theta[j]-theta[j+1])/max(distance[j],1.e-12)
            rhs=0.
            if np.isfinite(raw):
                donor=j if raw>=0. else j+1
                receiver=j+1 if raw>=0. else j
                available=max(0.,(theta[donor]-min(low[donor],high[donor]))*thick[donor]/dt)
                capacity=max(0.,(high[receiver]-theta[receiver])*thick[receiver]/dt)
                cap=min(available,capacity)
                rhs=min(raw,cap) if raw>=0. else max(raw,-cap)
            flux[j]=cond[j]+rhs
        drain=max(min(slope*cond[-1],dhigh),dlow)
        a=np.zeros(nl); bb=np.zeros(nl); c=np.zeros(nl); d=np.zeros(nl)
        for j in range(nl):
            if j>0: a[j]=-diffus[j-1]/(distance[j-1]*thick[j])
            if j<nl-1: c[j]=-diffus[j]/(distance[j]*thick[j])
            bb[j]=-(a[j]+c[j])
            incoming=p if j==0 else flux[j-1]
            outgoing=drain if j==nl-1 else flux[j]
            d[j]=(incoming-uptake[j]-outgoing)/thick[j]*dt
            a[j]*=dt; bb[j]=1.+bb[j]*dt; c[j]*=dt
        pp=np.zeros(nl); delta=np.zeros(nl)
        pp[0]=-c[0]/bb[0]; delta[0]=d[0]/bb[0]
        for j in range(1,nl):
            inv=1./(bb[j]+a[j]*pp[j-1])
            pp[j]=-c[j]*inv; delta[j]=(d[j]-a[j]*delta[j-1])*inv
        pp[-1]=delta[-1]
        for j in range(nl-2,-1,-1): pp[j]=pp[j]*pp[j+1]+delta[j]
        new=theta+pp
        for j in range(nl-1):
            excess=max(0.,new[j]-high[j])
            if excess>0.:
                new[j]=high[j]; new[j+1]+=excess*(thick[j]/thick[j+1])
        overflow=max(0.,new[-1]-high[-1])*thick[-1]
        new[-1]=min(new[-1],high[-1])
        before_floor=np.sum(new*thick)
        for j in range(nl):
            lower=min(low[j],high[j]); prev=max(theta[j],lower)
            if new[j]<lower:
                if prev<=lower*1.25:
                    new[j]=max(new[j],lower+(prev-lower)*decay)
                else: new[j]=lower
            new[j]=max(new[j],0.)
        theta=new
        after=np.sum(theta*thick)
        if end: states[t]=theta
        budget[t,0]=p*dt; budget[t,1]=np.sum(uptake)*dt; budget[t,2]=drain*dt
        budget[t,3]=overflow; budget[t,4]=after-before_floor; budget[t,5]=after-before
        budget[t,6]=budget[t,5]-(budget[t,0]-budget[t,1]-budget[t,2]-budget[t,3]+budget[t,4])
    return states,budget,valid,theta

def soil_water_balance_1d(precip_data,et_data,soil_properties,time,time_step=1.,initial_soil_moisture=None,evap_fraction=None,diff_factor=1e3,transpiration_data=None,state_timing='start'):
    """Use JIT for the primary study process options; retain reference fallback."""
    kwargs=dict(time_step=time_step,initial_soil_moisture=initial_soil_moisture,evap_fraction=evap_fraction,diff_factor=diff_factor,transpiration_data=transpiration_data,state_timing=state_timing)
    advanced={'rhs_diffusion_enabled','rhs_diffusion_limiter_enabled','rhs_diffusion_abs_cap_mm_day','sm_min_relax_tau_days','sm_min_relax_trigger_factor'}
    if transpiration_data is None or advanced.intersection(soil_properties):
        return reference.soil_water_balance_1d(precip_data,et_data,soil_properties,time,**kwargs)
    def vector(x):
        if isinstance(x,pd.DataFrame): x=x.iloc[:,0]
        return np.asarray(x,dtype=float).flatten()
    rain=vector(precip_data); et=vector(et_data); tr=vector(transpiration_data)
    times=pd.DatetimeIndex(time)
    props=soil_properties
    bottom=reference.layer_bottoms(props['layer_depth']); thick=np.asarray(props['layer_thickness'],dtype=float)
    if not np.isfinite(time_step) or time_step<=0 or state_timing not in {'start','end'}: raise ValueError('Invalid timestep or state timing')
    if not len(times) or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing: raise ValueError('Invalid time coordinate')
    if len(times)>1 and not np.allclose(np.diff(times.to_numpy(dtype='datetime64[ns]'))/np.timedelta64(1,'D'),time_step): raise ValueError('Time cadence mismatch')
    if not np.allclose(np.diff(np.r_[0.,bottom]),thick): raise ValueError('Layer thickness mismatch')
    assert len(rain)==len(et)==len(tr)==len(times)
    low,high=reference._resolve_sm_bounds(props)
    initial=low+.5*(high-low) if initial_soil_moisture is None else np.asarray(initial_soil_moisture,dtype=float)
    if initial.shape!=bottom.shape or not np.isfinite(initial).all() or np.any(initial<0) or np.any(initial>high): raise ValueError('Invalid initial state')
    roots=reference._compute_root_distribution_beta(props,beta=props.get('root_beta'),max_root_depth_mm=props.get('max_root_depth'),use_ndvi_root_depth=bool(props.get('use_ndvi_root_depth',False)))
    por=np.asarray(props['porosity'],dtype=float); bcoef=np.asarray(props['b_coefficient'],dtype=float); ks=np.asarray(props['conductivity_sat'],dtype=float)
    if not all(v.shape==bottom.shape and np.isfinite(v).all() for v in [por,bcoef,ks,low,high]) or np.any(por<=0):
        return reference.soil_water_balance_1d(precip_data,et_data,soil_properties,time,**kwargs)
    states,budgets,valid,final=_kernel(rain,et,tr,initial.copy(),por,bcoef,ks,thick,low,high,roots,float(diff_factor),float(time_step),float(props['drainage_slope']),float(props['drainage_lower_limit']),float(props['drainage_upper_limit']),state_timing=='end')
    result=pd.DataFrame(states,index=times,columns=[f'layer_{j+1}' for j in range(len(bottom))])
    result.attrs.update(state_timing=state_timing,initial_state=initial.copy(),final_state=final,final_state_time=times[-1]+pd.Timedelta(days=time_step),water_balance={k:budgets[:,i] for i,k in enumerate(BUDGET_NAMES)},forcing_valid=valid)
    return result
