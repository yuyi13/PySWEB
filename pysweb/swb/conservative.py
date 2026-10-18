"""
Script: conservative.py
Objective: Advance a soil column with jointly limited conservative hydraulic fluxes and separate ET sinks.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-18
Last updated: 2026-10-18
Inputs: Validated float64 column properties, forcing and fixed closure settings.
Outputs: States, interval budgets, interface fluxes, exposure durations and final state.
Usage: Called by solver.py directly or compiled by accelerated.py.
Dependencies: numpy
"""
import numpy as np


def column_kernel(rain, et, tr, theta, por, wp, bcoef, ks, thick, dry, wet,
                  roots, diff, dt, slope, dhigh, tau, end):
    """Finite volumes; interface flux positive downward, all budget volumes in mm.

    Hydraulic K and D at the upper layer retain the previous interface law.
    One bounded update is made per forcing interval (one day in workflows).
    Hydraulic rates and extraction caps use the beginning-of-interval state.
    All outgoing transfers and sinks share one donor budget. Incoming hydraulic
    transfers share the receiver's pre-step capacity; rainfall excess cascades
    downward with exactly the same transfer recorded in both neighbouring layers.
    """
    nt, nl = len(rain), len(theta)
    ns = min(2, nl)
    count = 14 + nl - 1 + 3 * nl + 2 * ns
    states = np.zeros((nt, nl))
    budgets = np.full((nt, count), np.nan)
    valid = np.zeros(nt, dtype=np.bool_)
    capacity = por * thick
    for t in range(nt):
        states[t] = theta
        if not (np.isfinite(rain[t]) and np.isfinite(et[t]) and np.isfinite(tr[t])):
            continue
        valid[t] = True
        p = max(0., rain[t])
        demand = max(0., et[t])
        trans = min(max(0., tr[t]), demand)
        evap = demand - trans
        before = theta * thick
        qsum = np.zeros(max(0, nl-1))
        tsum = np.zeros(nl)
        below_wp = np.zeros(nl)
        below_dry = np.zeros(ns)
        above_wet = np.zeros(ns)
        esum, drainage, overflow = 0., 0., 0.
        h = dt
        water = theta * thick
        cond = np.zeros(nl)
        diffus = np.zeros(nl)
        flux = np.zeros(max(0, nl-1))
        for j in range(nl):
            ratio = theta[j] / por[j]
            cond[j] = ks[j] * ratio ** (2. * bcoef[j] + 3.)
            diffus[j] = diff * ks[j] * ratio ** (bcoef[j] + 2.)
        for j in range(nl-1):
            conductance = diffus[j] / (.5 * (thick[j] + thick[j+1]))
            flux[j] = cond[j] + conductance * (theta[j] - theta[j+1])
        release = -np.expm1(-h / tau)
        e = 0.
        if theta[0] > 0.:
            e = min(evap * h, thick[0] * theta[0] * theta[0] / (theta[0] + dry[0]) * release)
        uptake = np.minimum(trans * roots * h, thick * np.maximum(theta - wp, 0.) * release)
        drain = min(slope * cond[-1], dhigh) * h
        q = flux * h
        outgoing = uptake.copy()
        outgoing[0] += e
        outgoing[-1] += drain
        for j in range(nl-1):
            if q[j] >= 0.:
                outgoing[j] += q[j]
            else:
                outgoing[j+1] -= q[j]
        donor_scale = np.ones(nl)
        for j in range(nl):
            if outgoing[j] > water[j]:
                donor_scale[j] = water[j] / outgoing[j]
        e *= donor_scale[0]
        uptake *= donor_scale
        drain *= donor_scale[-1]
        incoming = np.zeros(nl)
        for j in range(nl-1):
            donor = j if q[j] >= 0. else j+1
            receiver = j+1 if q[j] >= 0. else j
            q[j] *= donor_scale[donor]
            incoming[receiver] += abs(q[j])
        receiver_scale = np.ones(nl)
        for j in range(nl):
            headroom = max(0., capacity[j] - water[j])
            if incoming[j] > headroom:
                receiver_scale[j] = headroom / incoming[j]
        for j in range(nl-1):
            receiver = j+1 if q[j] >= 0. else j
            q[j] *= receiver_scale[receiver]
        for j in range(nl):
            if theta[j] < wp[j]:
                below_wp[j] += h
        for j in range(ns):
            if theta[j] < dry[j]:
                below_dry[j] += h
            if theta[j] > wet[j]:
                above_wet[j] += h
        water -= uptake
        water[0] += p * h - e
        water[-1] -= drain
        for j in range(nl-1):
            water[j] -= q[j]
            water[j+1] += q[j]
        # Saturation overflow is an explicit transfer, never discarded storage.
        for j in range(nl):
            excess = max(0., water[j] - capacity[j])
            water[j] -= excess
            if j < nl-1:
                water[j+1] += excess
                q[j] += excess
            else:
                overflow += excess
            if water[j] < -1e-10:
                raise ValueError("Negative storage after joint donor limiting")
            # At most floating point cancellation; no WP/reference restoration.
            water[j] = max(0., water[j])
        theta = water / thick
        esum += e
        tsum += uptake
        drainage += drain
        qsum += q
        if end:
            states[t] = theta
        change = theta * thick - before
        total_t = np.sum(tsum)
        budgets[t, :14] = np.array([p*dt, esum+total_t, esum, total_t,
            evap*dt, trans*dt, evap*dt-esum, trans*dt-total_t,
            drainage, overflow, 0., np.sum(change),
            np.sum(change)-(p*dt-esum-total_t-drainage-overflow), 1.])
        offset = 14
        budgets[t, offset:offset+nl-1] = qsum
        offset += nl-1
        budgets[t, offset:offset+nl] = tsum
        offset += nl
        for j in range(nl):
            incoming = p*dt if j == 0 else qsum[j-1]
            outgoing = drainage+overflow if j == nl-1 else qsum[j]
            budgets[t, offset+j] = change[j] - (incoming-outgoing-tsum[j]-(esum if j == 0 else 0.))
        offset += nl
        budgets[t, offset:offset+nl] = below_wp
        offset += nl
        budgets[t, offset:offset+ns] = below_dry
        budgets[t, offset+ns:offset+2*ns] = above_wet
    return states, budgets, valid, theta
