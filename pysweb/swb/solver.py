"""
Script: solver.py
Objective: Solve daily soil-water balances and vertical redistribution for a layered soil column.
Author: Yi Yu (with assistance from Codex)
Created: 2026-04-17
Last updated: 2026-10-18
Inputs: Daily forcing, per-layer hydraulic properties, initial states and solver settings.
Outputs: Layer moisture and final states, hydraulic coefficients, water-budget terms and quality flags.
Usage: import pysweb.swb.solver
Dependencies: numpy, pandas; pysweb
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pysweb.contracts import layer_bottoms

from pysweb.swb.formulation import (FORMULATION_VERSION, TAU_DAYS, MAX_SUBSTEP_DAYS,
    HYDRAULIC_CFL, budget_names, reject_legacy)
from pysweb.swb.conservative import column_kernel


def calculate_hydraulic_properties(soil_moisture, soil_properties, layer, diff_factor):
    """K = Ksat S^(2b+3); D = alpha_D Ksat S^(b+2), including S=0."""
    saturation = soil_moisture / soil_properties["porosity"][layer]
    if not 0 <= saturation <= 1:
        raise ValueError("Hydraulic state outside physical saturation bounds")
    b = soil_properties["b_coefficient"][layer]
    ks = soil_properties["conductivity_sat"][layer]
    return dict(conductivity=ks*saturation**(2*b+3),
                diffusivity=diff_factor*ks*saturation**(b+2))


def _resolve_sm_bounds(soil_properties):
    reject_legacy(soil_properties)
    porosity = np.asarray(soil_properties["porosity"], dtype=float)
    return np.zeros_like(porosity), porosity.copy()


def _solve(precip_data, et_data, soil_properties, time, time_step=1.,
           initial_soil_moisture=None, evap_fraction=None, diff_factor=1e3,
           transpiration_data=None, state_timing="start", *, kernel=column_kernel):
    """Validate once outside either numerical backend; retain missing-forcing states."""
    props = dict(soil_properties)
    reject_legacy(props)
    if not np.isfinite(time_step) or time_step <= 0 or state_timing not in {"start", "end"}:
        raise ValueError("Invalid timestep or state timing")
    if not np.isfinite(diff_factor) or diff_factor < 0:
        raise ValueError("diff_factor must be finite and nonnegative")
    bottoms = layer_bottoms(props["layer_depth"])
    shape = bottoms.shape
    arrays = {}
    for key in ("porosity", "wilting_point", "available_water_capacity", "b_coefficient",
                "conductivity_sat", "layer_thickness"):
        value = np.asarray(props[key], dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{key} must have one finite value per layer")
        arrays[key] = value
    por, wp, awc, b, ks, thick = (arrays[k] for k in arrays)
    fc = wp + awc
    if (np.any(por <= 0) or np.any(por > 1) or np.any(wp < 0) or np.any(awc < 0)
            or np.any(fc > por) or np.any(b <= 0) or np.any(ks < 0)):
        raise ValueError("Invalid physical hydraulic inputs: require 0 <= WP <= FC <= theta_s <= 1, b > 0, Ksat >= 0")
    if not np.allclose(np.diff(np.r_[0., bottoms]), thick, rtol=0, atol=1e-10):
        raise ValueError("Layer thickness and bottom depths disagree")
    times = pd.DatetimeIndex(time)
    if not len(times) or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Time must be non-empty, unique and increasing")
    if len(times) > 1 and not np.allclose(np.diff(times.to_numpy(dtype="datetime64[ns]")) / np.timedelta64(1, "D"), time_step):
        raise ValueError("Time coordinate cadence must match time_step")
    def vector(value):
        if isinstance(value, pd.DataFrame):
            value = value.iloc[:, 0]
        return np.asarray(value, dtype=float).reshape(-1)
    rain, et = vector(precip_data), vector(et_data)
    fraction = .3 if evap_fraction is None else float(evap_fraction)
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("evap_fraction must lie in [0,1]")
    tr = vector(transpiration_data) if transpiration_data is not None else np.maximum(et, 0.) * (1-fraction)
    if not len(rain) == len(et) == len(tr) == len(times):
        raise ValueError("Forcing and time lengths disagree")
    initial = fc.copy() if initial_soil_moisture is None else np.asarray(initial_soil_moisture, dtype=float).copy()
    if initial.shape != shape or not np.isfinite(initial).all() or np.any(initial < 0) or np.any(initial > por):
        raise ValueError("Initial soil moisture is outside physical storage bounds")
    dry_factor = float(props.get("dry_reference_factor", 1.))
    wet_factor = float(props.get("wet_reference_factor", 1.))
    if not (np.isfinite(dry_factor) and 0 <= dry_factor <= 1 and np.isfinite(wet_factor) and wet_factor >= 1):
        raise ValueError("Invalid fixed dry/wet reference factors")
    dry, wet = wp.copy(), por.copy()
    shallow = bottoms <= 150.
    dry[shallow] *= dry_factor
    wet[shallow] *= wet_factor
    roots = _compute_root_distribution_beta(props, beta=props.get("root_beta"),
        max_root_depth_mm=props.get("max_root_depth"), use_ndvi_root_depth=bool(props.get("use_ndvi_root_depth", False)))
    slope = float(props.get("drainage_slope", .5))
    upper = float(props.get("drainage_upper_limit", 25.))
    if props.get("drainage_lower_limit", 0.) != 0.:
        raise ValueError("A positive drainage floor violates the zero-water hydraulic limit")
    # Overrides are for explicit numerical/sensitivity experiments, never optimizer parameters.
    tau = float(props.get("supply_tau_days", TAU_DAYS))
    max_step = float(props.get("max_substep_days", MAX_SUBSTEP_DAYS))
    if not all(np.isfinite(v) and v > 0 for v in (tau, max_step)) or not np.isfinite(slope) or slope < 0 or not np.isfinite(upper) or upper < 0:
        raise ValueError("Invalid drainage or fixed closure settings")
    states, budgets, valid, final = kernel(rain, et, tr, initial.copy(), por, wp, b, ks,
        thick, dry, wet, roots, float(diff_factor), float(time_step), slope, upper,
        tau, max_step, HYDRAULIC_CFL, state_timing == "end")
    result = pd.DataFrame(states, index=times, columns=[f"layer_{i+1}" for i in range(len(bottoms))])
    result.attrs.update(state_timing=state_timing, initial_state=initial,
        final_state=final, final_state_time=times[-1]+pd.Timedelta(days=time_step),
        formulation_version=FORMULATION_VERSION, supply_tau_days=tau, max_substep_days=max_step,
        water_balance={key: budgets[:, i] for i, key in enumerate(budget_names(len(bottoms)))},
        forcing_valid=valid, dry_reference=dry, wet_reference=wet)
    return result


def soil_water_balance_1d(precip_data, et_data, soil_properties, time, time_step=1.,
                          initial_soil_moisture=None, evap_fraction=None, diff_factor=1e3,
                          transpiration_data=None, state_timing="start"):
    """Conservative two-parameter SWB with fixed physical FC initialisation."""
    return _solve(precip_data, et_data, soil_properties, time, time_step,
        initial_soil_moisture, evap_fraction, diff_factor, transpiration_data, state_timing)

_DEFAULT_NDVI_ROOT_QUANTILES = (0.2, 0.4, 0.6, 0.8, 1.0)
_DEFAULT_NDVI_ROOT_DEPTHS_CM = (5.0, 15.0, 30.0, 60.0, 100.0)


def _resolve_root_depth_from_ndvi(soil_properties):
    ndvi_value = soil_properties.get("ndvi", None)
    if ndvi_value is None:
        return None

    ndvi_arr = np.asarray(ndvi_value, dtype=float).reshape(-1)
    ndvi_valid = ndvi_arr[np.isfinite(ndvi_arr)]
    if ndvi_valid.size == 0:
        return None
    ndvi_scalar = float(ndvi_valid.mean())

    quantiles = np.asarray(
        soil_properties.get("ndvi_root_quantiles", _DEFAULT_NDVI_ROOT_QUANTILES),
        dtype=float,
    ).reshape(-1)
    if quantiles.size == 0:
        return None

    depth_values_mm = soil_properties.get("ndvi_root_depths_mm", None)
    if depth_values_mm is None:
        depth_values_cm = np.asarray(
            soil_properties.get("ndvi_root_depths_cm", _DEFAULT_NDVI_ROOT_DEPTHS_CM),
            dtype=float,
        ).reshape(-1)
        depth_values_mm = depth_values_cm * 10.0
    else:
        depth_values_mm = np.asarray(depth_values_mm, dtype=float).reshape(-1)

    if depth_values_mm.size != quantiles.size:
        return None

    order = np.argsort(quantiles)
    quantiles = quantiles[order]
    depth_values_mm = depth_values_mm[order]

    ndvi_clipped = float(np.clip(ndvi_scalar, quantiles[0], quantiles[-1]))
    idx = int(np.searchsorted(quantiles, ndvi_clipped, side="left"))
    idx = min(max(idx, 0), depth_values_mm.size - 1)

    root_depth_mm = float(depth_values_mm[idx])
    if not np.isfinite(root_depth_mm) or root_depth_mm <= 0.0:
        return None
    return root_depth_mm


def _compute_root_distribution_beta(
    soil_properties,
    beta=None,
    max_root_depth_mm=None,
    use_ndvi_root_depth=False,
):
    """
    Compute per-layer root fraction using the Jackson et al. (1996) beta profile.

    F(z) = 1 - beta^z for depth z in cm. The fraction in a layer
    [z_top, z_bot] is F(z_bot) - F(z_top), which simplifies to
    beta**z_top - beta**z_bot. Typical beta values are around 0.95-0.98.
    """
    layer_depth_mm = np.asarray(soil_properties["layer_depth"], dtype=float)
    num_layers = len(layer_depth_mm)

    if beta is None:
        beta = float(soil_properties.get("root_beta", 0.961))
    if not np.isfinite(beta) or not .90 <= beta <= .995:
        raise ValueError("root_beta must be between 0.9 and 0.995")

    if max_root_depth_mm is None:
        max_root_depth_mm = soil_properties.get("max_root_depth", None)
    if max_root_depth_mm is None and bool(use_ndvi_root_depth):
        max_root_depth_mm = _resolve_root_depth_from_ndvi(soil_properties)
    if max_root_depth_mm is not None:
        max_root_depth_mm = float(max_root_depth_mm)
        if not np.isfinite(max_root_depth_mm) or max_root_depth_mm <= 0:
            raise ValueError("Maximum root depth must be finite and positive")

    layer_bot_cm = layer_depth_mm / 10.0
    layer_top_cm = np.concatenate(([0.0], layer_bot_cm[:-1]))

    if max_root_depth_mm is not None:
        root_cap_cm = max_root_depth_mm / 10.0
        layer_bot_cm = np.minimum(layer_bot_cm, root_cap_cm)
        layer_top_cm = np.minimum(layer_top_cm, root_cap_cm)

    root_frac = np.power(beta, layer_top_cm) - np.power(beta, layer_bot_cm)
    root_frac = np.clip(root_frac, 0.0, None)
    total = root_frac.sum()
    if total <= 0:
        root_frac = np.zeros(num_layers)
        root_frac[0] = 1.0
    else:
        root_frac = root_frac / total

    return root_frac


def matrix_solver_tri_diagonal(A, B, C, D, ind_top_layer=0, num_layers=5):
    """
    Solve the Noah-MP-style tridiagonal matrix system used by the SWB layers.

    `A`, `B`, and `C` are the lower, main, and upper diagonals, and `D` is the
    right-hand side. The function returns the solution vector over
    `num_layers`, starting at `ind_top_layer`.
    """
    P = np.zeros_like(B)
    delta = np.zeros_like(B)

    C[num_layers - 1] = 0.0

    P[ind_top_layer] = -C[ind_top_layer] / B[ind_top_layer]
    delta[ind_top_layer] = D[ind_top_layer] / B[ind_top_layer]

    for k in range(ind_top_layer + 1, num_layers):
        P[k] = -C[k] * (1.0 / (B[k] + A[k] * P[k - 1]))
        delta[k] = (D[k] - A[k] * delta[k - 1]) * (1.0 / (B[k] + A[k] * P[k - 1]))

    P[num_layers - 1] = delta[num_layers - 1]

    for k in range(ind_top_layer + 1, num_layers):
        kk = (num_layers - 1) - k + ind_top_layer
        P[kk] = P[kk] * P[kk + 1] + delta[kk]

    return P


def thomas_solve_tridiagonal_matrix(a, b, c, d):
    """
    Solve a general tridiagonal linear system using the Thomas algorithm.

    This standalone helper preserves the clearer legacy utility interface:
    `a`, `b`, and `c` are lower, main, and upper diagonal arrays, and `d` is
    the right-hand-side vector. Inputs are copied before elimination.
    """
    n = len(d)
    a_c = np.asarray(a, dtype=float).copy()
    b_c = np.asarray(b, dtype=float).copy()
    c_c = np.asarray(c, dtype=float).copy()
    d_c = np.asarray(d, dtype=float).copy()
    x = np.zeros(n, dtype=float)

    for i in range(1, n):
        factor = a_c[i] / b_c[i - 1]
        b_c[i] = b_c[i] - factor * c_c[i - 1]
        d_c[i] = d_c[i] - factor * d_c[i - 1]

    x[n - 1] = d_c[n - 1] / b_c[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (d_c[i] - c_c[i] * x[i + 1]) / b_c[i]

    return x
