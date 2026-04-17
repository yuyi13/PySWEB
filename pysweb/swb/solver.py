"""
Script: solver.py
Objective: Provide the package-owned 1-D SWB solver and hydraulic helpers used by the SWB run workflow.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: Daily forcing arrays, per-cell soil-property dictionaries, boundary fluxes, and solver configuration values.
Outputs: Layer soil-moisture states, hydraulic matrix coefficients, and per-time-step SWB results.
Usage: Imported as `pysweb.swb.solver`
Dependencies: numpy, pandas
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_DEFAULT_NDVI_ROOT_QUANTILES = (0.2, 0.4, 0.6, 0.8, 1.0)
_DEFAULT_NDVI_ROOT_DEPTHS_CM = (5.0, 15.0, 30.0, 60.0, 100.0)


def calculate_hydraulic_properties(
    soil_moisture,
    soil_properties,
    layer,
    diff_factor,
):
    porosity = soil_properties["porosity"][layer]
    b_coefficient = soil_properties["b_coefficient"][layer]
    conductivity_sat = soil_properties["conductivity_sat"][layer]

    soil_pre_fac = max(0.01, soil_moisture / porosity)

    exponent_k = 2.0 * b_coefficient + 3.0
    conductivity = conductivity_sat * soil_pre_fac**exponent_k

    exponent_diff = b_coefficient + 1
    diffusivity = diff_factor * conductivity * (1.0 / soil_pre_fac**exponent_diff)

    return {
        "conductivity": conductivity,
        "diffusivity": diffusivity,
    }


def _resolve_sm_bounds_for_diffusion_limiter(soil_properties, porosity):
    sm_max_bound = np.asarray(soil_properties.get("sm_max_bound", porosity), dtype=float)
    sm_min_bound = np.asarray(soil_properties.get("sm_min_bound", 0.0), dtype=float)

    if sm_max_bound.ndim == 0:
        sm_max_bound = np.full_like(porosity, float(sm_max_bound))
    if sm_min_bound.ndim == 0:
        sm_min_bound = np.full_like(porosity, float(sm_min_bound))

    sm_min_bound = np.minimum(sm_min_bound, sm_max_bound)
    return sm_min_bound, sm_max_bound


def _limit_rhs_diffusive_interface_flux(
    raw_flux,
    upper_idx,
    lower_idx,
    soil_moisture,
    sm_min_bound,
    sm_max_bound,
    layer_thickness,
    time_step,
    abs_cap_mm_day=None,
):
    if not np.isfinite(raw_flux):
        return 0.0

    dt = max(float(time_step), 1.0e-12)
    if raw_flux >= 0.0:
        donor = upper_idx
        receiver = lower_idx
    else:
        donor = lower_idx
        receiver = upper_idx

    donor_available = max(
        0.0,
        (soil_moisture[donor] - sm_min_bound[donor]) * layer_thickness[donor] / dt,
    )
    receiver_capacity = max(
        0.0,
        (sm_max_bound[receiver] - soil_moisture[receiver]) * layer_thickness[receiver] / dt,
    )
    flux_cap = min(donor_available, receiver_capacity)

    if abs_cap_mm_day is not None and np.isfinite(abs_cap_mm_day) and abs_cap_mm_day > 0.0:
        flux_cap = min(flux_cap, float(abs_cap_mm_day))

    if raw_flux >= 0.0:
        return min(raw_flux, flux_cap)
    return max(raw_flux, -flux_cap)


def setup_richards_matrix(
    soil_moisture,
    soil_properties,
    boundary_fluxes,
    diff_factor,
    time_step=1.0,
):
    num_layers = len(soil_moisture)
    porosity = np.asarray(soil_properties["porosity"], dtype=float)
    sm_min_bound, sm_max_bound = _resolve_sm_bounds_for_diffusion_limiter(soil_properties, porosity)
    rhs_diffusion_enabled = bool(soil_properties.get("rhs_diffusion_enabled", True))
    rhs_diffusion_limiter_enabled = bool(soil_properties.get("rhs_diffusion_limiter_enabled", True))
    rhs_diffusion_abs_cap = soil_properties.get("rhs_diffusion_abs_cap_mm_day", None)
    time_step = max(float(time_step), 1.0e-12)

    mat_left1 = np.zeros(num_layers)
    mat_left2 = np.zeros(num_layers)
    mat_left3 = np.zeros(num_layers)
    mat_right = np.zeros(num_layers)

    conductivity = np.zeros(num_layers)
    diffusivity = np.zeros(num_layers)

    for i in range(num_layers):
        hydraulic_props = calculate_hydraulic_properties(
            soil_moisture[i],
            soil_properties,
            i,
            diff_factor,
        )
        conductivity[i] = hydraulic_props["conductivity"]
        diffusivity[i] = hydraulic_props["diffusivity"]

    layer_thickness = np.zeros(num_layers)
    distance_between_layers = np.zeros(num_layers - 1)

    for i in range(num_layers):
        if i == 0:
            layer_thickness[i] = soil_properties["layer_depth"][i]
        else:
            layer_thickness[i] = soil_properties["layer_depth"][i] - soil_properties["layer_depth"][i - 1]

    if num_layers > 1:
        distance_between_layers[:] = 0.5 * (layer_thickness[:-1] + layer_thickness[1:])

    interface_gradient = np.zeros(max(num_layers - 1, 0))
    rhs_diffusive_flux_interface = np.zeros(max(num_layers - 1, 0))
    interface_flux_downward = np.zeros(max(num_layers - 1, 0))
    if num_layers > 1:
        for i in range(num_layers - 1):
            distance = max(distance_between_layers[i], 1.0e-12)
            interface_gradient[i] = (soil_moisture[i] - soil_moisture[i + 1]) / distance

            if rhs_diffusion_enabled:
                raw_rhs_diff_flux = diffusivity[i] * interface_gradient[i]
                if rhs_diffusion_limiter_enabled:
                    rhs_diffusive_flux_interface[i] = _limit_rhs_diffusive_interface_flux(
                        raw_rhs_diff_flux,
                        i,
                        i + 1,
                        soil_moisture,
                        sm_min_bound,
                        sm_max_bound,
                        layer_thickness,
                        time_step,
                        rhs_diffusion_abs_cap,
                    )
                else:
                    rhs_diffusive_flux_interface[i] = raw_rhs_diff_flux

            interface_flux_downward[i] = conductivity[i] + rhs_diffusive_flux_interface[i]

    drainage_soil_bot = 0.0

    for i in range(num_layers):
        if i == 0:
            mat_left1[i] = 0.0
            if num_layers > 1:
                mat_left3[i] = -diffusivity[i] / (distance_between_layers[i] * layer_thickness[i])
                mat_left2[i] = -mat_left3[i]
                flux_to_below = interface_flux_downward[i]
            else:
                mat_left3[i] = 0.0
                mat_left2[i] = 0.0
                flux_to_below = conductivity[i]

            mat_right[i] = (
                boundary_fluxes["infiltration"]
                - boundary_fluxes["evapotranspiration"][i]
                - flux_to_below
            ) / layer_thickness[i]

        elif i < num_layers - 1:
            mat_left1[i] = -diffusivity[i - 1] / (distance_between_layers[i - 1] * layer_thickness[i])
            mat_left3[i] = -diffusivity[i] / (distance_between_layers[i] * layer_thickness[i])
            mat_left2[i] = -(mat_left1[i] + mat_left3[i])

            flux_from_above = interface_flux_downward[i - 1]
            flux_to_below = interface_flux_downward[i]
            mat_right[i] = (
                flux_from_above
                - flux_to_below
                - boundary_fluxes["evapotranspiration"][i]
            ) / layer_thickness[i]

        else:
            mat_left1[i] = -diffusivity[i - 1] / (distance_between_layers[i - 1] * layer_thickness[i])
            mat_left3[i] = 0.0
            mat_left2[i] = -mat_left1[i]

            drainage_soil_bot = soil_properties["drainage_slope"] * conductivity[i]
            drainage_soil_bot = min(drainage_soil_bot, soil_properties["drainage_upper_limit"])
            drainage_soil_bot = max(drainage_soil_bot, soil_properties["drainage_lower_limit"])

            flux_from_above = interface_flux_downward[i - 1]
            mat_right[i] = (
                flux_from_above
                - drainage_soil_bot
                - boundary_fluxes["evapotranspiration"][i]
            ) / layer_thickness[i]

    return {
        "mat_left1": mat_left1,
        "mat_left2": mat_left2,
        "mat_left3": mat_left3,
        "mat_right": mat_right,
        "drainage_soil_bot": drainage_soil_bot,
        "interface_gradient": interface_gradient,
        "rhs_diffusive_flux_interface": rhs_diffusive_flux_interface,
        "interface_flux_downward": interface_flux_downward,
    }


def matrix_solver_tri_diagonal(A, B, C, D, ind_top_layer=0, num_layers=5):
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


def solve_soil_moisture(soil_moisture, matrix_coeffs, time_step, soil_properties):
    num_layers = len(soil_moisture)

    porosity = np.asarray(soil_properties["porosity"], dtype=float)
    sm_max_bound = np.asarray(soil_properties.get("sm_max_bound", porosity), dtype=float)
    sm_min_bound = np.asarray(soil_properties.get("sm_min_bound", 0.0), dtype=float)
    if sm_max_bound.ndim == 0:
        sm_max_bound = np.full_like(porosity, float(sm_max_bound))
    if sm_min_bound.ndim == 0:
        sm_min_bound = np.full_like(porosity, float(sm_min_bound))
    sm_min_bound = np.minimum(sm_min_bound, sm_max_bound)
    thickness = np.asarray(soil_properties["layer_thickness"], dtype=float)

    a = matrix_coeffs["mat_left1"] * time_step
    b = 1 + matrix_coeffs["mat_left2"] * time_step
    c = matrix_coeffs["mat_left3"] * time_step
    d = matrix_coeffs["mat_right"] * time_step

    delta_moisture = matrix_solver_tri_diagonal(
        a,
        b,
        c,
        d,
        ind_top_layer=0,
        num_layers=num_layers,
    )

    new_soil_moisture = soil_moisture + delta_moisture

    for i in range(0, num_layers - 1):
        excess = max(0.0, new_soil_moisture[i] - sm_max_bound[i])
        if excess > 0.0:
            new_soil_moisture[i] = sm_max_bound[i]
            incr_next = excess * (thickness[i] / thickness[i + 1])
            new_soil_moisture[i + 1] += incr_next

    new_soil_moisture[-1] = min(new_soil_moisture[-1], sm_max_bound[-1])

    sm_min_relax_tau_days = float(soil_properties.get("sm_min_relax_tau_days", 3.0))
    sm_min_relax_trigger_factor = float(soil_properties.get("sm_min_relax_trigger_factor", 1.25))
    if not np.isfinite(sm_min_relax_trigger_factor):
        sm_min_relax_trigger_factor = 1.25
    sm_min_relax_trigger_factor = max(sm_min_relax_trigger_factor, 1.0)
    if np.isfinite(sm_min_relax_tau_days) and sm_min_relax_tau_days > 0.0:
        prev_for_relax = np.maximum(soil_moisture, sm_min_bound)
        decay = np.exp(-time_step / sm_min_relax_tau_days)
        relaxed_floor = sm_min_bound + (prev_for_relax - sm_min_bound) * decay
        relax_trigger = sm_min_bound * sm_min_relax_trigger_factor
        relax_mask = prev_for_relax <= relax_trigger
        below_min = new_soil_moisture < sm_min_bound
        apply_relax = below_min & relax_mask
        new_soil_moisture[apply_relax] = np.maximum(
            new_soil_moisture[apply_relax],
            relaxed_floor[apply_relax],
        )
        hard_clamp = below_min & (~relax_mask)
        new_soil_moisture[hard_clamp] = sm_min_bound[hard_clamp]
    else:
        new_soil_moisture = np.maximum(new_soil_moisture, sm_min_bound)

    new_soil_moisture = np.maximum(new_soil_moisture, 0.0)

    return {
        "soil_moisture": new_soil_moisture,
        "drainage_soil_bot": matrix_coeffs["drainage_soil_bot"] * time_step,
    }


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
    layer_depth_mm = np.asarray(soil_properties["layer_depth"], dtype=float)
    num_layers = len(layer_depth_mm)

    if beta is None:
        beta = float(soil_properties.get("root_beta", 0.961))
    beta = min(max(beta, 0.90), 0.995)

    if max_root_depth_mm is None:
        max_root_depth_mm = soil_properties.get("max_root_depth", None)
    if max_root_depth_mm is None and bool(use_ndvi_root_depth):
        max_root_depth_mm = _resolve_root_depth_from_ndvi(soil_properties)
    if max_root_depth_mm is not None:
        max_root_depth_mm = float(max_root_depth_mm)

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


def _as_layer_factor(value, num_layers, name):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(num_layers, float(arr))
    if arr.shape != (num_layers,):
        raise ValueError(f"{name} must be scalar or length {num_layers}.")
    return arr


def _resolve_sm_bounds(soil_properties):
    porosity = np.asarray(soil_properties["porosity"], dtype=float)
    wilting_point = np.asarray(soil_properties["wilting_point"], dtype=float)
    layer_depth = np.asarray(soil_properties["layer_depth"], dtype=float)
    num_layers = porosity.size

    if layer_depth.size != num_layers:
        raise ValueError("layer_depth length must match porosity/wilting_point length.")

    sm_max_factor = _as_layer_factor(
        soil_properties.get("sm_max_factor", 1.0),
        num_layers,
        "sm_max_factor",
    )
    sm_min_factor = _as_layer_factor(
        soil_properties.get("sm_min_factor", 1.0),
        num_layers,
        "sm_min_factor",
    )
    sm_min_factor_max_depth_mm = float(soil_properties.get("sm_min_factor_max_depth_mm", 150.0))

    sm_max_bound = porosity * sm_max_factor
    sm_min_bound = wilting_point.copy()
    shallow_mask = layer_depth <= sm_min_factor_max_depth_mm
    sm_min_bound[shallow_mask] = wilting_point[shallow_mask] * sm_min_factor[shallow_mask]
    return sm_min_bound, sm_max_bound


def soil_water_balance_1d(
    precip_data,
    et_data,
    soil_properties,
    time,
    time_step=1.0,
    initial_soil_moisture=None,
    evap_fraction=None,
    diff_factor=1e3,
    transpiration_data=None,
):
    if isinstance(precip_data, pd.DataFrame):
        precip_values = precip_data.iloc[:, 0].values
    elif isinstance(precip_data, pd.Series):
        precip_values = precip_data.values
    else:
        precip_values = np.array(precip_data).flatten()

    if isinstance(et_data, pd.DataFrame):
        et_values = et_data.iloc[:, 0].values
    elif isinstance(et_data, pd.Series):
        et_values = et_data.values
    else:
        et_values = np.array(et_data).flatten()

    if transpiration_data is not None:
        if isinstance(transpiration_data, pd.DataFrame):
            transp_values = transpiration_data.iloc[:, 0].values
        elif isinstance(transpiration_data, pd.Series):
            transp_values = transpiration_data.values
        else:
            transp_values = np.array(transpiration_data).flatten()
    else:
        transp_values = None

    time_index = pd.to_datetime(time)
    assert len(precip_values) == len(et_values), (
        "Effective precipitation and ET data must have the same length"
    )
    assert len(time_index) == len(precip_values), (
        "Time index must have the same length as effective precipitation and ET data"
    )
    if transp_values is not None:
        assert len(transp_values) == len(et_values), (
            "Transpiration and ET data must have the same length"
        )

    num_layers = len(soil_properties["layer_depth"])
    num_times = len(precip_values)

    sm_min_bound, sm_max_bound = _resolve_sm_bounds(soil_properties)
    soil_props = dict(soil_properties)
    soil_props["sm_min_bound"] = sm_min_bound
    soil_props["sm_max_bound"] = sm_max_bound

    soil_moisture_array = np.zeros((num_times, num_layers))

    root_distribution = _compute_root_distribution_beta(
        soil_props,
        beta=soil_props.get("root_beta", None),
        max_root_depth_mm=soil_props.get("max_root_depth", None),
        use_ndvi_root_depth=bool(soil_props.get("use_ndvi_root_depth", False)),
    )

    if initial_soil_moisture is None:
        current_soil_moisture = np.zeros(num_layers)
        for l in range(num_layers):
            current_soil_moisture[l] = sm_min_bound[l] + 0.5 * (sm_max_bound[l] - sm_min_bound[l])
    else:
        current_soil_moisture = np.array(initial_soil_moisture)

    soil_moisture_array[0, :] = current_soil_moisture

    for t_idx in range(num_times):
        eff_precip_t = precip_values[t_idx]
        et_t = et_values[t_idx]
        transp_t_in = transp_values[t_idx] if transp_values is not None else np.nan

        if np.isnan(eff_precip_t) or np.isnan(et_t) or (
            transp_values is not None and np.isnan(transp_t_in)
        ):
            if t_idx < num_times - 1:
                soil_moisture_array[t_idx + 1, :] = soil_moisture_array[t_idx, :]
            continue

        eff_precip_t = max(0.0, float(eff_precip_t))
        et_t = max(0.0, float(et_t))
        if transp_values is not None:
            transp_t = float(np.clip(transp_t_in, 0.0, et_t))
            evap_t = et_t - transp_t
        else:
            evap_frac = 0.3 if evap_fraction is None else float(evap_fraction)
            evap_frac = min(max(evap_frac, 0.0), 1.0)
            evap_t = et_t * evap_frac
            transp_t = et_t - evap_t

        et_by_layer = np.zeros(num_layers)
        et_by_layer[0] = evap_t
        transp_by_layer = transp_t * root_distribution
        et_by_layer += transp_by_layer

        sm_min_relax_tau_days = float(soil_props.get("sm_min_relax_tau_days", 3.0))
        sm_min_relax_trigger_factor = float(soil_props.get("sm_min_relax_trigger_factor", 1.25))
        if not np.isfinite(sm_min_relax_trigger_factor):
            sm_min_relax_trigger_factor = 1.25
        sm_min_relax_trigger_factor = max(sm_min_relax_trigger_factor, 1.0)
        if np.isfinite(sm_min_relax_tau_days) and sm_min_relax_tau_days > 0.0:
            decay = np.exp(-time_step / sm_min_relax_tau_days)
            prev_for_relax = np.maximum(current_soil_moisture, sm_min_bound)
            relaxed_floor = sm_min_bound + (prev_for_relax - sm_min_bound) * decay
            relax_trigger = sm_min_bound * sm_min_relax_trigger_factor
            relax_mask = prev_for_relax <= relax_trigger
            et_floor = np.where(relax_mask, relaxed_floor, sm_min_bound)
        else:
            et_floor = sm_min_bound

        for layer_idx in range(num_layers):
            available_for_et = max(
                0,
                (current_soil_moisture[layer_idx] - et_floor[layer_idx])
                * soil_props["layer_thickness"][layer_idx],
            )
            et_by_layer[layer_idx] = min(et_by_layer[layer_idx], available_for_et / time_step)

        boundary_fluxes = {
            "infiltration": eff_precip_t,
            "evapotranspiration": et_by_layer,
        }

        matrix_coeffs = setup_richards_matrix(
            current_soil_moisture,
            soil_props,
            boundary_fluxes,
            diff_factor=diff_factor,
            time_step=time_step,
        )

        result = solve_soil_moisture(
            current_soil_moisture,
            matrix_coeffs,
            time_step,
            soil_props,
        )

        current_soil_moisture = result["soil_moisture"]

        if t_idx < num_times - 1:
            soil_moisture_array[t_idx + 1, :] = current_soil_moisture

    layer_columns = [f"layer_{i + 1}" for i in range(num_layers)]
    soil_moisture_df = pd.DataFrame(
        soil_moisture_array,
        index=time_index,
        columns=layer_columns,
    )
    return soil_moisture_df
