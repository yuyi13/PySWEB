import sys
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.soil_hydra_funs import setup_richards_matrix
from core.soil_hydra_funs import solve_soil_moisture

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

    F(z) = 1 - beta^z (z in cm). Fraction in a layer [z_top,z_bot] is F(z_bot)-F(z_top)
    which simplifies to beta^{z_top} - beta^{z_bot}. Typical beta ~ 0.95-0.98.

    Parameters
    ----------
    soil_properties : dict
        Must contain 'layer_depth' (cumulative depth to layer bottoms, mm).
    beta : float, optional
        Root beta parameter; if None, looks for 'root_beta' in soil_properties,
        else defaults to 0.961 (crop average; Jackson et al., 1996).
    max_root_depth_mm : float, optional
        Maximum rooting depth in mm. If None, looks for 'max_root_depth' in
        soil_properties.
    use_ndvi_root_depth : bool, optional
        If True and max_root_depth_mm is not provided, derive rooting depth
        from NDVI quantile mapping when 'ndvi' is available. Default is False.

    Returns
    -------
    np.ndarray
        Length N array of root fractions that sum to 1 across layers within rooting depth.
    """
    layer_depth_mm = np.asarray(soil_properties['layer_depth'], dtype=float)
    num_layers = len(layer_depth_mm)

    if beta is None:
        beta = float(soil_properties.get('root_beta', 0.961))
    beta = min(max(beta, 0.90), 0.995)  # keep in a reasonable range

    if max_root_depth_mm is None:
        max_root_depth_mm = soil_properties.get('max_root_depth', None)
    if max_root_depth_mm is None and bool(use_ndvi_root_depth):
        max_root_depth_mm = _resolve_root_depth_from_ndvi(soil_properties)
    if max_root_depth_mm is not None:
        max_root_depth_mm = float(max_root_depth_mm)

    # depths in cm for the Jackson formulation
    layer_bot_cm = layer_depth_mm / 10.0
    layer_top_cm = np.concatenate(([0.0], layer_bot_cm[:-1]))

    # Apply rooting depth cap if provided
    if max_root_depth_mm is not None:
        root_cap_cm = max_root_depth_mm / 10.0
        layer_bot_cm = np.minimum(layer_bot_cm, root_cap_cm)
        layer_top_cm = np.minimum(layer_top_cm, root_cap_cm)

    # Fraction in each layer: beta^{z_top} - beta^{z_bot}
    # Layers fully below max_root_depth will have zero thickness and contribute 0.
    root_frac = np.power(beta, layer_top_cm) - np.power(beta, layer_bot_cm)

    # Zero-out any negative numerical noise and renormalize
    root_frac = np.clip(root_frac, 0.0, None)
    total = root_frac.sum()
    if total <= 0:
        # fallback: all roots in top layer
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
    porosity = np.asarray(soil_properties['porosity'], dtype=float)
    wilting_point = np.asarray(soil_properties['wilting_point'], dtype=float)
    layer_depth = np.asarray(soil_properties['layer_depth'], dtype=float)
    num_layers = porosity.size

    if layer_depth.size != num_layers:
        raise ValueError("layer_depth length must match porosity/wilting_point length.")

    sm_max_factor = _as_layer_factor(soil_properties.get('sm_max_factor', 1.0), num_layers, "sm_max_factor")
    sm_min_factor = _as_layer_factor(soil_properties.get('sm_min_factor', 1.0), num_layers, "sm_min_factor")
    sm_min_factor_max_depth_mm = float(soil_properties.get('sm_min_factor_max_depth_mm', 150.0))

    sm_max_bound = porosity * sm_max_factor
    sm_min_bound = wilting_point.copy()
    shallow_mask = layer_depth <= sm_min_factor_max_depth_mm
    sm_min_bound[shallow_mask] = wilting_point[shallow_mask] * sm_min_factor[shallow_mask]
    return sm_min_bound, sm_max_bound


def soil_water_balance_1d(precip_data, 
                          et_data,
                          soil_properties, 
                          time,
                          time_step=1.0, 
                          initial_soil_moisture=None, 
                          evap_fraction=None,
                          diff_factor=1e3,
                          transpiration_data=None):
    """
    Run soil water balance model for 1D time series data (no spatial dimensions)
    
    Parameters:
    -----------
    precip_data : pandas.DataFrame or pandas.Series or numpy.ndarray
        Effective precipitation/infiltration forcing [mm/day] as a time series.
    et_data : pandas.DataFrame or pandas.Series or numpy.ndarray
        Actual evapotranspiration data [mm/day] as a time series
    transpiration_data : pandas.DataFrame or pandas.Series or numpy.ndarray, optional
        Plant transpiration [mm/day] as a time series. If provided, ET partitioning uses
        transpiration_data directly and does not use evap_fraction.
    time : pandas.DatetimeIndex or array-like
        Time index for the data
    soil_properties : dict
        Dictionary of soil parameters including layer depths in mm
        Optional:
        - use_ndvi_root_depth: if True, allow NDVI-based root depth cap when
          max_root_depth is not provided (default: False)
        - sm_max_factor: scalar or per-layer multiplier for porosity (upper SM bound)
        - sm_min_factor: scalar or per-layer multiplier for wilting point (lower SM bound)
          applied only to layers with bottom depth <= sm_min_factor_max_depth_mm
        - sm_min_factor_max_depth_mm: depth threshold (mm) for applying sm_min_factor
        - sm_min_relax_tau_days: e-folding time (days) for buffered approach to lower bounds
        - sm_min_relax_trigger_factor: trigger multiple on sm_min_bound; buffered
          relaxation is applied only when layer moisture is at/below this threshold
        - rhs_diffusion_enabled: include Fortran-style explicit diffusion-gradient RHS flux
        - rhs_diffusion_limiter_enabled: constrain RHS diffusion by donor/receiver storage
        - rhs_diffusion_abs_cap_mm_day: optional hard cap for RHS diffusion flux magnitude
    time_step : float
        Time step in days (default: 1.0 day)
    initial_soil_moisture : numpy.ndarray, optional
        Initial soil moisture for each layer. If None, will initialize 
        with the midpoint between the SM lower/upper bounds.
    evap_fraction : float, optional
        Legacy fallback: fraction of ET allocated to surface evaporation.
        Used only when transpiration_data is not provided.
    diff_factor : float
        Diffusivity scaling factor (default: 1e3)

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing soil moisture for each layer and time step,
        with the provided time index
    """
    
    # Handle different input types for effective precipitation and ET data.
    if isinstance(precip_data, pd.DataFrame):
        precip_values = precip_data.iloc[:, 0].values  # Take first column
    elif isinstance(precip_data, pd.Series):
        precip_values = precip_data.values
    else:
        precip_values = np.array(precip_data).flatten()
    
    if isinstance(et_data, pd.DataFrame):
        et_values = et_data.iloc[:, 0].values  # Take first column
    elif isinstance(et_data, pd.Series):
        et_values = et_data.values
    else:
        et_values = np.array(et_data).flatten()

    transp_values = None
    if transpiration_data is not None:
        if isinstance(transpiration_data, pd.DataFrame):
            transp_values = transpiration_data.iloc[:, 0].values
        elif isinstance(transpiration_data, pd.Series):
            transp_values = transpiration_data.values
        else:
            transp_values = np.array(transpiration_data).flatten()
    
    # Convert time to pandas DatetimeIndex if it isn't already
    if not isinstance(time, pd.DatetimeIndex):
        time_index = pd.to_datetime(time)
    else:
        time_index = time
    
    # Ensure all inputs have the same length
    assert len(precip_values) == len(et_values), "Effective precipitation and ET data must have the same length"
    assert len(time_index) == len(precip_values), "Time index must have the same length as effective precipitation and ET data"
    if transp_values is not None:
        assert len(transp_values) == len(et_values), "Transpiration and ET data must have the same length"

    # Extract dimensions
    num_layers = len(soil_properties['layer_depth'])
    num_times = len(precip_values)

    sm_min_bound, sm_max_bound = _resolve_sm_bounds(soil_properties)
    soil_props = dict(soil_properties)
    soil_props["sm_min_bound"] = sm_min_bound
    soil_props["sm_max_bound"] = sm_max_bound
    
    # Create output array for soil moisture
    soil_moisture_array = np.zeros((num_times, num_layers))
    
    # Create a realistic root distribution profile using Jackson beta profile
    # Avoid thickness-proportional allocation that biases lower thick layers
    root_distribution = _compute_root_distribution_beta(
        soil_props,
        beta=soil_props.get('root_beta', None),
        max_root_depth_mm=soil_props.get('max_root_depth', None),
        use_ndvi_root_depth=bool(soil_props.get('use_ndvi_root_depth', False)),
    )

    # Initialize soil moisture (middle of available water capacity)
    if initial_soil_moisture is None:
        current_soil_moisture = np.zeros(num_layers)
        for l in range(num_layers):
            current_soil_moisture[l] = sm_min_bound[l] + 0.5 * (sm_max_bound[l] - sm_min_bound[l])
    else:
        current_soil_moisture = np.array(initial_soil_moisture)
    
    # Store initial soil moisture for first timestep
    soil_moisture_array[0, :] = current_soil_moisture
    
    # Main time loop
    for t in range(num_times):
        # Get effective precipitation/infiltration and ET for this time step.
        eff_precip_t = precip_values[t]
        et_t = et_values[t]
        transp_t_in = transp_values[t] if transp_values is not None else np.nan
        
        # Skip computation for missing values
        if np.isnan(eff_precip_t) or np.isnan(et_t) or (transp_values is not None and np.isnan(transp_t_in)):
            if t < num_times - 1:
                soil_moisture_array[t+1, :] = soil_moisture_array[t, :]  # Maintain previous value
            continue
        eff_precip_t = max(0.0, float(eff_precip_t))
        
        et_t = max(0.0, float(et_t))
        if transp_values is not None:
            # Primary path: transpiration provided externally (e.g., SSEBop T = ET * Tc).
            transp_t = float(np.clip(transp_t_in, 0.0, et_t))
            evap_t = et_t - transp_t
        else:
            # Backward-compatible fallback for legacy callers.
            evap_frac = 0.3 if evap_fraction is None else float(evap_fraction)
            evap_frac = min(max(evap_frac, 0.0), 1.0)
            evap_t = et_t * evap_frac
            transp_t = et_t - evap_t
        
        # Initialize ET by layer array
        et_by_layer = np.zeros(num_layers)
        
        # 1. Distribute evaporation to top layer
        et_by_layer[0] = evap_t

        # 2. Distribute transpiration across layers using root distribution
        transp_by_layer = transp_t * root_distribution
        et_by_layer += transp_by_layer
        
        # 3. Adjust ET to not exceed available water in each layer
        sm_min_relax_tau_days = float(soil_props.get('sm_min_relax_tau_days', 3.0))
        sm_min_relax_trigger_factor = float(soil_props.get('sm_min_relax_trigger_factor', 1.25))
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

        for l in range(num_layers):
            # Maximum possible ET is limited by available water above the buffered floor.
            available_for_et = max(0, (current_soil_moisture[l] - et_floor[l]) *
                                  soil_props['layer_thickness'][l])
            et_by_layer[l] = min(et_by_layer[l], available_for_et / time_step)
        
        # Set up boundary fluxes (mm/day)
        boundary_fluxes = {
            'infiltration': eff_precip_t,  # mm/day
            'evapotranspiration': et_by_layer  # mm/day for each layer
        }
        
        # Calculate matrix coefficients for Richards equation
        matrix_coeffs = setup_richards_matrix(
            current_soil_moisture,
            soil_props,
            boundary_fluxes,
            diff_factor = diff_factor,
            time_step = time_step,
        )
        
        # Solve for updated soil moisture
        result = solve_soil_moisture(
            current_soil_moisture,
            matrix_coeffs,
            time_step,
            soil_props
        )
        
        # Update current soil moisture
        current_soil_moisture = result['soil_moisture']
        
        # Store soil moisture for next time step
        if t < num_times - 1:
            soil_moisture_array[t+1, :] = current_soil_moisture
    
    # Create column names for each layer
    layer_columns = [f'layer_{i+1}' for i in range(num_layers)]
    
    # Convert to DataFrame with proper time index and layer columns
    soil_moisture_df = pd.DataFrame(
        soil_moisture_array, 
        index=time_index,
        columns=layer_columns
    )
    
    return soil_moisture_df
