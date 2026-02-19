# ------------------------ Code history --------------------------------------------------
# Original Noah-MP subroutine: ROSR12
# Original code: Guo-Yue Niu and Noah-MP team (Niu et al. 2011)
# Refactered code: C. He, P. Valayamkunnath, & refactor team (He et al. 2023)
# Python version: Yi Yu, Uni. Sydney (Yu et al., 2025)
# ----------------------------------------------------------------------------------------

import numpy as np

# Soil hydraulic properties functions
def calculate_hydraulic_properties(soil_moisture, soil_properties, layer, diff_factor):
    """
    Calculate soil hydraulic conductivity and diffusivity
    
    Parameters:
    -----------
    soil_moisture : float
        Current soil moisture [mm³/mm³]
    soil_properties : dict
        Dictionary of soil parameters including:
        - porosity: saturated soil moisture [mm³/mm³]
        - b_coefficient: Campbell's b parameter
        - conductivity_sat: saturated hydraulic conductivity [mm/day]
        - diffusivity_factor: (optional) calibration parameter for diffusivity calculation
                             If not provided, a default value is used
    layer : int
        Soil layer index
    diff_factor : float
        Calibration factor that relates K to D (default is 1e3)

    Returns:
    --------
    dict
        Dictionary containing hydraulic conductivity [mm/day] and diffusivity [mm²/day]
    """
    # Soil parameters
    porosity = soil_properties['porosity'][layer]
    b_coefficient = soil_properties['b_coefficient'][layer]
    conductivity_sat = soil_properties['conductivity_sat'][layer]  # mm/day
    
    # Calculate the pre-factor as in the Fortran code
    soil_pre_fac = max(0.01, soil_moisture / porosity)

    # Original code for conductivity calculation (commented out)
    # Calculate hydraulic conductivity
    exponent_k = 2.0 * b_coefficient + 3.0
    conductivity = conductivity_sat * soil_pre_fac**exponent_k

    # Calculate diffusivity using the calibration factor and physical relationship
    # Brooks-Corey model
    # D(θ) = diff_factor * K(θ) * (θs/θ)^(b+1)
    # diffusivity (diff_factor) is an optimisible coefficient
    exponent_diff = b_coefficient + 1
    diffusivity = diff_factor * conductivity * (1.0 / soil_pre_fac**exponent_diff)
    
    return {
        'conductivity': conductivity,
        'diffusivity': diffusivity
    }

# Matrix Setup for Richards Equation
def setup_richards_matrix(soil_moisture, soil_properties, boundary_fluxes, infil_coeff, diff_factor):
    """
    Set up the matrix coefficients for the Richards equation
    
    Parameters:
    -----------
    soil_moisture : numpy.ndarray
        Vector of current soil moisture for each layer [mm³/mm³]
    soil_properties : dict
        Dictionary of soil parameters
    boundary_fluxes : dict
        Dictionary of boundary condition fluxes (infiltration, ET, etc.) [mm/day]
    
    Returns:
    --------
    dict
        Dictionary containing matrix coefficients for the tridiagonal system
    """
    num_layers = len(soil_moisture)
    
    # Initialize matrix coefficients
    mat_left1 = np.zeros(num_layers)  # Lower diagonal
    mat_left2 = np.zeros(num_layers)  # Main diagonal
    mat_left3 = np.zeros(num_layers)  # Upper diagonal
    mat_right = np.zeros(num_layers)  # Right hand side
    
    # Calculate hydraulic properties for each layer
    conductivity = np.zeros(num_layers)
    diffusivity = np.zeros(num_layers)
    
    for i in range(num_layers):
        hydraulic_props = calculate_hydraulic_properties(soil_moisture[i], soil_properties, i, diff_factor)
        conductivity[i] = hydraulic_props['conductivity']
        diffusivity[i] = hydraulic_props['diffusivity']
    
    # Calculate layer thicknesses and distance between layer midpoints
    # Note: layer_depth and layer_thickness are in mm now
    layer_thickness = np.zeros(num_layers)
    distance_between_layers = np.zeros(num_layers - 1)

    for i in range(num_layers):
        if i == 0:  # Python is 0-indexed
            layer_thickness[i] = soil_properties['layer_depth'][i]
        else:
            layer_thickness[i] = soil_properties['layer_depth'][i] - soil_properties['layer_depth'][i - 1]

    if num_layers > 1:
        # Compute interface spacing only after all layer thicknesses are known.
        distance_between_layers[:] = 0.5 * (layer_thickness[:-1] + layer_thickness[1:])
    
    # Calculate fluxes and matrix coefficients
    drainage_soil_bot = 0.0
    
    for i in range(num_layers):
        if i == 0:
            # Top layer - surface boundary condition
            mat_left1[i] = 0  # No layer above the first one
            mat_left3[i] = -diffusivity[i] / (distance_between_layers[i] * layer_thickness[i])
            mat_left2[i] = -mat_left3[i]  # Conservation of mass
            
            # Right hand side includes infiltration and ET (all in mm/day)
            mat_right[i] = (boundary_fluxes['infiltration'] * infil_coeff - 
                           boundary_fluxes['evapotranspiration'][i] - 
                           conductivity[i]) / layer_thickness[i]
            
        elif i < num_layers - 1:
            # Middle layers
            mat_left1[i] = -diffusivity[i-1] / (distance_between_layers[i-1] * layer_thickness[i])
            mat_left3[i] = -diffusivity[i] / (distance_between_layers[i] * layer_thickness[i])
            mat_left2[i] = -(mat_left1[i] + mat_left3[i])  # Conservation of mass
            
            # Gravity-driven flux only on RHS; diffusion is handled implicitly via matrix terms.
            flux_from_above = conductivity[i-1]
            flux_to_below = conductivity[i]
            mat_right[i] = (flux_from_above - flux_to_below - boundary_fluxes['evapotranspiration'][i]) / layer_thickness[i]
            
        else:
            # Bottom layer - drainage boundary condition
            mat_left1[i] = -diffusivity[i-1] / (distance_between_layers[i-1] * layer_thickness[i])
            mat_left3[i] = 0  # No layer below the last one
            mat_left2[i] = -mat_left1[i]  # Conservation of mass
            
            # Calculate drainage based on hydraulic conductivity and slope
            drainage_soil_bot = soil_properties['drainage_slope'] * conductivity[i]
            drainage_soil_bot = min(drainage_soil_bot, soil_properties['drainage_upper_limit'])
            drainage_soil_bot = max(drainage_soil_bot, soil_properties['drainage_lower_limit'])
            
            # Flux at the bottom boundary (all in mm/day)
            # Diffusion handled implicitly; RHS carries gravity and drainage terms only.
            flux_from_above = conductivity[i-1]
            mat_right[i] = (flux_from_above - drainage_soil_bot - boundary_fluxes['evapotranspiration'][i]) / layer_thickness[i]

    return {
        'mat_left1': mat_left1,
        'mat_left2': mat_left2,
        'mat_left3': mat_left3,
        'mat_right': mat_right,
        'drainage_soil_bot': drainage_soil_bot
    }

# Noah-MP version of the tridiagonal matrix solver
def matrix_solver_tri_diagonal(A, B, C, D, ind_top_layer=0, num_layers=5):
    """
    Solve a tridiagonal matrix problem.

    This function solves the tri-diagonal matrix problem shown below:
    ###                                            ### ###  ###   ###  ###
    #B(1), C(1),  0  ,  0  ,  0  ,   . . .  ,    0   # #      #   #      #
    #A(2), B(2), C(2),  0  ,  0  ,   . . .  ,    0   # #      #   #      #
    # 0  , A(3), B(3), C(3),  0  ,   . . .  ,    0   # #      #   # D(3) #
    # 0  ,  0  , A(4), B(4), C(4),   . . .  ,    0   # # P(4) #   # D(4) #
    # 0  ,  0  ,  0  , A(5), B(5),   . . .  ,    0   # # P(5) #   # D(5) #
    # .                                          .   # #  .   # = #   .  #
    # .                                          .   # #  .   #   #   .  #
    # .                                          .   # #  .   #   #   .  #
    # 0  , . . . , 0 , A(M-2), B(M-2), C(M-2),   0   # #P(M-2)#   #D(M-2)#
    # 0  , . . . , 0 ,   0   , A(M-1), B(M-1), C(M-1)# #P(M-1)#   #D(M-1)#
    # 0  , . . . , 0 ,   0   ,   0   ,  A(M) ,  B(M) # # P(M) #   # D(M) #
    ###                                            ### ###  ###   ###  ###    

    Parameters:
    -----------
    A : ndarray
        Lower diagonal of the tridiagonal matrix
    B : ndarray
        Main diagonal of the tridiagonal matrix
    C : ndarray
        Upper diagonal of the tridiagonal matrix (modified in-place)
    D : ndarray
        Right-hand side vector
    ind_top_layer : int, optional
        Index of the top layer (default: 0)
    num_layers : int, optional
        Number of soil layers (default: 5)
        
    Returns:
    --------
    P : ndarray
        Solution vector
    """
    # Initialize solution vector P and temporary work array Delta
    P = np.zeros_like(B)
    Delta = np.zeros_like(B)
    
    # Initialize coefficient C for the lowest soil layer
    # Python uses 0-based indexing, so we need to adjust the index
    C[num_layers-1] = 0.0
    
    # Solve the coefficients for the top soil layer
    P[ind_top_layer] = -C[ind_top_layer] / B[ind_top_layer]
    Delta[ind_top_layer] = D[ind_top_layer] / B[ind_top_layer]
    
    # Solve the coefficients for soil layers 2 through num_layers
    for k in range(ind_top_layer + 1, num_layers):
        P[k] = -C[k] * (1.0 / (B[k] + A[k] * P[k-1]))
        Delta[k] = (D[k] - A[k] * Delta[k-1]) * (1.0 / (B[k] + A[k] * P[k-1]))
    
    # Set P to Delta for the lowest soil layer
    P[num_layers-1] = Delta[num_layers-1]
    
    # Adjust P for soil layers from (num_layers-1) down to ind_top_layer
    for k in range(ind_top_layer + 1, num_layers):
        kk = (num_layers - 1) - k + ind_top_layer
        P[kk] = P[kk] * P[kk+1] + Delta[kk]
    
    return P

# Soil moisture solver
def solve_soil_moisture(soil_moisture, matrix_coeffs, time_step, soil_properties):
    """
    Solve the soil moisture equations using an implicit scheme
    
    Parameters:
    -----------
    soil_moisture : numpy.ndarray
        Current soil moisture for each layer [mm³/mm³]
    matrix_coeffs : dict
        Matrix coefficients from setup_richards_matrix
    time_step : float
        Time step size [days] (default is 1.0)
    soil_properties : dict
        Dictionary of soil parameters
    
    Returns:
    --------
    dict
        Updated soil moisture and fluxes
    """
    num_layers = len(soil_moisture)

    # Pull required properties
    porosity  = np.asarray(soil_properties['porosity'], dtype=float)
    sm_max_bound = np.asarray(soil_properties.get('sm_max_bound', porosity), dtype=float)
    sm_min_bound = np.asarray(soil_properties.get('sm_min_bound', 0.0), dtype=float)
    if sm_max_bound.ndim == 0:
        sm_max_bound = np.full_like(porosity, float(sm_max_bound))
    if sm_min_bound.ndim == 0:
        sm_min_bound = np.full_like(porosity, float(sm_min_bound))
    # Ensure lower bounds do not exceed upper bounds
    sm_min_bound = np.minimum(sm_min_bound, sm_max_bound)
    thickness = np.asarray(soil_properties['layer_thickness'], dtype=float)

    # Prepare matrix coefficients for implicit time scheme
    a = matrix_coeffs['mat_left1'] * time_step
    b = 1 + matrix_coeffs['mat_left2'] * time_step
    c = matrix_coeffs['mat_left3'] * time_step
    d = matrix_coeffs['mat_right'] * time_step

    # Solve tridiagonal system for soil moisture change
    delta_moisture = matrix_solver_tri_diagonal(a, b, c, d, ind_top_layer=0, num_layers=num_layers)
    #delta_moisture = thomas_solve_tridiagonal_matrix(a, b, c, d)

    # Update soil moisture
    new_soil_moisture = soil_moisture + delta_moisture
    
    # Handle excess water (bucket approach)
    for i in range(0, num_layers-1):
        excess = max(0.0, new_soil_moisture[i] - sm_max_bound[i])
        if excess > 0.0:
            # remove excess from current layer
            new_soil_moisture[i] = sm_max_bound[i]
            # convert volumetric excess in layer i to an equivalent increment in layer i+1
            incr_next = excess * (thickness[i] / thickness[i+1])
            new_soil_moisture[i+1] += incr_next

    # bottom drainage cap (keep within upper SM bound, send rest to drainage)
    excess_bottom = max(0.0, new_soil_moisture[-1] - sm_max_bound[-1])
    new_soil_moisture[-1] = min(new_soil_moisture[-1], sm_max_bound[-1])
    drain_extra = excess_bottom * thickness[-1]  # mm

    # Enforce lower bounds with buffered relaxation so states approach the minimum
    # smoothly instead of sticking to a hard floor.
    sm_min_relax_tau_days = float(soil_properties.get('sm_min_relax_tau_days', 20.0))
    if np.isfinite(sm_min_relax_tau_days) and sm_min_relax_tau_days > 0.0:
        prev_for_relax = np.maximum(soil_moisture, sm_min_bound)
        decay = np.exp(-time_step / sm_min_relax_tau_days)
        relaxed_floor = sm_min_bound + (prev_for_relax - sm_min_bound) * decay
        below_min = new_soil_moisture < sm_min_bound
        new_soil_moisture[below_min] = np.maximum(new_soil_moisture[below_min], relaxed_floor[below_min])
    else:
        new_soil_moisture = np.maximum(new_soil_moisture, sm_min_bound)

    # Soil moisture cannot be negative.
    new_soil_moisture = np.maximum(new_soil_moisture, 0.0)

    return {
        'soil_moisture': new_soil_moisture,
        'drainage_soil_bot': matrix_coeffs['drainage_soil_bot'] * time_step  # mm of water over the time step
    }
