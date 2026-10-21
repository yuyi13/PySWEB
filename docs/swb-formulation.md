# Conservative two-parameter SWB

Formulation: `swb-daily-depth-extraction-2p-v3`. Reference schema: `gssm-native-p01-p99-v1`.

The revised water balance separates physical storage, empirical surface-moisture references and root-extraction limits. Only the diffusivity scale `diff_factor` (10–10,000 mm) and the beta root-distribution coefficient `root_beta` (0.9–0.995) are calibrated. Root beta defaults to **0.96** in normal runs, the low-level root-distribution helper and the initial calibration seed; explicit values override this default. Physical saturation water content θs, wilting point WP, field capacity FC, saturated conductivity Ksat and the hydraulic exponent b remain fixed. The model layers end at 50, 150, 300, 600 and 1,000 mm. Layer 1 represents 0–5 cm and directly matches the GSSM observation depth.

## Frozen GSSM references

For each admitted native GSSM cell, temporal p01 and p99 are calculated over 2011–2020 using the linear quantile method and finite volumetric observations. Complete-cell spherical areas then weight the cell quantiles. Each cell requires at least 99% temporal coverage; missing days remain missing. The artifact records per-cell quantiles, areas, valid counts and missing dates, together with the input hashes. The reference period is independent of the Landsat simulation and calibration dates.

Layer-1 hydraulic fields are calculated at the native soil-predictor cells before spatial averaging. Exact spherical overlap areas integrate these fields over the same union of complete GSSM cells. The implementation rejects incomplete native hydraulic coverage and invalid hydraulic values. This reference calculation preserves the nonlinear order of the pedotransfer calculation. Existing coarse and fine simulation soils retain their established preparation pathway; their spatial aggregation approximation remains a transfer limitation.

Let Gd and Gw denote the area means of the native-cell p01 and p99. The fixed scalars are `c_d = min(1, Gd / mean(WP_1))` and `c_w = max(1, Gw / mean(θs_1))`. The dry ratio is GSSM divided by hydraulic WP, so a lower GSSM reference reduces the extraction threshold. The first-layer reference is transferred to depth using the following fixed profile:

| Layer | Depth | Extraction threshold d_i |
|---|---|---|
| 1 | 0–5 cm | `c_d WP_1` |
| 2 | 5–15 cm | `c_d WP_2` |
| 3 | 15–30 cm | `(WP_3 + c_d WP_3) / 2` |
| 4 | 30–60 cm | `WP_4` |
| 5 | 60–100 cm | `WP_5` |

The third-layer midpoint combines its own hydraulic WP and its own fully adjusted value. For other layer grids, the adjustment weight is one over 0–150 mm, one-half over 150–300 mm and zero below 300 mm, averaged by layer thickness. Surface-derived thresholds in layers 2 and 3 are empirical extrapolations; GSSM provides no direct observation at those depths. Direct evaporation still draws from layer 1. The wet references remain `w_i = c_w θs_i` in the first two standard layers and support diagnostics only. They leave physical storage capacity unchanged.

The JSON artifact is hashed over its complete canonical content. Set `[model].reference_file` in a configured workflow, or pass `reference_file` to calibration and simulation. Default and calibrated runs, initialisation checks and coarse-to-fine transfer use the same artifact. Native configured preparation retains the artifact's frozen complete-cell population and rejects loss of a member through invalid forcing or soil. Generic component runs without an artifact use the explicitly identified `physical-unadjusted` reference, with dry/wet factors of one.

## Storage, hydraulic fluxes and extraction

Physical state bounds are `0 ≤ θ_i ≤ θs_i`. Initialisation uses `FC = WP + available_water_capacity`. An initially dry layer receives no floor-restoration water. Root fractions integrate the beta profile over the actual rooted portion of each layer, using depth in centimetres and the fixed maximum root depth (1,000 mm by default). Fractions sum to one over the rooted profile, with no compensatory uptake.

For the daily update, h = 1 day and `r = -expm1(-h/τ)`, with fixed `τ = 3 days`. The surface evaporation supply cap is `dz_1 θ_1²/(θ_1+d_1) r` in mm. Actual evaporation is the minimum of that cap and requested surface evaporation multiplied by h. The cap is zero at zero water and has a continuous limit when the dry reference is zero. Root extraction in layer i is bounded by both `root_fraction_i × requested_transpiration × h` and `dz_i max(θ_i−d_i, 0) r`. This preserves the continuous three-day relaxation accepted on 18 October. With ample transpiration demand and no other fluxes, `θ_next = d_i + (θ_i − d_i) exp(−h/τ)` for states above d_i. Extraction stops at d_i; evaporation and hydraulic redistribution can continue below it. Initially lower states receive no restoration water. The older 1.25-times trigger and post-update floor restoration remain retired.

Physical WP remains available for diagnostics and calculation of `FC = WP + available_water_capacity`. V3 deliberately replaces the WP extraction cutoff in the upper three layers with the empirical d_i profile. This changes the root-extraction assumption. A surface moisture percentile does not independently establish plant access to water below the hydraulic WP estimate; physiological interpretation and improved field skill require separate evidence.

## Assessment of the upper and lower limits

Hydraulic saturation/porosity remains the physical upper bound in every layer. Saturation represents filled pore space, as described in the [FAO drainage guidance](https://www.fao.org/4/R4082E/r4082e07.htm). A GSSM wet percentile is an observed climatological statistic. It can differ from saturation through sampling, scale and retrieval effects; it does not provide a direct estimate of pore volume. Therefore no GSSM multiplier expands or reduces the physical upper bound. Saturation excess continues through the conservative overflow pathway.

WP is a plant-water extraction threshold and cannot represent a universal lower bound on total soil moisture. The [FAO soil-water discussion](https://www.fao.org/4/a0072e/a0072e07.htm) explicitly describes continued evaporation below permanent wilting point. The [FAO transpiration treatment](https://www.fao.org/4/x0490e/x0490e0e.htm) separately associates WP with cessation of root uptake. This distinction supports allowing soil moisture below WP, while the chosen reduction of the model extraction threshold remains an empirical development assumption. Physical model bounds remain zero to porosity; zero is a non-negativity limit and is not presented as a calibrated air-dry water content.

Hydraulic saturation ratio is `S = θ/θs`, including the zero-water limit. Conductivity is `K = Ksat S^(2b+3)` in mm day⁻¹ and diffusivity is `D = diff_factor Ksat S^(b+2)` in mm² day⁻¹. The inherited interface discretisation uses the upper-layer coefficients: downward flux is `K_i + D_i(θ_i−θ_(i+1))/distance`. Bottom drainage is the smaller of the fixed drainage-slope times bottom-layer K and the fixed drainage upper limit. The lower drainage limit must be zero. An exactly empty upper layer has zero interface coefficients under this inherited discretisation; upward replenishment in that limiting state requires subsequent wetting. This is a structural limitation to retain in scientific interpretation.

The finite-volume update shares each donor's water across all proposed transfers and E/T sinks. Incoming hydraulic transfers share the receiver's pre-step capacity. A limited interface volume is applied with equal magnitude and opposite sign to neighbouring layers. Rainfall excess passes conservatively through successive saturated layers and leaves the bottom as separately reported overflow. Overflow represents the profile storage constraint; catchment routing is outside this model.

The solver makes exactly one bounded update per daily forcing record, with no hydraulic subcycling or hidden CFL steps. Hydraulic rates and extraction caps use the beginning-of-day state. All sinks and transfers are jointly limited before applying the daily rainfall and conservative saturation cascade. Consequently, new rainfall can affect storage and overflow that day, but supplies extraction and ordinary redistribution from the following day's state. This ordering is an explicit daily approximation; storm timing is unavailable from daily forcing.

This is a discrete daily water-balance formulation. Donor and receiver limits preserve admissible storage and exact flux accounting; they do not establish timestep convergence or accuracy of the daily flux partition. Strong hydraulic gradients can make these bounds active and can produce different redistribution from a finely integrated model. The previous `swb-conservative-2p-v1` used up to 1/768-day steps to meet a numerical refinement criterion. That improved resolution of its within-day equations, not mass closure or forcing information. V2 intentionally changes the integration contract and requires recalibration. Direct calls can specify a different forcing interval through `time_step`, with one update per interval; configured workflows remain daily. Substep overrides are rejected. The interpreted and Numba backends execute the same algorithm without silent fallback.

## Outputs and calibration identity

Every interval retains requested and actual E/T, unmet E/T, bottom drainage, saturation overflow, profile and layer mass residuals, signed interface transfers, layer transpiration and update counts (the compatibility variable `substeps` equals one on each valid day). Exposure diagnostics count beginning-of-day states below physical WP and below the extraction threshold in every layer, and above the wet reference in the first two layers; they do not resolve within-day crossings. Column metadata retains the full extraction-threshold and physical-WP arrays. Calibration and product identities record the extraction profile and physical upper-bound policy. The empirical Gd/Gw envelope is distinct from the transferred d/w references and can be evaluated using the frozen artifact. The compatibility field `storage_adjustment` is identically zero.

The profile identity is `ΔW = P_eff − E_actual − T_actual − drainage − overflow`. Layer identities use the same interface volumes on either side. Missing forcing holds the state, sets the forcing-validity flag to zero and leaves interval budgets missing. Start/end state timestamps and the final restart state are retained.

Calibration preserves the area-weighted native GSSM cell–day RMSE, fixed support, initialisation exclusion and coarse-to-fine transfer semantics. Seeds, optimizer populations, polishing, CSV output and downstream loading use exactly two fitted coefficients. Calibration CSVs and cache identities include formulation version, reference identity and fixed numerical settings. Daily-v2, subcycled-v1, four-parameter and unversioned calibrations raise explicit migration errors. Unmodified p01/p99 reference artifacts from v1 and v2 remain valid because their observations, hydraulic support and scalar definitions are unchanged; their hashes are preserved. Migration requires recalibrating with this formulation and generating new trajectories; previous fits cannot be relabelled.

## Validation and interpretation

Validation combines independent analytical drydown limits, physical bounds, layer/profile mass balances, extreme forcing, an independently calculated daily transfer, fixed-τ sensitivity, actual compiled execution, restarts, missing forcing and optimizer execution. Software validation and bounded real-forcing checks establish numerical behaviour within their tested cases. Production calibration and complete fine-grid reruns require separate execution and acceptance, including complete-forcing and budget gates. Wet/dry or repeated-cycle convergence gates apply only when that additional protocol is explicitly configured.

GSSM supplies reference statistics and calibration targets. In-situ observations support evaluation only. Previously inspected 2021–2025 observations provide retrospective evaluation of this revision. Native-cell estimates, point observations and fine-grid soil columns have distinct supports; successful numerical checks do not establish observational skill or hydraulic identifiability.


## Default initialisation with actual forcing

Start each layer at physical field capacity on the first available simulation date. For a Landsat 8/9 record starting in 2013, in both calibration and normal runs, use the available actual 2013 daily forcing as warm-up, carry the resulting state into 2014 without resetting, and score from 2014-01-01. Do not prepend synthetic years. Omit the optional `[spinup]` table or leave `spinup.enabled = false` to select this pathway. Repetition requires `spinup.enabled = true`. The warm-up is excluded from the GSSM objective, while its forcing still affects subsequent states. Coverage must be checked. This protocol retains field capacity as an initial-state assumption and does not claim independence from every possible initial profile.

Resolved plans and paired outputs record the initialization policy. The output `is_warmup` time flag separates the retained actual 2013 record from analysis dates, including when calibration is disabled. A normal run ending in 2013 is explicitly warm-up only. Later-starting studies and explicitly selected historical missions retain their requested dates.
