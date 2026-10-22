# Methods and scientific limits

## Process chain

SSEBop uses Landsat surface temperature, vegetation information and meteorological reference evapotranspiration to estimate ET fraction and daily ET. The default cold boundary is the WorldCover FANO adaptation described below; the previous local correction is available with `tcold_method="legacy"`. Daily NDVI partitions ET using `f_t = clip(1.26 * NDVI - 0.18, 0, 1)`, `E_t = f_t * ET_a`, and `E_s = ET_a - E_t`. The forcing identity is `ET_a = ET_f * ET_0 = E_s + E_t`; NetCDF total ET retains its `ET`/`et` names. This empirical partition requires evaluation for the vegetation and season being studied.

The soil model uses conservative finite volumes with physical storage bounds, beta-distributed root uptake and a bottom drainage boundary. Evaporation draws from the surface layer and can continue below WP; root extraction stops at WP. A fixed three-day supply closure limits E/T, and joint donor constraints prevent transfers and sinks from spending the same water. Physical field capacity initialises the profile. The exact equations, frozen GSSM reference derivation, daily update and migration contract are recorded in [the formulation specification](swb-formulation.md).

Separate `actual_evaporation`, `actual_transpiration`, `unmet_evaporation` and `unmet_transpiration` budgets distinguish forcing demand from water-limited extraction. `actual_et` is their realised E/T sum. The `E_s` and `E_t` SEB variables retain their forcing meanings. No interception store or flux is introduced.

OpenLandMap clay and sand predictors are percentages. Organic carbon uses the asset scale of 5 g kg⁻¹ per encoded unit. The existing mapping samples bands at 0, 10, 30, 60 and 100 cm and assigns them to model bottoms at 50, 150, 300, 600 and 1000 mm. These source point-depth predictions and model layer averages have different vertical support. Missing organic carbon uses the existing 5 g kg⁻¹ fallback, recorded in preprocessing configuration; clay/sand gaps remain missing. The local MLConstraints organic-matter convention is documented separately in [soil inputs](soil-inputs.md).

## Land-cover-aware FANO cold boundary

`pysweb.ssebop.fano.worldcover_fano` follows the fine/coarse and mixed-landscape logic in [OpenET SSEBop](https://github.com/Open-ET/openet-ssebop/blob/main/openet/ssebop/image.py), inspected 22 October 2026 (Git blob `5aef6a7a3833679a2a1598ef33b2e7adbdc25cdd`). The implementation is an Australian land-cover adaptation, not an exact reproduction of OpenET ET products.

The input classification is [ESA WorldCover v200](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200), band `Map`, a global 10 m **2021** map. It is static for every Landsat date, including earlier years. Land-cover change and classification error therefore remain sources of uncertainty. Credit: Zanaga et al. (2022), [doi:10.5281/zenodo.7254221](https://doi.org/10.5281/zenodo.7254221), CC BY 4.0.

| WorldCover classes | FANO use |
| --- | --- |
| 30 grassland, 40 cropland, 90 herbaceous wetland | Agricultural branch; eligible for regional references when native NDVI ≥ 0.4 |
| 20 shrubland, 50 built-up, 60 bare/sparse vegetation | Local fine branch; anomalous negative-NDVI treatment |
| 10 tree cover, 70 snow/ice, 95 mangroves, 100 moss/lichen | Local fine branch; excluded from the agricultural reference mask |
| 80 permanent water | Excluded from terrestrial ET and all cold-reference means |
| Missing or unknown class | Error wherever the scene otherwise has valid LST, NDVI and positive dT |

This positive reference mask excludes more than forests. OpenET's NLCD mapping also includes developed open/low-intensity areas and woody wetlands. WorldCover cannot distinguish those developed subclasses; PySWEB treats all built-up land as non-agricultural and excludes mangroves from the reference group. This is a deliberate, conservative crosswalk.

1. Average valid, non-water LST, NDVI and dT onto a 240 m grid. Compute `Tc_fine = LST_fine - 0.125 × dT_fine × (0.9 − NDVI_fine) × 10`.
2. Sample the native green agricultural mask onto the fine grid by nearest neighbour. Aggregate the eligible fine LST/NDVI cells to 4800 m and apply the same expression with coarse dT. As in the OpenET structure, fine means can mix neighbouring land covers at class boundaries; exclusion from the reference mask does not make mixed fine cells pure crop observations.
3. Apply a 3 × 3 coarse mean, fill remaining holes with the mean of all valid fine Tc cells over the supplied scene extent, then apply another 3 × 3 coarse mean. The scene fallback can include non-agricultural land and is explicitly tracked.
4. At 90 m, use local fine Tc for green agricultural cells (fine NDVI ≥ 0.4) and for non-agricultural land. Dry agricultural cells use the regional estimate. Smooth the combined field with a square **radius of 5 pixels (11 × 11)**, then sample back to native Landsat pixels. Negative NDVI uses native LST, except anomalous shrub/bare/built classes use 250 K. Water remains missing.

The projected grid origin is `(15, 15)` at every scale, matching OpenET's transforms; changing a crop window does not reset the grid origin. CRS must be north-up and in metres. Continuous reductions use full area-weighted raster means, not Earth Engine's `bestEffort` pyramid reductions. Categorical fields and final grid sampling use nearest neighbour. PySWEB averages dT at each working scale and retains its existing meteorology, Landsat temperature scaling and QA. It does not add OpenET's separate emissivity correction or its NDWI/JRC surface-water logic. These differences prevent claims of numerical parity with OpenET.

Reference extent is a spatial-support decision, not a multiple of AOI area. Two coarse means can depend on a 5 × 5 set of original coarse cells (24 km across), plus final smoothing; the fallback depends on the whole supplied scene. Native workflows default to 14.4 km acquisition padding, but this is not a proof of sufficient coverage for every scene. Configured retrieval protocols continue to compare two explicit padding windows and reject fallback-dependent or unstable estimates using their stated thresholds. `cold_boundary_fallback_fraction` now follows the actual branch selection and smoothing weights. `worldcover_class` and `fano_reference_eligible` accompany scene temperatures and ET fractions in diagnostics.

Removing forest references does not guarantee positive ET: the selected reference may still place a hot, dry field above the hot boundary. Independent ET evaluation and padding checks remain necessary before interpreting a changed fraction as an improvement.

## Calendars, storage and closure

Effective rain uses the repository's monthly Smith calculation. Preprocessing requires every daily timestamp in each intersecting calendar month, computes monthly effective totals and distributes them proportionally to daily precipitation, then selects the requested simulation dates. Any missing pixel-day makes that pixel's effective precipitation missing for the month. This avoids interpreting an incomplete month as a complete rainfall total.

All gridded SWB forcing arrays must share exact spatial coordinates and complete daily timestamps. A missing timestamp raises an error. A present day with missing forcing freezes the cell state and leaves its daily budgets missing; `valid_forcing_fraction` records that distinction. The legacy `nan_to_zero` option explicitly imputes zeros and is recorded in provenance. Screen QC when analysing outputs; frozen states do not constitute simulated observations.

`state_timing="start"` reports the state before each day's forcing. `state_timing="end"` reports the updated state. Final-layer states and their timestamps are always exported, allowing the last update to be examined; direct solver calls support restart from `final_state`. Gridded restart ingestion is not yet implemented.

For each simulated day, the ledger checks (all terms in mm per step):

```text
storage_change = infiltration - actual_et - drainage - overflow
balance_residual = storage_change - (infiltration - actual_et - drainage - overflow)
```

The compatibility variable `storage_adjustment` is identically zero. Internal interface transfers cancel in the profile balance and remain available for independent layer checks. Saturation excess cascades conservatively to the bottom boundary as `overflow`; this quantity is a column storage diagnostic. It does not represent routed catchment runoff. Temperature uses uppercase `T`; time-step duration uses `Delta t`.

## Calibration support

Calibration uses seeded differential evolution and area-weighted cell–day RMSE on the native GSSM grid. Its denominator is the sum of cell areas over valid cell–days. The same fixed observation/soil/complete-forcing population is used for every candidate; a solver failure penalizes a candidate instead of removing difficult cells. Regular geographic cell areas are proportional to cosine latitude. The previous daily domain-mean loss remains available as `--objective domain_mean` for explicit legacy component runs.

Whole GSSM cells covering the AOI are retained, including their complete footprints outside the nominal chip. Continuous predictors are aggregated with area-overlap averaging; the reference is an exact native-grid subset. The final default and calibrated simulations use the selected Landsat UTM grid. The two fitted parameters are domain scalars. The frozen GSSM reference artifact supplies fixed shallow dry/wet references. Coarse-to-fine transfer is approximate because the SWB solver and pedotransfer functions are nonlinear: fewer cells reduce work but do not guarantee smaller errors. The development-only transfer diagnostic aggregates fine outputs back onto complete GSSM footprints, reports its own support, and never retunes parameters.

OpenLandMap texture/SOC predictors are aligned before the existing pedotransfer calculation. Thus the coarse product uses hydraulic functions of aggregated predictors; it is not numerically identical to aggregated fine-grid hydraulics. This approximation and the existing vertical support differences require scientific evaluation.

Spin-up days are simulated but excluded from the objective. Choose an adequate period for the profile and climate; the configurable value is not a validated universal duration. The initial state is physical field capacity. Wet/dry convergence gates compare zero and physical-saturation initial states and retain their configured coverage and tolerance requirements. Dry forcing can preserve deep initial-state differences; a failed gate requires further development-only preparation and remains a failed gate.

## Priorities for scientific validation

1. Benchmark storage changes, fluxes and numerical adjustments against analytical or trusted column-model cases before changing solver equations or hydraulic coefficients.
2. Validate surface and deeper layers independently using measured soil moisture, with clearly matched depth and timestamp support. Hold out sites and periods; separate calibration from testing.
3. Compare measured/custom hydraulics with OpenLandMap and MLConstraints under the same forcing and validation population. Audit pedotransfer units and layer mapping before attributing differences to data quality.
4. Quantify sensitivity to ET partitioning, initial state, spin-up, root profile and source-data uncertainty. Report wet/dry and seasonal errors alongside aggregate RMSE.
5. Validate SSEBop against independent flux or ET evidence. Coarse meteorology and reference soil moisture do not become fine-resolution observations through resampling.

The synthetic demo and regression tests establish software contracts and numerical bookkeeping. They do not replace these observational evaluations. Scientific equations and references in package comments remain the implementation record; the superseded planning documents are recoverable from Git history.
