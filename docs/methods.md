# Methods and scientific limits

## Process chain

SSEBop uses Landsat surface temperature, vegetation information and meteorological reference evapotranspiration to estimate ET fraction and daily ET. The maintained local cold-temperature/FANO correction uses the existing configurable NDVI thresholds and spatial scales in `pysweb.ssebop.core`. Daily NDVI partitions ET using `f_t = clip(1.26 * NDVI - 0.18, 0, 1)`, `E_t = f_t * ET_a`, and `E_s = ET_a - E_t`. The forcing identity is `ET_a = ET_f * ET_0 = E_s + E_t`; NetCDF total ET retains its `ET`/`et` names. This empirical partition requires evaluation for the vegetation and season being studied.

The soil model uses conservative finite volumes with physical storage bounds, beta-distributed root uptake and a bottom drainage boundary. Evaporation draws from the surface layer and can continue below WP; root extraction stops at WP. A fixed three-day supply closure limits E/T, and joint donor constraints prevent transfers and sinks from spending the same water. Physical field capacity initialises the profile. The exact equations, frozen GSSM reference derivation, subcycling and migration contract are recorded in [the formulation specification](swb-formulation.md).

Separate `actual_evaporation`, `actual_transpiration`, `unmet_evaporation` and `unmet_transpiration` budgets distinguish forcing demand from water-limited extraction. `actual_et` is their realised E/T sum. The `E_s` and `E_t` SEB variables retain their forcing meanings. No interception store or flux is introduced.

OpenLandMap clay and sand predictors are percentages. Organic carbon uses the asset scale of 5 g kg⁻¹ per encoded unit. The existing mapping samples bands at 0, 10, 30, 60 and 100 cm and assigns them to model bottoms at 50, 150, 300, 600 and 1000 mm. These source point-depth predictions and model layer averages have different vertical support. Missing organic carbon uses the existing 5 g kg⁻¹ fallback, recorded in preprocessing configuration; clay/sand gaps remain missing. The local MLConstraints organic-matter convention is documented separately in [soil inputs](soil-inputs.md).

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
