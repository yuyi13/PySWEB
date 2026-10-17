# Methods and scientific limits

## Process chain

SSEBop uses Landsat surface temperature, vegetation information and meteorological reference evapotranspiration to estimate ET fraction and daily ET. The maintained local cold-temperature/FANO correction uses the existing configurable NDVI thresholds and spatial scales in `pysweb.ssebop.core`. Daily NDVI partitions ET using `f_t = clip(1.26 * NDVI - 0.18, 0, 1)`, `E_t = f_t * ET_a`, and `E_s = ET_a - E_t`. The forcing identity is `ET_a = ET_f * ET_0 = E_s + E_t`; NetCDF total ET retains its `ET`/`et` names. This empirical partition requires evaluation for the vegetation and season being studied.

The soil model uses layered hydraulic properties, a tridiagonal update for vertical redistribution, a limited explicit diffusive contribution, Jackson-style beta root uptake and a bottom drainage boundary. Evaporation draws from the surface layer, transpiration follows the root profile, and actual extraction is limited by available storage. Its dry-end relaxation and saturation constraints remain part of the implemented numerical method. The code and parameter manifests define the exact variant; this repository does not claim equivalence to a fully coupled Richards/heat transport model.

The requested top-layer extraction is `E_s + root_fraction[0] * E_t`; deeper layers receive `root_fraction[i] * E_t`. The solver exports the sum of water-limited extraction as `actual_et`. The `E_s` and `E_t` outputs are SEB forcing components; separate actual soil-limited component budgets are unavailable. No interception store or flux is introduced.

OpenLandMap clay and sand predictors are percentages. Organic carbon uses the asset scale of 5 g kg⁻¹ per encoded unit. The existing mapping samples bands at 0, 10, 30, 60 and 100 cm and assigns them to model bottoms at 50, 150, 300, 600 and 1000 mm. These source point-depth predictions and model layer averages have different vertical support. Missing organic carbon uses the existing 5 g kg⁻¹ fallback, recorded in preprocessing configuration; clay/sand gaps remain missing. The local MLConstraints organic-matter convention is documented separately in [soil inputs](soil-inputs.md).

## Calendars, storage and closure

Effective rain uses the repository's monthly Smith calculation. Preprocessing requires every daily timestamp in each intersecting calendar month, computes monthly effective totals and distributes them proportionally to daily precipitation, then selects the requested simulation dates. Any missing pixel-day makes that pixel's effective precipitation missing for the month. This avoids interpreting an incomplete month as a complete rainfall total.

All gridded SWB forcing arrays must share exact spatial coordinates and complete daily timestamps. A missing timestamp raises an error. A present day with missing forcing freezes the cell state and leaves its daily budgets missing; `valid_forcing_fraction` records that distinction. The legacy `nan_to_zero` option explicitly imputes zeros and is recorded in provenance. Screen QC when analysing outputs; frozen states do not constitute simulated observations.

`state_timing="start"` reports the state before each day's forcing. `state_timing="end"` reports the updated state. Final-layer states and their timestamps are always exported, allowing the last update to be examined; direct solver calls support restart from `final_state`. Gridded restart ingestion is not yet implemented.

For each simulated day, the ledger checks (all terms in mm per step):

```text
storage_change = infiltration - actual_et - drainage - overflow + storage_adjustment
balance_residual = storage_change - (infiltration - actual_et - drainage - overflow + storage_adjustment)
```

In manuscript notation, `Delta W = (P_e - ET_a - Q) * Delta t - O + A + residual`, where `Q` is bottom drainage, `q` denotes internal vertical flux, `O` is overflow and `A` is the lower-bound storage adjustment. Temperature uses uppercase `T` (including `T_cold`); `Delta T` denotes thermal separation and `Delta t` the time step. Existing descriptive temperature, drainage and time-step code identifiers retain their meanings.

Saturation overflow is water removed at the profile boundary by the storage constraint. It is not a routed catchment runoff estimate. Storage adjustment records floor corrections explicitly; a small ledger residual can coexist with a scientifically material adjustment. Evaluate both terms. One-layer drainage is now reported consistently with the sink used in the state update.

## Calibration support

Calibration uses seeded differential evolution and area-weighted cell–day RMSE on the native GSSM grid. Its denominator is the sum of cell areas over valid cell–days. The same fixed observation/soil/complete-forcing population is used for every candidate; a solver failure penalizes a candidate instead of removing difficult cells. Regular geographic cell areas are proportional to cosine latitude. The previous daily domain-mean loss remains available as `--objective domain_mean` for explicit legacy component runs.

Whole GSSM cells covering the AOI are retained, including their complete footprints outside the nominal chip. Continuous predictors are aggregated with area-overlap averaging; the reference is an exact native-grid subset. The final default and calibrated simulations use the selected Landsat UTM grid. The four parameters are domain scalars. Coarse-to-fine transfer is approximate because the SWB solver and pedotransfer functions are nonlinear: fewer cells reduce work but do not guarantee smaller errors. The development-only transfer diagnostic aggregates fine outputs back onto complete GSSM footprints, reports its own support, and never retunes parameters.

OpenLandMap texture/SOC predictors are aligned before the existing pedotransfer calculation. Thus the coarse product uses hydraulic functions of aggregated predictors; it is not numerically identical to aggregated fine-grid hydraulics. This approximation and the existing vertical support differences require scientific evaluation.

Spin-up days are simulated but excluded from the objective. Choose an adequate period for the profile and climate; the configurable value is not a validated universal duration. The initial state is the midpoint of configured storage bounds. Existing `sm_max_factor` bounds permit storage above nominal porosity; these empirical bounds are retained for compatibility and should be reviewed against physical observations.

## Priorities for scientific validation

1. Benchmark storage changes, fluxes and numerical adjustments against analytical or trusted column-model cases before changing solver equations or hydraulic coefficients.
2. Validate surface and deeper layers independently using measured soil moisture, with clearly matched depth and timestamp support. Hold out sites and periods; separate calibration from testing.
3. Compare measured/custom hydraulics with OpenLandMap and MLConstraints under the same forcing and validation population. Audit pedotransfer units and layer mapping before attributing differences to data quality.
4. Quantify sensitivity to ET partitioning, initial state, spin-up, root profile and source-data uncertainty. Report wet/dry and seasonal errors alongside aggregate RMSE.
5. Validate SSEBop against independent flux or ET evidence. Coarse meteorology and reference soil moisture do not become fine-resolution observations through resampling.

The synthetic demo and regression tests establish software contracts and numerical bookkeeping. They do not replace these observational evaluations. Scientific equations and references in package comments remain the implementation record; the superseded planning documents are recoverable from Git history.
