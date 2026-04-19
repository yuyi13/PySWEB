# PySWEB SWB Global Soil and Reference-SSM Migration Design

Date: 2026-04-19
Repository: `/g/data/ym05/github/PySWEB`
Status: Proposed and user-approved for planning

## Goal

Complete the remaining SWB package migration so that:

- `pysweb.swb.preprocess(...)` becomes the canonical implementation for SWB input preparation
- `pysweb.swb.calibrate(...)` becomes the canonical implementation for domain calibration
- `workflows/3_sweb_preprocess_inputs.py` and `workflows/4_sweb_calib_domain.py` become thin CLI wrappers over package code
- local Australia-specific SLGA soil inputs are replaced by global Google Earth Engine `OpenLandMap` soil inputs
- SMAP-DS-specific reference-soil-moisture handling is replaced by a neutral `reference_ssm` interface
- the default reference soil-moisture source becomes the Earth Engine `gssm1km` asset `users/qianrswaterr/GlobalSSM1km0509`

The design must stay consistent with the existing SSEBop package-first refactor: reusable logic belongs in `pysweb/`, and numbered workflow scripts remain parser-only entrypoints.

## Scope

This design covers:

- SWB preprocess packaging
- SWB calibration packaging
- global soil ingestion from Earth Engine
- neutral `reference_ssm` naming and file contracts
- GEE-backed `gssm1km` reference SSM ingestion
- workflow-wrapper updates
- tests and documentation updates needed for the migration

This design does not cover:

- adding multiple reference SSM backends in the same change
- preserving SMAP-DS as a first-class source
- changing the external behavior of `pysweb.swb.run(...)` beyond its existing soil/input contracts
- redesigning the SWB solver physics

## Core Decisions

### Canonical ownership

The canonical implementations for SWB preprocess and calibration move into:

- `pysweb/swb/preprocess.py`
- `pysweb/swb/calibrate.py`

The workflow scripts:

- `workflows/3_sweb_preprocess_inputs.py`
- `workflows/4_sweb_calib_domain.py`

remain available, but only as thin CLI wrappers that parse arguments and call the package functions.

### Reference SSM naming

The public interface stops using `smap_*` names because the default source is no longer SMAP-DS.

Neutral naming is used instead:

- `reference_ssm`
- `reference_var`
- `reference_source`
- `reference_ssm_asset`
- `skip_reference_ssm`

The preprocess output becomes:

- `reference_ssm_daily_<start>_<end>.nc`

The default variable name inside that file becomes:

- `reference_ssm`

### Single supported reference source for now

Only one reference SSM source is supported in this migration:

- `reference_source = "gssm1km"`

The argument remains explicit for future extensibility, but preprocess must reject anything other than `gssm1km` for now. There is no silent fallback to SMAP-DS or any other product.

### Earth Engine dependency placement

Earth Engine is required in preprocess, not in calibration.

`pysweb.swb.preprocess(...)` is responsible for:

- Earth Engine initialization
- pulling global soil predictors from `OpenLandMap`
- pulling the reference SSM product from `GlobalSSM1km0509`
- resampling both onto the SWB target grid
- writing NetCDF products for downstream use

`pysweb.swb.calibrate(...)` remains file-based and reads prepared NetCDF inputs only.

This keeps calibration reproducible, batch-friendly, and aligned with the current workflow architecture.

## Target Module Layout

```text
pysweb/
  swb/
    __init__.py
    api.py
    core.py
    preprocess.py
    calibrate.py
    run.py
```

### `pysweb.swb.api`

Exports real public entrypoints:

- `preprocess`
- `calibrate`
- `run`

`preprocess` and `calibrate` should stop raising `NotImplementedError` and forward to package-owned implementations.

### `pysweb.swb.preprocess`

Owns:

- parser construction for the preprocess CLI
- target-grid creation
- rainfall/effective-rainfall preparation
- ET and transpiration preparation
- OpenLandMap soil download and derivation
- `gssm1km` reference SSM download and rasterization
- writing SWB-ready NetCDF outputs

### `pysweb.swb.calibrate`

Owns:

- parser construction for the calibration CLI
- neutral `reference_ssm` argument handling
- reading preprocessed forcing, soil, NDVI, and reference SSM inputs
- optimization setup and summary CSV writing
- removal of direct dependency on `core/swb_model_1d.py` in favor of package-owned SWB logic

## Data Flow

### Preprocess

`pysweb.swb.preprocess(...)` should perform the following sequence:

1. Read precipitation inputs and compute daily effective precipitation.
2. Read ET and transpiration inputs.
3. Build one target grid and reuse it for all outputs.
4. Download soil predictor layers from Earth Engine `OpenLandMap`.
5. Derive SWB soil hydraulic properties on the target grid.
6. Download `gssm1km` daily surface soil moisture for the requested dates and extent.
7. Resample the reference SSM product to the target grid.
8. Write all outputs as NetCDF in the same directory contract currently expected by SWB run and calibration.

### Calibration

`pysweb.swb.calibrate(...)` should:

1. Read effective precipitation, ET, transpiration, soil NetCDFs, optional NDVI, and `reference_ssm`.
2. Align them to a common calibration domain and date range.
3. Compare simulated surface-layer soil moisture against `reference_ssm`.
4. Optimize the same parameter set currently used by calibration:
   - `diff_factor`
   - `sm_max_factor`
   - `sm_min_factor`
   - `root_beta`
5. Write the calibration CSV in the current style.

The run stage remains stable at its interface boundary. It continues consuming the prepared soil-property NetCDFs and forcing NetCDFs without requiring direct Earth Engine access.

## Soil Data Migration

### Replace SLGA with OpenLandMap

The current preprocess script derives soil properties from local SLGA texture and SOC rasters. That is not appropriate for a global `pysweb` implementation.

The replacement source is `OpenLandMap` in Google Earth Engine, using the global image products for:

- clay fraction
- sand fraction
- soil organic carbon

These products are stored as separate Earth Engine images with standard depth bands.

### Depth mapping

The current SWB preprocessing logic expects SLGA-style depth boundaries:

- 0 to 5 cm
- 5 to 15 cm
- 15 to 30 cm
- 30 to 60 cm
- 60 to 100 cm

`OpenLandMap` uses depths:

- 0 cm
- 10 cm
- 30 cm
- 60 cm
- 100 cm
- 200 cm

The approved mapping is:

- `OpenLandMap 0 cm` replaces SWB `5 cm`
- `OpenLandMap 10 cm` replaces SWB `15 cm`
- `OpenLandMap 30 cm` stays `30 cm`
- `OpenLandMap 60 cm` stays `60 cm`
- `OpenLandMap 100 cm` stays `100 cm`

The resulting SWB layer-bottom sequence remains:

- `50, 150, 300, 600, 1000 mm`

This preserves the existing SWB layering contract while adapting to the available global dataset.

### Derived soil properties

The preprocess stage should continue deriving the same hydraulic properties that the existing SWB path writes:

- `soil_porosity.nc`
- `soil_wilting_point.nc`
- `soil_available_water_capacity.nc`
- `soil_b_coefficient.nc`
- `soil_conductivity_sat.nc`

The internal pedotransfer calculations can remain structurally similar to the current workflow implementation, but they now consume Earth Engine-derived `OpenLandMap` predictors rather than local SLGA rasters.

## Reference SSM Migration

### Default source

The default reference SSM source becomes:

- `reference_source = "gssm1km"`
- `reference_ssm_asset = "users/qianrswaterr/GlobalSSM1km0509"`
- `gee_project = "yiyu-research"`

### Asset structure

The `GlobalSSM1km0509` asset is an Earth Engine `ImageCollection`, not a single image.

Observed structure:

- one image per year/continent tile
- band names encode dates
- example band name: `band_2000_03_05_classification`

The loader must therefore:

1. initialize Earth Engine with the provided GEE project
2. identify which collection image(s) intersect the requested extent
3. parse daily band names for the requested date range
4. extract matching daily bands
5. mosaic or select the needed tile coverage for the requested area
6. resample the output to the SWB target grid

### Units and scaling

The published `gssm1km` dataset description states that values are stored with a scale factor of `1000`.

Therefore:

- raw band values must be divided by `1000`
- the written NetCDF product should store volumetric soil moisture in `m3 m-3`

### Temporal and depth alignment

`gssm1km` is surface soil moisture for 0 to 5 cm depth. That is a suitable replacement for the current calibration reference target, which calibrates against surface observations using a default `surface_depth` of `50 mm`.

### Output contract

The preprocess output should be written as:

- `reference_ssm_daily_<start>_<end>.nc`

with:

- variable name `reference_ssm`
- dimensions `time`, `lat`, `lon`
- units `m3 m-3`

## Public Interface

### Preprocess interface

`pysweb.swb.preprocess(...)` and its CLI wrapper should use neutral reference-SSM arguments:

- `reference_source`
- `reference_ssm_asset`
- `gee_project`
- `skip_reference_ssm`

The default source is `gssm1km`, and only `gssm1km` is accepted in this migration.

The previous `--skip-smap` naming is retired and replaced by `--skip-reference-ssm`.

### Calibration interface

`pysweb.swb.calibrate(...)` and its CLI wrapper should replace:

- `--smap-ssm` with `--reference-ssm`
- `--smap-var` with `--reference-var`

The default variable name becomes:

- `reference_ssm`

Internal naming in calibration code should also be updated from `smap`/`smap_ssm` to `reference_ssm`.

### Workflow wrappers

`workflows/3_sweb_preprocess_inputs.py` should:

- construct its parser via `pysweb.swb.preprocess`
- call the package implementation
- remain only a CLI entrypoint

`workflows/4_sweb_calib_domain.py` should:

- construct its parser via `pysweb.swb.calibrate`
- call the package implementation
- remain only a CLI entrypoint

This mirrors the existing SSEBop refactor style already used for:

- `workflows/2_ssebop_run_model.py`
- `workflows/5_sweb_run_model.py`

## Error Handling

The new GEE-backed reference and soil path must fail clearly and early.

`pysweb.swb.preprocess(...)` should raise explicit errors when:

- Earth Engine cannot initialize with `gee_project`
- the `reference_ssm_asset` is unreadable
- the requested dates fall outside asset coverage
- no collection tile intersects the requested extent
- the required OpenLandMap datasets are unreadable

It must not silently fall back to another soil or reference product.

For supported dates where specific daily bands are missing, preprocess may:

- write `NaN` for missing dates
- emit a warning summary after processing

This is acceptable because calibration already works on the valid overlap between observations and model output.

## Testing Strategy

Tests should focus on the new interfaces and Earth Engine-specific assumptions without requiring live Earth Engine access in CI.

### Required coverage

- `pysweb.swb.api` exports real callable `preprocess`, `calibrate`, and `run`
- workflow wrappers delegate to package implementations
- calibration accepts `reference_ssm` / `reference_var` and no longer depends on `smap_*`
- `gssm1km` collection metadata and band-name parsing logic using mocked EE objects
- OpenLandMap depth mapping logic
- derived soil-property output shapes and layer metadata
- output naming changes from `smap_ssm` to `reference_ssm`

### Non-goals for tests

- no dependence on live Earth Engine credentials in automated tests
- no attempt to verify global Earth Engine downloads end-to-end in unit tests

## Documentation Updates

The README should be updated to reflect that:

- `pysweb.swb.preprocess` and `pysweb.swb.calibrate` are package-backed
- global soil inputs now come from Earth Engine `OpenLandMap`
- the default calibration reference is `gssm1km`, not SMAP-DS
- preprocess outputs `reference_ssm_daily_*.nc` instead of `smap_ssm_daily_*.nc`

## Consequences

This migration makes the SWB side consistent with the package-first SSEBop refactor by:

- moving reusable implementation into `pysweb`
- keeping workflow scripts thin
- removing Australia-only local soil dependencies from the main path
- removing SMAP-specific naming from interfaces that are no longer SMAP-specific
- making the global GEE-backed preprocessing pipeline the default preparation path for SWB

The main cost is a stronger preprocess dependency on Earth Engine and custom handling for the tiled, band-encoded `gssm1km` asset. That cost is acceptable because it replaces difficult-to-obtain local SMAP-DS inputs with a simpler, reproducible GEE-backed source.
