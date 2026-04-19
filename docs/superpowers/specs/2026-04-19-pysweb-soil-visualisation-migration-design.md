# PySWEB Soil Package, Visualisation Migration, and Notebook Alignment Design

Date: 2026-04-19
Repository: `/g/data/ym05/github/PySWEB`
Status: Proposed and user-approved for planning

## Goal

Coordinate the next package-first refactor so that:

- soil-data acquisition moves out of `pysweb.swb.preprocess` into a dedicated `pysweb.soil` package
- `openlandmap` becomes the only implemented soil backend in this change
- `mlcons`, `slga`, and `custom` are retained as explicit future backends with a stable package surface
- plotting code moves from `visualisation/` into `pysweb.visualisation`
- legacy `visualisation/*.py` entrypoints remain as thin shims for one transition round
- `notebooks/01_run_pysweb.ipynb` reflects the current package-backed SSEBop and SWB workflow
- `README.md` and `notebooks/README.md` describe the current notebook inventory and canonical package paths

This design builds on the existing SWB global-soil and `reference_ssm` migration. It does not replace that work. It isolates soil sourcing as its own reusable package layer and finishes the plotting migration into `pysweb`.

## Scope

This design covers:

- introduction of a new top-level `pysweb.soil` package
- extraction of current OpenLandMap soil logic from `pysweb/swb/preprocess.py`
- a stable soil-backend selection interface for SWB preprocess
- placeholder package surfaces for `mlcons`, `slga`, and `custom`
- migration of plotting modules into `pysweb.visualisation`
- one-transition-round wrappers in `visualisation/`
- updates to `notebooks/01_run_pysweb.ipynb`
- README and notebook-documentation corrections tied to the new canonical package paths
- targeted test restructuring for soil and visualisation boundaries

This design does not cover:

- implementation of `mlcons`, `slga`, or user-supplied soil datasets
- removal of the legacy `visualisation/` directory in this same change
- redesign of the SWB soil hydraulic calculations
- a general plugin or registry system for arbitrary third-party soil backends

## Core Decisions

### Canonical soil boundary

Soil acquisition and source-specific transformations move into a new top-level package:

```text
pysweb/
  soil/
```

This mirrors the architectural role of `pysweb.met`: SWB preprocess consumes a stable package interface rather than encoding source-specific logic internally.

`pysweb.swb.preprocess` remains responsible for preprocess orchestration, output naming, grid construction, forcing preparation, and `reference_ssm` handling. It stops owning the details of:

- Earth Engine OpenLandMap dataset identifiers
- source-specific depth-band mapping
- source-specific predictor loading
- source-specific validation for future soil backends

### Single implemented backend now

Only one soil backend is implemented in this change:

- `soil_source = "openlandmap"`

The package also exposes the following future backend namespaces:

- `pysweb.soil.mlcons`
- `pysweb.soil.slga`
- `pysweb.soil.custom`

These placeholders exist to make the target package structure explicit. They are not operational in this change. If selected through the public soil-source interface, they must fail immediately with clear “not implemented yet” errors.

### Future compatibility with user soil data

The package boundary must be designed so that a future `custom` backend can ingest user-prepared soil datasets without changing SWB preprocess consumers.

That means the public contract is dataset-oriented, not source-oriented: backends must return or materialize the standard SWB hydraulic outputs and profile metadata expected by downstream code, regardless of whether the source is OpenLandMap, MLConstraints, SLGA, or user-provided rasters.

### Canonical visualisation boundary

Plotting logic moves into:

```text
pysweb/
  visualisation/
```

The canonical modules become:

- `pysweb.visualisation.plot_time_series`
- `pysweb.visualisation.plot_heatmap`

The legacy entrypoints:

- `visualisation/plot_time_series.py`
- `visualisation/plot_heatmap.py`

remain for one transition round as thin wrappers over the new package modules.

### Notebook and documentation alignment

`notebooks/01_run_pysweb.ipynb` becomes the canonical model-run example and must describe the package-backed workflow that now exists. Documentation must stop implying that:

- SWB preprocess and calibration still live only in workflow scripts
- the notebook inventory uses the older file names
- `visualisation/` is the only or preferred plotting location

## Target Module Layout

```text
pysweb/
  soil/
    __init__.py
    api.py
    openlandmap.py
    mlcons.py
    slga.py
    custom.py
  visualisation/
    __init__.py
    plot_time_series.py
    plot_heatmap.py
```

The exact split between `__init__.py` and `api.py` can follow the existing package style, but the public boundary must support:

- explicit package-level access to available backend modules
- one dispatcher or loader entrypoint used by SWB preprocess

Recommended package-level behavior:

- `pysweb.soil.openlandmap` imports the implemented backend
- `pysweb.soil.mlcons`, `pysweb.soil.slga`, and `pysweb.soil.custom` import placeholder modules
- `pysweb.soil.load(...)` or equivalent dispatcher resolves `soil_source`

`pysweb.visualisation` should similarly expose the package-backed plotting modules without requiring consumers to import from the legacy top-level directory.

## Public Interfaces

### Soil selection interface

`pysweb.swb.preprocess(...)` should gain a neutral soil-source selection surface rather than embedding OpenLandMap assumptions in its body.

Required public behavior:

- `soil_source` defaults to `"openlandmap"`
- unsupported values fail early with a list of supported choices
- known but unimplemented values fail with clear `NotImplementedError`-style messages

Expected source values:

- `"openlandmap"`: implemented
- `"mlcons"`: declared, not implemented
- `"slga"`: declared, not implemented
- `"custom"`: declared, not implemented

### Backend-specific argument shape

Backend-specific inputs should use prefixed argument names rather than overloading the shared surface. The only fully active backend-specific controls in this change are the OpenLandMap ones.

Examples of the intended surface:

- `soil_source`
- `soil_openlandmap_project`
- `soil_custom_dir`
- `soil_mlcons_dir`
- `soil_slga_dir`

Implementation rule for this change:

- wire only the `openlandmap` arguments end to end
- reserve the `mlcons`, `slga`, and `custom` argument names for future use
- do not expose inactive backend-specific directory arguments in the public CLI until their backends exist

### Standard soil output contract

The dispatcher must return data in a stable SWB-oriented contract. Source-specific differences are internal to the backend.

The standard derived outputs remain:

- `porosity`
- `wilting_point`
- `available_water_capacity`
- `b_coefficient`
- `conductivity_sat`

The backend must also preserve the layer metadata needed by SWB, including the current five-layer profile bottoms:

- `50 mm`
- `150 mm`
- `300 mm`
- `600 mm`
- `1000 mm`

This contract is the future target for `custom` as well. A user-supplied backend should eventually be able to satisfy the same outputs without forcing changes in `pysweb.swb.preprocess` or `pysweb.swb.run`.

## Data Flow

### SWB preprocess orchestration

`pysweb.swb.preprocess(...)` remains the orchestrator and should perform the following high-level sequence:

1. Read precipitation inputs and compute daily effective precipitation.
2. Read ET and transpiration inputs.
3. Build one target grid and reuse it for all preprocess outputs.
4. Resolve the requested soil backend through `pysweb.soil`.
5. Ask the selected backend to produce the standard soil hydraulic outputs on the target grid.
6. Load and align `reference_ssm` using the existing package-backed reference-SSM path.
7. Write NetCDF outputs using the established SWB preprocess file contract.

The change in this refactor is step 4 and step 5: SWB preprocess becomes a consumer of soil backends rather than the owner of OpenLandMap implementation details.

### OpenLandMap backend behavior

`pysweb.soil.openlandmap` owns:

- Earth Engine dataset identifiers for clay, sand, and SOC
- OpenLandMap export scale choices
- depth-band selection
- the approved depth mapping from OpenLandMap depths into the SWB layer contract
- source-specific predictor validation before hydraulic derivation

The backend-specific depth mapping remains:

- `OpenLandMap 0 cm -> SWB 5 cm`
- `OpenLandMap 10 cm -> SWB 15 cm`
- `OpenLandMap 30 cm -> SWB 30 cm`
- `OpenLandMap 60 cm -> SWB 60 cm`
- `OpenLandMap 100 cm -> SWB 100 cm`

This mapping is source-specific and must not remain embedded in general SWB orchestration code.

### Placeholder backend behavior

`pysweb.soil.mlcons`, `pysweb.soil.slga`, and `pysweb.soil.custom` should document their intended role through module docstrings and explicit stubs.

Public behavior for this change:

- importing the modules succeeds
- selecting them as active backends fails immediately
- failure messages explain that the backend surface is reserved for future implementation

This makes the architecture explicit without pretending these backends already work.

## Visualisation Migration

### Canonical modules

The real plotting logic moves from the legacy directory into:

- `pysweb.visualisation.plot_time_series`
- `pysweb.visualisation.plot_heatmap`

Internal coupling should be cleaned up during the move so package modules import each other through package paths rather than sibling top-level imports. In particular, the current direct dependency from `plot_heatmap.py` to `plot_time_series.py` should be rewritten to use the new package path.

### One-transition-round wrappers

The legacy scripts remain for compatibility, but only as shims that:

- preserve the current CLI surface
- forward execution to the new package modules
- avoid duplicating implementation logic

The top-level `visualisation/` directory is therefore transitional after this change, not canonical. Its removal is deferred to a later cleanup round once downstream usage has moved over.

## Notebook And Documentation Updates

### `notebooks/01_run_pysweb.ipynb`

The notebook should be updated so that its narrative and example cells match the current package-first state of the repo.

Required updates:

- describe `01_run_pysweb.ipynb` as the model-run example
- stop stating that SWB preprocess and calibration are only workflow-driven
- show package-backed SWB steps through `pysweb.swb.preprocess(...)`, `pysweb.swb.calibrate(...)`, and `pysweb.swb.run(...)`
- keep execution toggles conservative so the notebook remains safe to open without launching downloads or heavy processing

The notebook does not need to demonstrate future `mlcons`, `slga`, or `custom` selection. It should show the implemented default path.

### README corrections

`README.md` and `notebooks/README.md` should be updated to:

- list the actual notebook file names:
  - `01_run_pysweb.ipynb`
  - `02_plot_heatmap.ipynb`
  - `03_plot_time_series.ipynb`
- identify `01_run_pysweb.ipynb` as the run example
- point plotting references toward `pysweb.visualisation`
- mention that legacy `visualisation/*.py` wrappers remain available during the transition
- describe `pysweb.soil` as the new canonical location for soil-source logic

## Error Handling

The refactor should preserve strict, early failures where ambiguity would make debugging difficult.

Required behavior:

- unknown `soil_source` values fail before any backend work starts
- placeholder backends fail with clear, direct “not implemented yet” messages
- OpenLandMap loading fails clearly on missing bands, invalid depth expectations, or empty exports
- grid alignment and layer-shape validation happen before NetCDF outputs are written
- SWB preprocess should continue surfacing actionable errors rather than silently skipping soil outputs

The `custom` backend must not be silently accepted with partial behavior. In this change it is a declared interface target only.

## Testing Strategy

Testing should separate backend behavior from SWB orchestration more cleanly than the current layout.

Recommended split:

- keep preprocess orchestration tests in `tests/swb/test_preprocess.py`
- create `tests/soil/` for backend dispatch and OpenLandMap-specific behavior
- add placeholder-backend tests that assert early failure for `mlcons`, `slga`, and `custom`
- add visualisation migration tests that confirm the legacy scripts still forward to the package modules

The intent is not to eliminate SWB preprocess tests, but to stop treating source-specific soil logic as an internal detail of SWB orchestration.

## Migration Strategy

This refactor is a coordinated move, but not a hard cut.

End-state expectations for this change:

- canonical soil logic lives in `pysweb.soil`
- canonical plotting logic lives in `pysweb.visualisation`
- `pysweb.swb.preprocess` consumes `pysweb.soil`
- `visualisation/` remains only as a wrapper layer
- docs and notebooks point to the new canonical package paths

Deferred to a later cleanup change:

- removing the legacy `visualisation/` directory entirely
- implementing `mlcons`, `slga`, or `custom`
- broadening the public soil interface beyond the immediate SWB use case if future consumers need it

## Implementation Notes

This change should follow the existing package-first direction established by the SSEBop refactor and the SWB `reference_ssm` migration:

- reusable logic in `pysweb/`
- workflows and legacy scripts as thin interfaces
- data-source-specific code isolated behind stable package boundaries

The design intentionally favors a slightly stronger boundary now so that future work on:

- USYD-specific MLConstraints inputs
- Australian SLGA support
- user-supplied soil rasters

can be added as backends rather than as new branches inside `pysweb.swb.preprocess`.
