# PySWEB SSEBop `gee_project`, DEM Package, and Landsat Module Design

Date: 2026-04-20
Repository: `/g/data/ym05/github/PySWEB`
Status: Proposed and user-approved for planning

## Goal

Refine the package-backed SSEBop preparation surface so that:

- `gee_project` becomes an explicit public argument rather than a hidden hard-coded value
- DEM acquisition moves into a dedicated top-level `pysweb.dem` package alongside `pysweb.met` and `pysweb.soil`
- SSEBop `prepare_inputs` downloads and materializes a DEM during preparation instead of requiring users to provide a pre-existing local DEM file
- the default DEM source is `NASA/NASADEM_HGT/001`
- canonical Landsat preparation helpers move from `pysweb.ssebop.inputs.landsat` to `pysweb.ssebop.landsat`
- the run notebook and workflow documentation show the package-first, GEE-project-driven interface

This design keeps the existing SSEBop run step stable while removing environment-specific personal defaults from the published package interface.

## Scope

This design covers:

- introduction of a new top-level `pysweb.dem` package
- a GEE-backed NASADEM backend for SSEBop input preparation
- the public API change for `pysweb.ssebop.prepare_inputs(...)`
- migration of canonical Landsat helpers to `pysweb.ssebop.landsat`
- one-transition-round compatibility for `pysweb.ssebop.inputs.landsat`
- workflow CLI changes for SSEBop preparation
- notebook and README updates tied to the new package-backed behavior
- targeted test updates for imports, API forwarding, CLI parsing, and notebook text

This design does not cover:

- changing the core SSEBop ET model calculations
- removing the `dem` argument from `pysweb.ssebop.run(...)` in this change
- rewriting the legacy Earth Engine downloader into a fully new downloader architecture
- implementing additional DEM backends beyond NASADEM in this change
- redesigning SWB preprocess beyond sharing the same explicit `gee_project` style

## Core Decisions

### Explicit `gee_project` in the public API

The package must stop relying on a hard-coded Earth Engine project id such as `"yiyu-research"` in internal downloader code.

For package users, `gee_project` becomes a first-class public argument on GEE-backed paths. In this change that means:

- `pysweb.ssebop.prepare_inputs(...)`
- the SSEBop preparation workflow CLI
- `notebooks/01_run_pysweb.ipynb`

The same notebook should also pass the explicit `gee_project` into `pysweb.swb.preprocess(...)`, so both package workflows use the same visible Earth Engine project setting.

### Canonical DEM boundary

DEM sourcing moves into a dedicated package:

```text
pysweb/
  dem/
```

This gives DEM acquisition the same architectural role as meteorology and soil: a reusable data-source layer consumed by model orchestration code.

`pysweb.ssebop.prepare_inputs(...)` becomes responsible for orchestrating Landsat, ERA5-Land, and DEM preparation, but it should no longer embed DEM-source details directly in its body.

### DEM is prepared during `prepare_inputs`

The normal SSEBop workflow should materialize the DEM during the preparation step and store it in the prepared-inputs tree.

That means package consumers do not need to supply a pre-existing local DEM file for preparation. Instead:

1. `prepare_inputs` downloads the requested DEM for the target extent.
2. The DEM is written as a concrete local raster under the run’s prepared-inputs directory.
3. Later steps consume that prepared local raster.

This keeps all GEE-backed acquisition in step 1 and makes later processing stages operate on reproducible local artifacts.

### Default DEM source is NASADEM

The only implemented DEM backend in this change is:

- `dem_source = "nasadem"`

The default dataset is:

- `NASA/NASADEM_HGT/001`

Future DEM backends should follow the same package pattern as `pysweb.soil`, but only NASADEM is operational now.

### Canonical Landsat boundary

Canonical Landsat preparation helpers move to:

- `pysweb.ssebop.landsat`

The old path:

- `pysweb.ssebop.inputs.landsat`

remains as a thin compatibility shim for one transition round. The goal is a flatter, clearer package layout where SSEBop-specific helpers live directly under `pysweb.ssebop`.

### Keep `run()` stable for now

`pysweb.ssebop.run(...)` continues to accept:

- `dem = <local raster path>`

in this change.

The notebook and docs should point this argument at the DEM prepared by `prepare_inputs`, but the run-step contract itself should not be widened or removed in the same refactor. This limits risk by changing acquisition and preparation boundaries without altering SSEBop execution semantics.

## Target Module Layout

```text
pysweb/
  __init__.py
  dem/
    __init__.py
    api.py
    nasadem.py
  ssebop/
    __init__.py
    api.py
    landsat.py
    inputs/
      landsat.py   # compatibility shim for one transition round
```

The top-level `pysweb` package should expose `dem` alongside the existing subpackages:

- `io`
- `met`
- `ssebop`
- `soil`
- `swb`
- `visualisation`
- `dem`

Recommended package behavior:

- `pysweb.dem` exposes a package-level loader or dispatcher entrypoint
- `pysweb.dem.nasadem` contains the implemented backend
- future backends can later mirror the `pysweb.soil` structure without changing consumers

## Public Interfaces

### `pysweb.ssebop.prepare_inputs(...)`

The normal public preparation surface should be argument-driven rather than config-file-driven.

Required public behavior:

- `gee_project` is explicit and required for GEE-backed paths
- `dem_source` defaults to `"nasadem"`
- users provide output directories for Landsat, meteorology raw/stack outputs, and DEM outputs
- users do not need to supply a local DEM file for preparation

Expected public inputs after this refactor:

- `date_range`
- `extent`
- `met_source`
- `gee_project`
- `landsat_dir`
- `met_raw_dir`
- `met_stack_dir`
- `dem_dir`
- `dem_source = "nasadem"` by default

Optional advanced behavior may remain available through an internal or expert-oriented template override such as `gee_config_template`, but it must not be the notebook or README default.

### Workflow CLI

The SSEBop preparation workflow should follow the same public contract:

- replace the current user-facing `--gee-config` default path guidance with `--gee-project`
- stop requiring `--dem` as a mandatory user-supplied input for step 1
- derive `dem_dir` from the existing wrapper output layout as `<out-dir>/dem`

The workflow still acts as a thin wrapper over the package API rather than owning preparation logic itself.

### Notebook interface

`notebooks/01_run_pysweb.ipynb` should use one explicit notebook variable such as:

- `GEE_PROJECT = "your-ee-project"`

and pass it to:

- `pysweb.ssebop.prepare_inputs(...)`
- `pysweb.swb.preprocess(...)`

The notebook should stop presenting:

- `GEE_CONFIG = PROJECT_DIR / "gee_config" / "base_gee.yaml"`
- `DEM = Path("/path/to/dem.tif")`

as the normal user-configured preparation inputs.

Instead, it should define a prepared DEM path inside the run tree and point `pysweb.ssebop.run(...)` at that prepared artifact.

The notebook should use a deterministic prepared DEM path such as:

- `DEM_DIR = RUN_DIR / "inputs" / "dem"`
- `PREPARED_DEM = DEM_DIR / "nasadem.tif"`

### DEM source selection surface

The new `pysweb.dem` boundary should expose a stable source-selection contract, even though only one backend is implemented now.

Required public behavior:

- `dem_source` defaults to `"nasadem"`
- unsupported values fail immediately with supported choices
- known-but-unimplemented future values should eventually fail clearly, matching the `pysweb.soil` pattern

This refactor only wires the NASADEM backend end to end.

## Data Flow

### SSEBop preparation orchestration

`pysweb.ssebop.prepare_inputs(...)` should perform the following high-level sequence:

1. Parse and validate the requested date range and extent.
2. Prepare Landsat inputs using the canonical `pysweb.ssebop.landsat` module.
3. Download raw ERA5-Land daily inputs.
4. Resolve the requested DEM backend through `pysweb.dem`.
5. Download and materialize the DEM for the target extent under the prepared-inputs tree.
6. Build the ERA5-Land stacked NetCDF products against that prepared DEM grid and elevation raster.

The important boundary is that step 4 and step 5 become package-owned DEM acquisition rather than a user-managed external prerequisite.

### Prepared directory contract

After `prepare_inputs`, the prepared-inputs tree should contain:

- `landsat/`
- `met/era5land/raw/`
- `met/era5land/stack/`
- `dem/`

The DEM output should be a concrete local raster with a stable file name:

- `dem/nasadem.tif`

Later notebook cells and workflow wrappers should refer to that exact prepared artifact.

### NASADEM backend behavior

`pysweb.dem.nasadem` owns:

- the Earth Engine dataset identifier `NASA/NASADEM_HGT/001`
- Earth Engine initialization using the explicit `gee_project`
- extent validation
- dataset loading and export/download behavior
- writing the prepared DEM raster to the requested local output path

The backend should produce a DEM raster suitable for:

- geometry reference in `pysweb.met.era5land.stack`
- elevation input for ET0 calculations
- later use by `pysweb.ssebop.run(...)`

### Meteorology stack behavior

`pysweb.met.era5land.stack` continues to use the DEM as the reference for:

- grid shape
- CRS and transform alignment
- latitude extraction
- elevation input to short-crop reference ET calculations

This design does not change that dependency. It changes where the DEM comes from and how it is prepared.

## Migration Rules

### Landsat module migration

The real Landsat logic moves from:

- `pysweb/ssebop/inputs/landsat.py`

to:

- `pysweb/ssebop/landsat.py`

The old path remains as a thin re-export shim for one transition round so existing imports do not break immediately.

### Hard-coded project removal

Internal downloader code must no longer silently initialize Earth Engine with a personal hard-coded project id.

Migration rule:

- GEE-backed preparation paths require an explicit `gee_project`
- missing `gee_project` fails fast with a clear error
- there is no package-level fallback to `"yiyu-research"` or any other personal default

### Keep run-step compatibility

`pysweb.ssebop.run(...)` still takes `dem=<local raster path>` in this change.

Migration rule:

- package docs and notebooks point that argument at the prepared DEM
- existing run-step consumers that already provide a local DEM continue to work

This gives the repo one transition round where preparation is modernized without forcing an immediate redesign of the execution interface.

## Error Handling

Required failure behavior:

- if `gee_project` is missing for GEE-backed preparation, raise a clear argument or validation error
- if Earth Engine initialization fails for Landsat or DEM acquisition, raise a clear runtime error naming the failing project or backend
- if NASADEM cannot be read or produces an empty export for the requested extent, fail fast
- if the prepared DEM is missing when the meteorology stack stage begins, fail clearly
- if ERA5-Land rasters do not align with the prepared DEM grid, preserve the current stack validation errors

The package must prefer explicit failures over hidden defaults or silent fallback behavior.

## Documentation Notes

### Notebook clarification

`notebooks/01_run_pysweb.ipynb` should explicitly clarify that:

- `SM_RES` is the SWB preprocess target grid
- SSEBop preparation does not expose a separate notebook-level `ET_RES` parameter in the current package flow
- Earth Engine project selection is done through a visible `GEE_PROJECT` variable
- the DEM used by SSEBop is prepared in step 1 and then consumed locally in later steps

### README and notebook README

`README.md` and `notebooks/README.md` should stop describing the normal SSEBop preparation path in terms of:

- a required local DEM file provided by the user
- a prebuilt local GEE YAML file as the normal entrypoint

They should instead describe:

- explicit `gee_project`
- package-first preparation
- prepared DEM artifacts under the run tree

## Testing

The refactor should add or update tests for:

- top-level `pysweb.dem` import exposure
- `pysweb.dem.nasadem` backend dispatch and failure surfaces
- `pysweb.ssebop.prepare_inputs(...)` forwarding `gee_project` into Landsat and DEM acquisition
- canonical import of `pysweb.ssebop.landsat`
- compatibility import of `pysweb.ssebop.inputs.landsat`
- workflow CLI parsing for `--gee-project` and the new prepared-DEM behavior
- notebook and README text showing `GEE_PROJECT` and removal of stale `GEE_CONFIG` and local-DEM guidance
- preservation of existing SSEBop run-step tests so execution semantics stay stable

Testing should focus on contract changes and migration boundaries rather than re-proving the underlying ET model math.

## Non-Goals and Deferred Work

The following are intentionally deferred:

- removal or redesign of the `dem` argument on `pysweb.ssebop.run(...)`
- implementation of additional DEM backends
- a full replacement of the legacy YAML-driven Earth Engine downloader internals
- a generalized third-party plugin system for DEM sources

Those can be addressed later after the package surface is cleaned up and the NASADEM-backed path is stable.
