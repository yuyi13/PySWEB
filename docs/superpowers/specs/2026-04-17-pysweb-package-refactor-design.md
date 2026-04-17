# PySWEB Package-First Refactor Design

Date: 2026-04-17
Repository: `/g/data/ym05/github/PySWEB`
Status: Proposed and user-approved for planning

## Goal

Refactor the repository into a light Python package named `pysweb` with two explicit scientific entrypoints:

- `pysweb.ssebop` for the energy-balance side
- `pysweb.swb` for the water-balance side

The refactor should:

- keep the codebase usable during active development
- aggressively rename AU-specific and script-centric interfaces where needed
- make Python APIs and notebook usage the primary interface
- keep bash runners as thin convenience wrappers, not the main architecture
- separate SSEBop math from meteorology-source adapters such as ERA5-Land and SILO
- integrate the ERA5-Land download/stack substeps into the high-level SSEBop input-preparation entrypoint without creating a new monolith

## Core Decisions

### Package and naming

- Project/distribution name remains `PySWEB`.
- Import/package namespace becomes `pysweb`.
- The two top-level scientific modules are:
  - `pysweb.ssebop`
  - `pysweb.swb`
- Meteorology-source code does not live under `pysweb.ssebop`; it lives under `pysweb.met`.

### User-facing API style

The primary public interface is high-level Python APIs:

- `pysweb.ssebop.prepare_inputs(...)`
- `pysweb.ssebop.run(...)`
- `pysweb.swb.preprocess(...)`
- `pysweb.swb.calibrate(...)`
- `pysweb.swb.run(...)`

These high-level APIs are backed by lower-level internal modules so notebook users get a simple interface while the codebase stays testable and composable.

### Bash runner policy

Bash runners are retained, but only as thin wrappers around Python entrypoints. They are no longer the canonical architecture. The canonical logic lives in the package.

## Target Package Layout

```text
pysweb/
  __init__.py
  ssebop/
    __init__.py
    api.py
    core.py
    landcover.py
    grid.py
    inputs/
      __init__.py
      landsat.py
  swb/
    __init__.py
    api.py
    core.py
    preprocess.py
    calibrate.py
    run.py
  met/
    __init__.py
    paths.py
    era5land/
      __init__.py
      download.py
      stack.py
      refet.py
    silo/
      __init__.py
      paths.py
      readers.py
  io/
    __init__.py
    gee.py
```

## Module Responsibilities

### `pysweb.ssebop`

This package owns SSEBop-specific logic only.

- `api.py`
  - user-facing high-level functions such as `prepare_inputs()` and `run()`
- `core.py`
  - SSEBop math and model-facing helpers
  - examples: `dt_fao56_xr`, `compute_dt_daily`, `et_fraction_xr`, `tcold_fano_simple_xr`, `build_doy_climatology`, `daily_et_from_etf`
- `landcover.py`
  - WorldCover loading and mask generation
- `grid.py`
  - reprojection and grid-alignment helpers currently mixed into `ssebop_au.py`
- `inputs/landsat.py`
  - Landsat-specific preparation and GEE download orchestration

### `pysweb.swb`

This package owns the water-balance side.

- `api.py`
  - user-facing high-level functions such as `preprocess()`, `calibrate()`, and `run()`
- `core.py`
  - SWB core wrappers and shared model-facing helpers
- `preprocess.py`
  - logic now driven by `3_sweb_preprocess_inputs.py`
- `calibrate.py`
  - logic now driven by `4_sweb_calib_domain.py`
- `run.py`
  - logic now driven by `5_sweb_run_model.py`

### `pysweb.met`

This package owns meteorology-source adapters.

- `paths.py`
  - source-neutral path resolution
  - logic currently introduced in `core/met_input_paths.py`
- `era5land.download`
  - ERA5-Land DAILY_AGGR config creation and download orchestration
- `era5land.stack`
  - daily raster discovery, reading, and NetCDF stacking
- `era5land.refet`
  - ETo/refET helpers derived from the current ERA5-Land stack logic
- `silo.paths`
  - SILO path conventions and yearly file resolution
- `silo.readers`
  - SILO NetCDF readers and variable handling

### `pysweb.io`

- `io.gee`
  - wraps and eventually absorbs reusable Earth Engine download utilities from `core/gee_downloader.py`

## File and Rename Strategy

The refactor is intentionally aggressive about naming cleanup.

### Immediate naming principles

- use `prepare_inputs` for the SSEBop first step
- reserve `preprocess` for the SWB side
- remove `au` from the main SSEBop module naming
- keep `ssebop` and `swb` explicit in package/module names

### Planned renames

- `core/ssebop_au.py`
  - split and replace with `pysweb.ssebop.core`, `pysweb.ssebop.landcover`, and `pysweb.ssebop.grid`
- `core/era5land_refet.py`
  - move to `pysweb.met.era5land.refet`
- `core/era5land_stack.py`
  - move to `pysweb.met.era5land.stack`
- `core/era5land_download_config.py`
  - move to `pysweb.met.era5land.download`
- `core/met_input_paths.py`
  - move to `pysweb.met.paths`
- `workflows/1_ssebop_preprocess_inputs.py`
  - rename to `workflows/1_ssebop_prepare_inputs.py`
  - this becomes the unified SSEBop input-preparation CLI wrapper

Backward compatibility is not a design goal for legacy names. Downstream usage will be updated to the new names during the refactor.

## Integration of `1`, `1b`, and `1c`

The user wants the ERA5-Land substeps integrated into the first SSEBop preparation step. The integration point is orchestration, not implementation-body merging.

### Final behavior

`pysweb.ssebop.prepare_inputs(...)` performs:

1. Landsat GEE download/prep
2. meteorology download/prep for the chosen source
3. meteorology stacking into model-ready NetCDFs

For the current implementation, the meteorology source is ERA5-Land. SILO support remains available through `pysweb.met.silo`, but the package should treat the source as a pluggable adapter.

### CLI behavior

`workflows/1_ssebop_prepare_inputs.py` becomes the main Python CLI for the entire first step.

It should:

- accept the overall date range and extent
- accept a meteorology source selector such as `era5land` or `silo`
- call the Landsat preparation helper
- call the chosen meteorology adapter
- write outputs into the expected run directory structure

### Role of `1b` and `1c`

`1b_download_era5land_daily.py` and `1c_stack_era5land_daily.py` remain available, but only as direct-access substep CLIs and debug tools. They are not the primary entrypoint after the refactor.

This preserves:

- a simple high-level path for most users
- a stepwise debug path for developers
- clean separation of responsibilities inside the package

## SSEBop and Meteorology Boundary

This is the most important technical boundary in the refactor.

### SSEBop owns

- dT and ETf math
- Landsat/LST/NDVI-driven logic
- land-cover and mask helpers needed by SSEBop
- grid matching and reprojection helpers that are SSEBop-facing

### Meteorology adapters own

- source-specific download conventions
- source-specific file naming and path resolution
- source-specific reading/parsing logic
- source-specific ETo/refET derivation
- source-specific unit conventions

### Consequence

`pysweb.ssebop.run(...)` consumes explicit meteorology products or a source adapter configuration. It does not contain ERA5-Land-specific or SILO-specific path conventions internally.

This makes SSEBop “global-capable” by decoupling it from the meteorology source rather than by stuffing all source logic into the SSEBop module.

## Public API Sketch

### High-level package usage

```python
import pysweb

pysweb.ssebop.prepare_inputs(
    date_range="2024-01-01 to 2024-01-31",
    extent=[147.2, -35.1, 147.3, -35.0],
    met_source="era5land",
    out_dir="...",
)

pysweb.ssebop.run(
    date_range="2024-01-01 to 2024-01-31",
    landsat_dir="...",
    met_dir="...",
    dem="...",
    output_dir="...",
)

pysweb.swb.preprocess(...)
pysweb.swb.calibrate(...)
pysweb.swb.run(...)
```

### Lower-level internals remain available

Internal modules remain directly importable for notebooks and development, but they are secondary interfaces:

```python
from pysweb.met.era5land.download import build_config, download_daily
from pysweb.met.era5land.stack import stack_daily_inputs
from pysweb.ssebop.core import compute_dt_daily
```

## Packaging Strategy

The package should be introduced lightly.

### Add now

- `pyproject.toml`
- editable install support
- package directory `pysweb/`
- minimal package metadata

### Do not overdo yet

- no attempt to freeze a broad stable public API beyond the top-level functions listed above
- no aggressive plugin framework
- no heavy build/release process changes unless needed later

The goal is to make notebooks and Python scripts able to do:

```python
import pysweb
```

without turning the repository into a rigid published-library project too early.

## Transition and Implementation Phases

### Phase 1: package skeleton and wrappers

- create `pysweb/`
- add `pysweb.ssebop.api` and `pysweb.swb.api`
- wrap existing workflow logic rather than rewriting immediately
- keep current scripts callable while updating them to delegate into the package

### Phase 2: split `ssebop_au.py`

- move SSEBop math into `pysweb.ssebop.core`
- move landcover helpers into `pysweb.ssebop.landcover`
- move reprojection/grid helpers into `pysweb.ssebop.grid`
- remove AU-specific naming from the main SSEBop logic

### Phase 3: move meteorology logic into `pysweb.met`

- ERA5-Land download/stack/refET modules
- SILO path/read modules
- source-neutral path resolver

### Phase 4: unify first-step orchestration

- rename `1_ssebop_preprocess_inputs.py` to `1_ssebop_prepare_inputs.py`
- make it call `pysweb.ssebop.prepare_inputs(...)`
- leave `1b` and `1c` as direct-access substep CLIs

### Phase 5: convert runners to thin wrappers

- bash runners call Python entrypoints only
- update notebooks and README examples to import `pysweb`

## Testing Expectations

The refactor should preserve and expand the current test style.

Required checks during implementation:

- package import tests
- unit tests for moved SSEBop math
- unit tests for ERA5-Land and SILO meteorology adapters
- workflow CLI tests for the renamed `1_ssebop_prepare_inputs.py`
- regression tests for runner argument wiring

## Non-Goals for This Refactor

- full soil-globalization work
- replacing SMAP/reference-data strategy
- removing bash runners entirely
- publishing a mature external package on day one
- complete reimplementation of all workflows before introducing the package

## Recommended First Implementation Batch

The first implementation batch should focus on structure, not algorithm change:

1. add `pysweb/` package skeleton and `pyproject.toml`
2. split `ssebop_au.py` into SSEBop-focused modules
3. move ERA5-Land helpers under `pysweb.met.era5land`
4. move path resolution under `pysweb.met.paths`
5. rename and unify `1_ssebop_prepare_inputs.py` as the first-step CLI wrapper
6. update runners and imports to the new module names

This sequence minimizes churn while establishing the long-term architecture.
