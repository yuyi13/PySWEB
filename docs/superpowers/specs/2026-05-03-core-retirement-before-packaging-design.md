# Core Retirement Before Packaging Design

Date: 2026-05-03

## Goal

Retire the top-level `core/` directory as a runtime dependency before publishing PySWEB as an installable package.

The package wheel currently includes only `pysweb*`, but `pysweb.io.gee` still imports `core.gee_downloader`. That means installed-package usage can fail even though local editable-repo usage works. The refactor should make `pysweb/` self-contained for supported runtime paths while keeping short-term compatibility shims where useful.

## Current Context

The repository is already package-first:

- `pysweb/` is the canonical package namespace.
- `workflows/` contains thin CLI wrappers around package APIs.
- `core/` remains as transitional legacy/reused code.

Most model execution paths have already moved to package modules, but tests and some compatibility imports still reference `core.*`.

## Approved Direction

Do not publish `core` as a package namespace.

Instead:

1. Move or absorb remaining real implementation into `pysweb/`.
2. Repoint package, workflow, test, and documentation references to canonical `pysweb.*` modules.
3. Keep top-level `core/` files only as temporary deprecated compatibility shims where they reduce migration friction.
4. Remove stale `core/` files only after there are no supported internal callers and tests no longer depend on them.

## Comment And Documentation Preservation

Some `core/` files contain useful scientific comments, formula notes, unit assumptions, and operational explanations. Migration must compare each affected `core` module against the current `pysweb` equivalent before deleting or replacing it.

Keep or merge comments when they explain:

- Scientific assumptions, equations, empirical constants, or cited methods.
- Units, dimensional conventions, and sign conventions.
- Boundary conditions or numerical stability choices.
- Earth Engine, raster, NetCDF, or HPC behavior that is not obvious from code alone.

Do not preserve comments that are stale, script-runner-specific, duplicated by clearer package code, or contradicted by current behavior.

## Module Actions

### `core/gee_downloader.py`

Priority: highest.

Move the downloader implementation into package-owned code, likely `pysweb.io.gee` or a helper module such as `pysweb.io.gee_downloader`.

`pysweb.io.gee` should no longer import from `core.gee_downloader`. Existing package callers such as Landsat and ERA5-Land download code should continue importing from `pysweb.io.gee`.

The old `core/gee_downloader.py` can become a compatibility shim that imports from `pysweb.io.gee` and exposes the old CLI behavior for a transition period.

### ERA5-Land Helpers

Retire these in favor of package modules that already exist:

- `core/era5land_download_config.py` -> `pysweb.met.era5land.download`
- `core/era5land_refet.py` -> `pysweb.met.era5land.refet`
- `core/era5land_stack.py` -> `pysweb.met.era5land.stack`

Tests should import the package modules. Compatibility shims can remain temporarily if external scripts may still use the old paths.

### Meteorology Paths

Retire `core/met_input_paths.py` in favor of `pysweb.met.paths`.

All tests, workflows, and docs should refer to `pysweb.met.paths`.

### SSEBop AU Compatibility

Retire `core/ssebop_au.py` as a canonical source.

Any remaining tests or consumers should move to the package modules:

- `pysweb.ssebop.core`
- `pysweb.ssebop.landcover`
- `pysweb.met.silo.readers`

If old names such as `SsebopAuConfig` are still useful for compatibility, expose them through a temporary shim with deprecation guidance rather than duplicating logic.

### SWB Legacy Solver Files

Treat these as legacy/reference code:

- `core/swb_model_1d.py`
- `core/soil_hydra_funs.py`

The runtime path already uses `pysweb.swb.solver`. Before retiring the old files, compare their comments and formula explanations against `pysweb.swb.solver` and merge any still-valid explanatory notes.

If compatibility shims remain, they should forward to `pysweb.swb.solver` rather than maintaining a separate solver implementation.

### Tridiagonal Solver Utility

Review `core/thomas_solve_tridiagonal_matrix.py` before removal.

If it is unused and redundant with package solver code, retire it. If it provides a clearer standalone implementation or useful comments, incorporate that clarity into `pysweb.swb.solver` tests or documentation before deleting it.

## Testing

Add or update focused tests for:

- Installed-package smoke behavior: package modules must not require top-level `core`.
- `pysweb.io.gee` works without importing `core.gee_downloader`.
- ERA5-Land helper tests import from `pysweb.met.era5land.*`.
- Meteorology path tests import from `pysweb.met.paths`.
- SSEBop and SWB tests import from canonical package modules.
- Any retained `core` compatibility shims import successfully and emit or document deprecation without owning separate logic.

Run the full test suite after the staged migration.

## Documentation

Update README and notebook guidance so:

- `pysweb/` is the only canonical runtime package.
- `core/` is described as deprecated compatibility shims if it remains at all.
- New development must not add functionality to `core/`.
- Package publication notes mention that runtime imports are self-contained under `pysweb`.

## Out Of Scope

This work will not:

- Redesign model equations or recalibrate defaults.
- Change user-facing high-level APIs such as `pysweb.ssebop.run` or `pysweb.swb.run`.
- Add new plotting behavior.
- Publish the package to PyPI or another registry.

Packaging metadata improvements such as dependencies, license metadata, and console scripts can be handled after `pysweb/` no longer depends on `core/`.

