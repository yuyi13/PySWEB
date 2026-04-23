# PySWEB Local Tcold/FANO Correction Design

Date: 2026-04-23
Repository: `/g/data/ym05/github/yuyi13/PySWEB`
Status: Proposed and user-approved for planning

## Goal

Correct the local PySWEB SSEBop `Tcold` workflow so ET fraction is no longer an NDVI-only function, while keeping the existing workflow architecture:

- download Landsat scenes from GEE first
- run SSEBop calculations locally with xarray/rioxarray
- keep full-scene ET outputs
- avoid introducing cropland-only or landcover-only validity assumptions into `Tcold`

The corrected design should follow the operational logic of upstream `openet-ssebop` where it matters scientifically for findings 1 to 3, but it should be implemented as a simpler local analog rather than a full Earth Engine parity port.

## Problem Summary

The current PySWEB implementation computes:

- `tcold = lst - 0.125 * dt * (0.9 - ndvi) * 10`
- `etf = (tcold + dt - lst) / dt`

using the same per-pixel `lst`, `ndvi`, and `dt`.

This causes the `lst` and `dt` terms to cancel algebraically, so scene ET fraction collapses to a clipped linear function of NDVI before interpolation and ET scaling. In practice, the current workflow no longer behaves like a thermal-contrast SSEBop model.

The upstream `openet-ssebop` FANO workflow avoids this collapse because the cold boundary is not taken directly from the same fine pixel being evaluated. Instead, it is reconstructed through masked multiscale aggregation, smoothing, and fallback logic.

## Scope

### In scope

- replace the local `Tcold` builder used in the package-backed SSEBop workflow
- preserve the existing local `dT` calculation, ETf equation, interpolation, and daily ET assembly
- keep the current download-first workflow structure
- keep full-scene outputs
- use full-scene pixels to estimate the cold boundary, without requiring cropland masks
- include an upstream-style fallback when no `NDVI >= 0.4` anchor pixels are available

### Out of scope

- full Earth Engine parity with upstream `openet-ssebop`
- adding `qa_water`, `ndwi`, or other new Landsat support bands to the local workflow
- making landcover masks part of `Tcold` estimation
- changing the local `dT` climatology method
- changing the final ET interpolation or partitioning logic
- masking final outputs to cropland or any other landcover class

## Core Design Decision

PySWEB will keep the current local pipeline but replace the same-pixel `Tcold` construction with a local FANO analog that:

1. computes a fine-scale candidate `Tc` surface from `lst`, `ndvi`, and `dt`
2. identifies a vegetated anchor subset from the full scene using `NDVI >= 0.4`
3. reconstructs a coarse cold-boundary surface from those anchors
4. fills missing coarse support with a scene fallback derived from valid fine-scale `Tc`
5. smooths the resulting cold boundary and maps it back to the Landsat grid
6. computes ETf for the full scene against that reconstructed `Tcold`

This keeps the upstream scientific logic that matters most:

- the cold boundary is estimated from a subset of likely cold/wet pixels
- the final `Tcold` used for ETf is not the raw same-pixel `Tc`
- sparse or non-vegetated scenes do not fail outright because a fallback exists

## Recommended Local Algorithm

### Inputs

The corrected `Tcold` workflow uses only variables already available in the current local pipeline:

- `lst`
- `ndvi`
- `dt`

No new Landsat bands are required for this correction.

### Step 1: Fine-scale candidate temperature

Compute a fine-scale candidate cold temperature field using the same upstream FANO form:

```text
tc_fine = lst - 0.125 * dt * (0.9 - ndvi) * 10
```

This step is retained from the current implementation, but it is no longer used directly as the final `Tcold`.

### Step 2: Full-scene vegetated anchor pool

Define anchor candidates from the full scene using only scene variables:

- `lst`, `ndvi`, and `dt` must be finite
- `NDVI >= 0.4`

The `0.4` threshold is inherited from upstream `openet-ssebop`, where it is used operationally to define the high-NDVI pool that anchors the cold boundary.

This threshold should be:

- fixed at `0.4` by default
- exposed as a configuration parameter for advanced users

### Step 3: Coarse cold-boundary support field

Aggregate the anchor pixels to a coarser support grid so the cold boundary is not defined by noisy single-pixel values.

The local analog should preserve the intent of the upstream multiscale structure:

- derive a coarse `Tcold` support field from anchor pixels only
- use spatial averaging rather than same-pixel assignment

The implementation does not need to reproduce Earth Engine `reduceResolution()` exactly. A local raster analog based on coarsening/block aggregation is acceptable.

The first implementation should use upstream-inspired support scales:

- fine support equivalent: approximately 240 m
- coarse support equivalent: approximately 4800 m

When exact scale matching is awkward in local rasters, the implementation may use integer window factors computed from the native Landsat resolution. The scientific requirement is the multiscale separation, not exact Earth Engine reprojection semantics.

### Step 4: Scene fallback for sparse or empty anchor pools

If the `NDVI >= 0.4` anchor pool is sparse or empty, the workflow should fall back in the same spirit as upstream `openet-ssebop`.

Compute a scene fallback:

```text
tc_scene = mean(tc_fine over all valid fine pixels)
```

Important details:

- this fallback is computed from valid fine-scale `tc_fine`
- it is not restricted to `NDVI >= 0.4`
- it exists specifically so bare or harvested scenes still produce a stable `Tcold`

This matches the current upstream fallback logic more closely than using a crop mask or requiring anchor pixels to exist in all scenes.

### Step 5: Fill and smooth the coarse `Tcold`

Fill missing coarse `Tcold` support cells with `tc_scene`, then smooth the resulting support field before projecting it back to the Landsat grid.

The local implementation should use a simple, explicit smoothing rule such as:

- fill missing coarse cells with the scalar `tc_scene`
- apply a small mean filter to the coarse surface
- reproject or resample back to the Landsat grid
- apply a final light smoothing pass on the projected `Tcold`

The goal is not to mimic every upstream pixel operation. The goal is to avoid discontinuous, hole-ridden `Tcold` surfaces while preserving the coarse cold-boundary structure.

### Step 6: Full-scene ET fraction

Compute ET fraction for the full scene using the existing SSEBop equation:

```text
etf = (tcold + dt - lst) / dt
```

Then keep the current PySWEB masking and clipping behavior:

- mask ETf values above the configured maximum
- clip ETf to the configured range

The corrected design does not apply a final cropland or landcover mask.

## Behavior in Bare or Harvested Seasons

If a scene contains no pixels with `NDVI >= 0.4`, the corrected workflow should still run.

In that case:

- the vegetated anchor pool is empty
- the coarse anchor field is therefore missing
- the algorithm fills the missing support using `tc_scene`
- ETf is computed against that fallback `Tcold`

This mirrors the practical behavior of upstream `openet-ssebop`: the model still produces ETf, but the cold boundary is fallback-driven rather than vegetation-anchored.

This is scientifically weaker than a well-vegetated scene, but it is preferable to either:

- crashing the workflow
- silently reverting to the current NDVI-only collapse
- requiring cropland masks that make the method invalid outside croplands

## Why Landcover Is Excluded

The correction explicitly avoids using ESA WorldCover or any cropland-only landcover mask in `Tcold` estimation.

Reasoning:

- PySWEB SSEBop should remain usable for scenes that are not cropland-dominated
- a cropland-only cold-anchor rule would make model validity depend on external landcover assumptions rather than scene behavior
- the scientific failure being corrected is the same-pixel collapse, not the absence of landcover masks

Landcover may still remain elsewhere in the workflow for unrelated purposes, but it should not define the corrected `Tcold` anchor pool.

## Pipeline Impact

### Download stage

No required changes for this correction.

The current Landsat download configuration can remain based on:

- `ST_B10`
- `SR_B4`
- `SR_B5`

### Local SSEBop run stage

The local run stage will change in these ways:

- replace the current simple `tcold_fano_simple_xr()` logic
- add a new multiscale local `Tcold` builder
- keep using full-scene `lst`, `ndvi`, and `dt`
- keep existing ETf interpolation and daily ET assembly

### Configuration surface

The corrected design should expose these `Tcold/FANO` parameters in a controlled way:

- `dt_coeff`, default `0.125`
- `high_ndvi_threshold`, default `0.9`
- `anchor_ndvi_threshold`, default `0.4`
- coarse/fine support controls, defaulting to upstream-inspired values

Default behavior should match the design above without requiring user intervention.

## Validation Strategy

The correction should be validated against the scientific failure modes identified in the inspection.

### Required checks

1. **No algebraic collapse**
   - With fixed `ndvi` and `dt`, changing `lst` must change ETf after the correction.

2. **Anchor-pool path works**
   - In a scene with some `NDVI >= 0.4` pixels, the final `Tcold` must differ from raw same-pixel `tc_fine` for at least part of the scene.

3. **Fallback path works**
   - In a scene with no `NDVI >= 0.4` pixels, the workflow must still produce finite `Tcold` and ETf using the scene fallback.

4. **Full-scene output preserved**
   - Final ETf and daily ET products must still cover the full scene rather than only anchor pixels.

5. **Backward compatibility of outer workflow**
   - Existing run entrypoints, band mapping, and output dataset structure should remain stable unless explicitly documented otherwise.

## Consequences

### Expected benefits

- restores thermal sensitivity to ETf
- removes the current NDVI-only collapse
- keeps the workflow valid outside croplands
- preserves the current local processing architecture
- adds a robust fallback for sparse-vegetation scenes

### Known limitations

- this remains a local analog of upstream FANO, not a full Earth Engine reproduction
- without `qa_water` and `ndwi`, water and shadow conditioning will remain simpler than upstream
- fallback-driven `Tcold` in very bare scenes is operationally useful but scientifically weaker than vegetation-anchored `Tcold`

## Summary

The correction should not attempt to make PySWEB a cropland-only SSEBop model and should not keep the current same-pixel `Tcold` shortcut.

Instead, PySWEB should implement a local, full-scene FANO analog that:

- uses a data-driven vegetated anchor subset from the full scene
- reconstructs `Tcold` through multiscale aggregation and smoothing
- falls back to a scene-level `Tc_scene` when anchor pixels are absent
- computes full-scene ETf against that reconstructed cold boundary

This is the minimum scientifically defensible correction that addresses findings 1 to 3 while staying inside the current GEE-download plus local-calculation workflow.
