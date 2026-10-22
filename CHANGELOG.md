# Changelog

## Repeated calendar-block spin-up — 2026-10-20

- Add complete, consecutive `spinup.source_years` blocks to both solver backends, calibration and native workflows, retaining leap days and the original output calendar.
- Count repetitions including the retained source block and exclude that block from scoring. Keep memory independent of repetition count by discarding each column's extra warm-up trajectories.
- Retain mandatory convergence gates by default; support explicitly labelled fixed-duration experiments that preserve failed diagnostics without claiming convergence.
- Validate compact histories against expanded histories and verify TOML/API integration; 480 offline tests passed in Gadi job 181730321.

## Daily two-parameter SWB — 2026-10-18

- Use one conservative update per daily forcing record; retain donor/receiver limits, physical bounds and separate actual/unmet E/T budgets.
- Version the daily formulation separately from the subcycled solver and reject its prior calibrations; retain compatible immutable GSSM references.
- Document and verify physical field capacity → available actual 2013 warm-up → 2014–2020 scoring, with repeated-year spin-up only by explicit opt-in.
- Preserve Landsat 8/9 and the 5100 m centre-based chip defaults.

## Native-grid workflow — 2026-10-17

- Preserve Earth Engine source CRS and full affine alignment in downloaded products; retain native soil/reference inputs and grid provenance.
- Default to Landsat 8/9, with Landsat 5/7 opt-in.
- Prepare separate native GSSM calibration and Landsat 30 m UTM simulation grids. Fit area-weighted cell-day errors and transfer four scalar parameters.
- Produce one auxiliary-rich SEB NetCDF and one default/calibrated SWB NetCDF on an automatically verified identical spatial grid.
- Protect a configured holdout boundary in calibration, dT climatology, temporal reconstruction and optional NDVI root-depth statistics. Preserve explicit legacy grid workflows.


## 2026-10-17

- Keep local FANO coarsening compatible with xarray 2026.9.0 while preserving NaN-aware averaging and padded-edge coordinates.

## 2026-10-16

- Preserve Landsat QA in default downloads, enable pixel cloud masking and avoid median compositing of bitfields.
- Require QA during SSEBop scene processing, retain valid zero saturation flags and reject missing requested bands; explicitly support independently pre-masked legacy inputs.
- Add a reusable Collection 2 QA decoder for study adapters and record the QA mode in scene and daily outputs.
- Correct the OpenLandMap organic-matter input to mass percent in the Saxton–Rawls pedotransfer equations (from the AU-Ync pilot).

## 2026-10-11

- Align component notation with the manuscript: `E_s`, `E_t`, `f_t`; use descriptive CLI/API options and prepared filenames.
- Validate legacy input aliases and provenance, reject conflicting duplicates, and preserve numerical model behaviour.

## 0.1.0 — 2026-09-12

- Aligned GitHub, local and Gadi development histories from the Gadi notebook improvements.
- Added declared dependencies, installed CLIs, MIT licence, citation metadata, offline demo and CI.
- Centralized stage orchestration in TOML configuration, with portable shell and notebook clients.
- Integrated custom hydraulic soils and the MLConstraints adapter into the maintained package.
- Added forcing/calendar/grid validation, full-month effective rainfall, final states, explicit water budgets and missing-forcing QC.
- Matched calibration support and geographic weights, seeded optimization and recorded convergence.
- Added atomic NetCDF writing and provenance-based output validation; incomplete Earth Engine downloads now fail explicitly.
- Replaced obsolete implementation plans with current methods, data and migration documentation; revised the scientific logo and README.

See `docs/migration.md` for behavior changes and retained assumptions. This version identifies the package interface; predictive skill requires study-specific validation.
- Preserved observed SSEBop dates at interpolation boundaries and across long gaps; added sparse-scene regression tests and explicit NumPy datetime units.
