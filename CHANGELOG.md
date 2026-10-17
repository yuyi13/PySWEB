# Changelog

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
