# Changelog

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
