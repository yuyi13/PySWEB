# Migration to the maintained package

## September 2026 baseline

GitHub and Gadi shared commit `e96879a`; the local checkout was four commits behind. Gadi additionally contained notebook/PBS improvements. Those two files were committed as `3fb0f23`, then GitHub, local and Gadi were fast-forwarded to that baseline before this update. This preserves the latest development rather than replacing it with an older checkout.

## Architecture and paths

| Previous surface | Current surface |
|---|---|
| Top-level `core.*` | `pysweb.io.gee_downloader`, `pysweb.met.*`, `pysweb.ssebop.*`, `pysweb.swb.solver` |
| Numbered workflows with implementation | Thin `workflows/` wrappers and installed `pysweb` commands |
| Duplicated `spec` preprocessing/solver | `pysweb.soil.mlcons` / `pysweb.soil.custom` and canonical SWB workflow |
| Shell-specific run settings and notebook copies | Shared TOML configuration and `pysweb.workflow` |
| `docs/superpowers` plans | Maintained methods, data, workflow and migration documentation |
| Root `SWEB_logo.png` | `docs/assets/pysweb-logo.png` |

The former top-level `core/` had already been removed from the latest baseline. This update closes the retirement plan through installed-wheel verification outside the checkout, canonical imports and direct solver regression tests. Package modules named `core.py` are maintained implementation modules within `pysweb`; they are unrelated to the retired top-level namespace. The small deprecated `pysweb.ssebop.inputs` compatibility namespace remains available.

Historical solver explanations already moved into `pysweb.swb.solver` remain there. This update retains the existing hydraulic equations, root distribution, diffusion limiter, ET floor relaxation, OpenLandMap mapping and local cold-temperature/FANO behavior. Current assumptions and unresolved scientific checks are summarized in `docs/methods.md`. Older implementation notes and code remain available through Git history at `3fb0f23` and its predecessors.

## Intentional behavior changes

- Shell launchers now require `--config study.toml`; old positional date/extent conventions and embedded paths are replaced by explicit configuration. The numbered Python CLIs remain available.
- `spec/3_sweb_mlcons_preprocess_inputs.py` defaults to the MLConstraints adapter and otherwise uses the canonical preprocess arguments. Set local source paths explicitly. `spec/5_sweb_mlcons_run_model.py` delegates to the package solver.
- Full calendar-month rainfall is required for monthly effective-precipitation calculations. Previously produced partial-month effective rain should be regenerated.
- Missing daily timestamps and mismatched grids raise errors. Present-but-missing forcing has explicit QC; zero filling remains an opt-in compatibility choice.
- The default start-of-day state convention is preserved. End-of-day reporting is optional, and final states plus water budgets are now retained.
- One-layer bottom drainage bookkeeping is corrected. Saturation overflow and floor adjustments are exposed; these changes can alter one-layer historical results.
- Calibration uses matched spatial support and geographic area weights, native resolution by default, a reproducible seed and explicit convergence checks. Old parameter estimates should be re-evaluated before reuse.
- User-provided soils are supported through one hydraulic contract. Historical MLConstraints pedotransfer conventions are preserved and identified in metadata.
- Old cached model outputs without a matching manifest are rejected by `skip_existing`. Rerun without that option to regenerate a product from verified inputs.

The update adds packaging, MIT licensing, CI and examples. It does not publish to PyPI or establish a new scientifically validated model release.

SSEBop interpolation now retains actual observation dates, including the last scene and a single-scene input. The maximum-gap rule applies to interpolated dates. Previously the last measured day, or a measured day preceding a long gap, could be masked despite having an observation. Dates outside the observed interval remain missing.

## Component notation migration (11 October 2026)

New SSEBop files write `ET`, `E_s`, `E_t` and `f_t`, alongside the existing interpolation fields. `E_s` and `E_t` use mm/day and describe requested SEB forcing; `f_t` is dimensionless. Total `ET`/`et`, the water-limited `actual_et` budget, temperature identifiers (including `tcold_k`), drainage and time indices retain their existing meanings. Numerical equations, masks, calendars and state timing are unchanged.

| Previous contract | Canonical contract |
|---|---|
| `E` / `e` | `E_s` (soil evaporation) |
| `T` / `t` | `E_t` (plant transpiration) |
| `Tc` | `f_t` (transpiration fraction) |
| `--t`, Python `t=` | `--transpiration`, Python `transpiration=` |
| `--t-var`, Python `t_var=` | `--transpiration-var`, Python `transpiration_var=` (default `E_t`) |
| `--e-var`, Python `e_var=` | `--soil-evaporation-var`, Python `soil_evaporation_var=` |
| `--tc-var`, Python `tc_var=` | `--transpiration-fraction-var`, Python `transpiration_fraction_var=` (default `f_t`; reserved, unused by preprocessing) |
| `t_daily_<dates>.nc` containing `t` | `transpiration_daily_<dates>.nc` containing `E_t` |
| `e_daily_<dates>.nc` containing `e` | `soil_evaporation_daily_<dates>.nc` containing `E_s` |

Preprocessing accepts total ET plus `E_t` (`et_var="ET", transpiration_var="E_t"`) or `E_s` plus `E_t` (`soil_evaporation_var="E_s", transpiration_var="E_t"`). The latter reconstructs total `et` as `E_s + E_t`. Date tags and total-ET filenames are preserved. The shared workflow applies these settings to both simulation and calibration windows.

Legacy variable names and argument spellings remain accepted during the current 0.1.x migration period and emit `FutureWarning` deprecation notices. Explicit variable selections take precedence. Automatic fallback checks only the listed component aliases; temperature fields cannot supply transpiration. Coexisting canonical and legacy variables must have matching dimensions, coordinates, masks, values and equivalent units. Conflicting variables or duplicate old/new arguments raise an error. New files contain only canonical component fields.

Simulation and calibration can resolve a missing `transpiration_daily_<dates>.nc` to an existing `t_daily_<dates>.nc` when the old preprocessing manifest has supported schema version 1, its recorded output checksum matches, and its component fields validate. The shared resolver also supports the equivalent soil-evaporation filename mapping. Missing or conflicting provenance requires rerunning preprocessing. Explicit legacy source paths remain accepted and are hashed as inputs. Source files are preserved; resolved paths are recorded in new manifests.

New manifests carry `component_schema="sweb-components-v2"` in both metadata and configuration identity, and record regenerated output hashes. Existing final simulation outputs from an older schema fail `--skip-existing` validation and must be regenerated. A matching filename alone does not permit cache reuse.

## Native grids and paired products (17 October 2026)

New configured runs use [native downloads and two preparation grids](native-grids.md). Remove `run.resolution`, choose `grid_mode = "native"`, and use a new output directory. Download reprojection is replaced by CRS-plus-affine preservation, Landsat 8/9 are the default sensors, calibration uses area-weighted GSSM cell-day errors, and the final SEB/SWB products share a Landsat 30 m UTM grid. Refit the four parameters after changing the calibration grid or objective. Existing degree-resolution TOML files select legacy mode; individual old-style component outputs remain supported.

## Conservative SWB (18 October 2026)

The [two-parameter formulation](swb-formulation.md) supersedes the historical four-parameter model described above. Remove `sm_max_factor`, `sm_min_factor` and their calibration bounds, derive a frozen native-cell GSSM reference artifact, set `model.reference_file`, and calibrate diffusivity and root beta into a new output directory. Old calibrations and final-state trajectories require regeneration with the revised equations. New calibration files bind the formulation, fixed numerical and drainage/root-depth settings, and reference identity. Existing historical records remain readable with their pinned source.


## Daily SWB and actual-prefix warm-up (18 October 2026)

`swb-daily-conservative-2p-v2` makes one bounded update per daily forcing record. It replaces v1's 768-or-more internal steps with a discrete daily balance; fixed physical properties, GSSM reference definitions, tau = 3 days, and the two fitted parameters are retained. Remove numerical `max_substep_days`/`hydraulic_cfl` overrides. V1 fitted parameters cannot be reused or relabelled: recalibrate and regenerate trajectories under the new formulation identity. Existing immutable p01/p99 reference artifacts remain compatible and retain their hashes.

For field capacity → actual 2013 → scoring in 2014–2020, remove the `[spinup]` table and set the run start to its first available 2013 date. Set `calibration.start = "2014-01-01"` and `calibration.end = "2020-12-31"` (the 2014 start is already the Landsat 8/9 default). Use a new output directory so initialized files from an earlier repeated-year run cannot be confused with the revised protocol. Physical field capacity remains an explicit assumption; the shorter warm-up does not imply extreme-start convergence.
