# Configured scientific acceptance

The native workflow can enforce an explicit scene inventory, validated L7 preprocessing, development-only initialization, paired forcing support and site-configured evaluation. These options are opt-in; their rules belong in the study TOML and immutable scene manifest. They are rejected by the legacy workflow.

## Scene inventory and preprocessing

`[landsat]` accepts `manifest_file`, `wrs = [path, row]`, per-sensor `sensor_windows`, and a `gapfill` table. The JSON manifest has a `scenes` list with `scene_id`, `product_id`, `sensor`, ISO `date`, `path`, `row`, `cloud_cover`, and `file`; L7 records also have `gap_mask_file`. Files contain original C2 DN and named QA bands. The acquisition API verifies product ID, date, WRS and L2SP status against the service. Selection uses lowest catalogue cloud cover then immutable ID per date, overriding the generic workflow's sensor priority. Paths in a JSON scene manifest should be absolute.

`gapfill` specifies `start`, `end`, `max_days`, `max_donors`, `min_common`, `max_window` and `validation`. Donors are original LE07 observations from the same WRS. The development boundary rejects later donors. Source optical-mask absence within the footprint, corroborated by zero raw red/NIR, identifies eligible scan gaps; cloud and temperature-only holes remain unavailable. Native donor grids must have the same CRS and affine basis and integer pixel offsets.

The validation table specifies `years`, `stripe_shift_rows`, `border_pixels`, `crop_pixels`, `min_scored_pixels`, `minimum_fill_fraction`, `LST_RMSE_max_K`, `LST_absolute_bias_max_K`, `NDVI_RMSE_max`, and `minimum_passed_scenes`. The least-cloudy scene in each half-year is selected before testing. Passing requires the configured total and at least one scene from each year. Withheld thermal and NDVI errors must pass before any filled product enters retrieval. Written float32 originals are compared exactly; each band retains donor/method/window/sample-count fields. Input, product and acceptance identities are checked when consumed. See [gap-fill method](landsat-gapfill.md).

## Forcing and native acquisition

`[forcing.reconstruction]` specifies `acquisition_max_gap_days`, `original_max_gap_days`, `window_days`, `minimum_samples` and `boundary_carry`. Reconstruction is retrospective and occurs independently on either side of the development boundary. Explicit boundary carry uses only the same partition; it cannot supply a pre-launch L8 history. Raw scene support, original support, spatial filling, boundary extrapolation, extended acquisition gaps and reconstruction flags accompany the daily fields. Neither missing rain nor invalid forcing is turned into zero.

`[forcing.precipitation_qc]` specifies `small_negative_tolerance_mm` and `maximum_annual_correction_mm`. Only finite negatives within that bound become zero. Larger negative or nonfinite values become unavailable. Native corrections and flags are retained separately from the unchanged raw download, with an annual per-cell acceptance check.

`[forcing.retrieval]` specifies `acquisition_extent`, `meteorology_extent`, `meteorology_raw_dir`, `meteorology_stack_dir`, `meteorology_end`, `climatology_start`, `climatology_end`, `dem_file`, `scene_cache_dir`, `template_file`, a `meteorology_files` mapping and a `padding` table. Required meteorology mappings are `tmax`, `tmin`, `rs`, `ea`, `et_tall_crop` and `precipitation`, pointing to explicit native daily NetCDF files. Dates must cover both actual simulation and complete development climatology years. The preparation stage uses monthly containers of original ERA5 daily bands, expanding temporary daily files under `PBS_JOBFS` or `TMPDIR`. DEM coverage includes the full exported ERA5 cell footprints. Downloads record the original source grid and checksum; verified samples can be reused by the complete acquisition.

The padding table contains `primary_bounds`, `comparison_bounds`, their `crs`, `maximum_fallback_fraction`, `minimum_agreement` and `maximum_p99_difference`. It screens cold-anchor fallback and compares fixed retrieval windows on the actual preparation grid. Rejected scenes contribute no ET observations. The dT climatology evaluates each development year's matching month/day before averaging; February 29 uses February 28 and March 1 from every year. It never uses holdout meteorology. Set `forcing.retrieval.require_supported_start = true` for a workflow whose start must be the first accepted observation. If the requested start is earlier, retrieval writes `scene-calendar.json` and stops before reconstruction so the scientifically supported start can be frozen. Both sensor workflows should select the same native `template_file` and shared cache location. Cache identities include inputs, padding, climatology, software and output template.

## Paired support and initialization

The usual Landsat 8/9 workflow runs alone. `forcing.peer_runs` is optional and should be set only for an explicitly requested comparison. For runs beginning in 2013, the default calibration scoring start is 2014-01-01, so available 2013 forcing supplies initialisation. This calendar choice does not itself establish convergence. The repeated-year protocol below is an additional explicit option; it retains its complete-year and convergence requirements. Use a different complete development year when 2013 coverage is incomplete.

`forcing.peer_runs` lists the other run directories for a paired comparison. **Preprocess both runs before calibrating either.** Each run writes its own complete development forcing/physical-soil mask on both preparation grids. Initialization intersects corresponding masks with exact coordinates and CRS, preserving missing holdout forcing without using it to select cells. OpenLandMap's optional `soil.maximum_organic_matter_percent` also rejects hydraulic estimates outside the configured pedotransfer applicability domain.

```toml
[spinup]
source_year = 2014
repeats = 18
tolerance = 0.005
minimum_fraction = 0.99
solver_backend = "numba"
```

A source year must be complete, non-leap and inside development. Its repeated days rotate to finish immediately before the actual start's calendar phase. Artificial reference observations remain missing. Wet and dry initial states are simulated through the day before scoring; every layer must agree within tolerance in the required fraction of admitted cells. The gate runs for default and calibrated parameters on both coarse and fine grids. Failure requires extending development-only initialization and refitting. Repetition establishes numerical convergence, not the true historical state.

Normal simulations initialize each layer at physical field capacity (wilting point plus available water capacity). The default `spinup.convergence_mode = "wet_dry"` tests independence from empty and saturated initial profiles. An explicitly selected `convergence_mode = "field_capacity_cycles"` instead compares field-capacity starts after N and 2N repeated years, followed by the identical real initialization prefix. It retains the configured layer tolerance, admitted-cell fraction and coarse/fine default/calibrated gates. This narrower test assesses warm-up length under the chosen field-capacity initialization; passing does **not** establish independence from extreme initial states. Receipts identify the mode and its interpretation. The production trajectory still uses N repeats.

Only initialization dates are removed from final products. An `initial_history_valid_<scenario>` field preserves their validity, so evaluation still requires a continuous trajectory. Actual states continue through the development/holdout boundary without resetting.

## Benchmark, fit and verify

The optional `benchmark` stage loads the exact calibration problem and measures warmed objectives plus two population evaluations. It writes timing, input/software identity, cell-day counts and conditional generation projections. It saves no fitted parameters. Run it within an appropriate compute allocation before choosing a production allocation.

```sh
pysweb workflow --config site.toml --stages benchmark
pysweb workflow --config site.toml --stages calibrate
pysweb workflow --config site.toml --stages run
```

Set `calibration.serial_polish = true` to preserve a converged `calibration_de.csv` and run serial bounded L-BFGS-B polishing. Unsuccessful or worsening polishing retains the DE vector. The lower-level `pysweb.swb.calibrate(polish_file=..., **same_inputs)` can use a separate allocation; input/config/software identities must match, while workers and output path can differ. Bounds, bound hits, convergence and polishing provenance are retained.

Coarse paired simulations independently reproduce the area-weighted cell-day loss and observation count. The acceptance receipt binds the parameter CSV, development files, paired support, scientific settings and package source. Changed inputs invalidate acceptance. Final numerical gates check complete admitted development histories, bounds, budget closure and profile-storage differences. Final SEB/SWB CRS, affine, shape and coordinates must match exactly. Fine-to-coarse transfer remains a diagnostic and never retunes parameters.

## Evaluation

`[evaluation]` contains the fixed `longitude`, `latitude`, `depth_interval_mm`, `scoring_start`, `observations_file`, optional `regimes_file`/`regime_column`, `pairs` and `phases`. Each pair names a model column, observed column and units; phases map names to inclusive ISO date pairs. Site labels, depth rules and scoring windows are configuration, not package constants.

Point extraction transforms the fixed coordinate into the output CRS, uses its nearest native cell and requires membership in the frozen support mask. It reports layer start states and daily endpoint means, using stored final state for the last endpoint. Storage weights are exact layer overlaps with the configured interval. Evaluation requires cumulative forcing validity and reports own/common-date annual and pooled sample counts, RMSE, MAE, bias, unbiased RMSE and Pearson correlation. Undefined statistics remain missing. `aggregate_probe_storage` handles explicit fixed-probe weights, joint QC and sample thresholds; `evaluate_series` accepts arbitrary workflow/scenario dictionaries for cross-workflow common-date comparisons. The package does not infer irrigation labels or make a point observation representative of a whole GSSM cell.

`pysweb.forcing.write_seasonal_seb` writes the required same-parameter control forcing in bounded spatial blocks. Its circular 365-day ET-fraction climatology uses only configured development dates, a fixed ±15-day neighbourhood by default, and at least two distinct contributing years per pixel/bin. February 29 averages adjacent climatology bins. Actual NDVI, component partition and reference ET remain unchanged. Feed that separate control SEB through the same native preprocessing, convergence and simulation APIs; reuse primary parameters for the fixed control. Any separately calibrated control must receive its own fit and acceptance receipts.

For the revised SWB, set `model.reference_file` to the frozen GSSM reference JSON and retain the same file for both scenarios and all initialisation gates. `model.max_root_depth` is fixed at 1000 mm by default. Calibrations contain only diffusivity and root beta; obsolete max/min bounds and factors raise migration errors. Native preparation freezes the reference population before initialisation.
