# Running PySWEB

## Shared configuration

Use `pysweb workflow --config study.toml --dry-run` to check a configuration before execution. The committed TOML examples describe the accepted sections. Paths are relative to the TOML file, and dates are inclusive. Native outputs are `<run.root>/<run.name>/{inputs,ssebop,calibration_inputs,run_inputs,products}`; legacy runs retain `swb_inputs` and `swb`. Each run saves its resolved configuration. Study data can live on a separate filesystem.

The CLI and `notebooks/01_run_pysweb.ipynb` both call `load_config` and `run_config` in `pysweb.workflow`. Shell launchers use the same implementation:

```bash
bash workflows/ssebop_runner_landsat.sh --config study.toml
bash workflows/sweb_domain_runner.sh --config study.toml
bash spec/sweb_mlcons_runner.sh --config study.toml
```

Set `PYTHON` to the desired interpreter when using these launchers. They work from any working directory and do not use GNU `date` or embed institutional paths. The `spec` shell launcher follows the soil source in the configuration; set it explicitly to `mlcons` or `custom`.

## Earth Engine and offline execution

Run `earthengine authenticate` using your supported Earth Engine account. Set `run.gee_project` to a project you can use. The prepare stage requires internet access. OpenLandMap soil acquisition and remote reference SSM acquisition also require Earth Engine. No account or project is needed by the demo, custom/MLConstraints preprocessing with a local reference (or `skip_reference_ssm=True`), calibration of existing files, or the SWB simulation.

On Gadi, download and preprocess online inputs on a network-enabled host. Submit local-data stages through your normal PBS environment, with `workers` matched to the CPU allocation. A configured OpenLandMap preprocess still needs network access: offline PBS jobs should begin at `calibrate,run` after preprocessing, or use local soil data. The run notebook keeps imports and execution toggles at the front and imports plotting dependencies only when diagnostics are enabled. No scheduler resources are requested automatically.

## Individual stages and Python APIs

```bash
pysweb ssebop-prepare --help
pysweb ssebop-run --help
pysweb swb-preprocess --help
pysweb swb-calibrate --help
pysweb swb-run --help
pysweb time-series --help
pysweb heatmap --help
```

The corresponding public APIs are `pysweb.ssebop.prepare_inputs`, `pysweb.ssebop.run`, `pysweb.swb.preprocess`, `pysweb.swb.calibrate` and `pysweb.swb.run`. API keyword arguments use underscores in place of CLI hyphens. Individual stages accept existing paths, so older prepared-data layouts can be reused when they satisfy the current contracts.

See [daily water-balance heatmaps](heatmaps.md) for the aligned flux, soil-moisture
and storage panels, the Python figure API, and optional study-phase/event shading.

```python
import pysweb

pysweb.swb.preprocess(
    start_date="2024-01-01", end_date="2024-01-14",
    extent=[148.0, -35.02, 148.02, -35.0], sm_res=0.01,
    rain_file="examples/outputs/demo/source/rain.nc", rain_var="precipitation",
    et_file="examples/outputs/demo/source/et.nc", et_var="ET", transpiration_var="E_t",
    soil_source="custom", soil_file="examples/outputs/demo/source/custom_soil.nc",
    skip_reference_ssm=True, output_dir="examples/outputs/custom-prepared",
)
```

Run `pysweb demo` first to generate those synthetic source files. `sm_res` is in degrees for the geographic preprocessing grid; finer output spacing does not create finer source information.

## Calibration and persistence

### WorldCover FANO inputs

SSEBop defaults to `tcold_method="worldcover_fano"`. `ssebop-prepare` downloads a native WorldCover v200 chip to `dem/worldcover_v200_2021.tif`, covering the same padded acquisition extent as Landsat. The native workflow passes this path automatically. For standalone execution, supply `--landcover <prepared>/dem/worldcover_v200_2021.tif`. Existing inputs without land cover require this additional preparation before retrieval.

An optional `[ssebop] landcover = "path/to/worldcover_v200_2021.tif"` selects an existing raster (resolved relative to the TOML). In the configured scene workflow, `forcing.retrieval.landcover_file` is also supported; `[ssebop].landcover` takes precedence. Without either override, the prepared file is placed beside `forcing.retrieval.dem_file`. Preparation preserves explicitly supplied existing land-cover rasters. They must contain WorldCover v200 class codes and cover all valid pixels of every downloaded scene; the loader reads only the required spatial window. Acquisition follows the existing bounded direct-download contract; requests exceeding Earth Engine's download limits must be split into smaller study regions.

Use `tcold_method="legacy"` explicitly when reproducing previous land-cover-unaware retrievals. `tcold_smooth_scale_m` applies only to that method. The new method retains the existing coefficient, NDVI threshold, fine-scale and coarse-scale options, and adds `tcold_smooth_resolution_m=90` and `tcold_smooth_radius_pixels=5`. Method/settings, code and land-cover contents participate in retrieval cache identity. Recompute SSEBop forcing and recalibrate after changing methods; do not treat old calibrated parameters as a fit to the new forcing. See [the algorithm and crosswalk](methods.md#land-cover-aware-fano-cold-boundary).

Calibration is opt-in. Set its dates, random seed and spin-up exclusion in `[calibration]`. A non-converged optimizer fails unless `allow_unconverged=true` is explicit; the saved manifest records convergence, objective, sample count and optimizer status. The native workflow uses GSSM cell-day errors. Individual legacy calibration calls retain optional `sm_res` block coarsening and `objective="domain_mean"` for deliberate reproduction; these options are not the native workflow defaults.

The workflow automatically supplies the calibrated CSV to the run. Its checksum is included in run provenance. The calibrated parameters are `diff_factor` and `root_beta`. Bounds and units are available in CLI help. Defaults are research starting values, and independent validation remains necessary.

NetCDF products are written to temporary files, read back and atomically promoted. Run manifests validate code, configuration and source checksums before reuse. Earth Engine downloads use checksums and readable raster/CRS checks before reuse; incomplete downloads fail the stage. Multi-collection same-day collisions retain the first successful collection in configured order, recorded in the download manifest.

The grid solver loads forcing arrays in memory and distributes rows over processes. Memory demand grows with dates × cells × layers, plus budget outputs. Begin with a small extent and one worker before larger runs. The API's `time_step` must be one day for gridded workflows; direct solver calls additionally check timestamp cadence.

Component outputs use `E_s` (soil evaporation), `E_t` (transpiration) and `f_t` (transpiration fraction). See the [notation migration and legacy-input policy](migration.md#component-notation-migration-11-october-2026).

## Native-grid configured runs

The current example uses the [native-grid workflow](native-grids.md): retain original download grids, calibrate on native GSSM cells, and simulate default/calibrated SWB on one Landsat UTM grid shared with the final SEB output. Its calibration scoring interval must be inside the continuous run interval. The earlier independent-window behavior described above remains the explicit legacy workflow for configurations with `run.resolution`.

The [revised SWB contract](swb-formulation.md) defines fixed GSSM references, physical storage limits and explicit rejection of old four-parameter fits.
