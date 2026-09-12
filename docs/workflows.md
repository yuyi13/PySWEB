# Running PySWEB

## Shared configuration

Use `pysweb workflow --config study.toml --dry-run` to check a configuration before execution. The committed TOML examples describe the accepted sections. Paths are relative to the TOML file, and dates are inclusive. Outputs are `<run.root>/<run.name>/{inputs,ssebop,swb_inputs,swb}`. Each run saves its resolved configuration. Study data can live on a separate filesystem.

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

```python
import pysweb

pysweb.swb.preprocess(
    start_date="2024-01-01", end_date="2024-01-14",
    extent=[148.0, -35.02, 148.02, -35.0], sm_res=0.01,
    rain_file="examples/outputs/demo/source/rain.nc", rain_var="precipitation",
    et_file="examples/outputs/demo/source/et.nc", et_var="ET", t_var="T",
    soil_source="custom", soil_file="examples/outputs/demo/source/custom_soil.nc",
    skip_reference_ssm=True, output_dir="examples/outputs/custom-prepared",
)
```

Run `pysweb demo` first to generate those synthetic source files. `sm_res` is in degrees for the geographic preprocessing grid; finer output spacing does not create finer source information.

## Calibration and persistence

Calibration is opt-in. Set its dates, random seed and spin-up exclusion in `[calibration]`. A non-converged optimizer fails unless `allow_unconverged=true` is explicit; the saved manifest records convergence, objective, sample count and optimizer status. Native-grid calibration is the default. Optional `sm_res` coarsening requires exact whole blocks, propagates missing values and can change the fitted parameters through aggregation of nonlinear processes.

The workflow automatically supplies the calibrated CSV to the run. Its checksum is included in run provenance. The calibrated parameters are `diff_factor`, `sm_max_factor`, `sm_min_factor` and `root_beta`. Bounds and units are available in CLI help. Defaults are research starting values, and independent validation remains necessary.

NetCDF products are written to temporary files, read back and atomically promoted. Run manifests validate code, configuration and source checksums before reuse. Earth Engine downloads use checksums and readable raster/CRS checks before reuse; incomplete downloads fail the stage. Multi-collection same-day collisions retain the first successful collection in configured order, recorded in the download manifest.

The grid solver loads forcing arrays in memory and distributes rows over processes. Memory demand grows with dates × cells × layers, plus budget outputs. Begin with a small extent and one worker before larger runs. The API's `time_step` must be one day for gridded workflows; direct solver calls additionally check timestamp cadence.
