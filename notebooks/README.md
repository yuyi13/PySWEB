# PySWEB Notebooks

Notebook examples for common PySWEB workflows and result visualisation.

## Running guidance
These notebooks are mainly for inspection, plotting, and lightweight workflow control. They are not yet a complete replacement for the CLI and shell-runner paths.

If you were previously running the model from an operations directory like:

```bash
cd /g/data/ym05/sweb_model/code/operations
./ssebop_landsat_runner.sh Llara --workers 40
./sweb_domain_runner.sh Llara --workers 40
```

the closest current equivalents in this repository are:

```bash
cd /path/to/PySWEB

bash workflows/ssebop_runner_landsat.sh Llara --workers 40
bash workflows/sweb_domain_runner.sh Llara --workers 40
```

Those wrapper scripts remain the easiest end-to-end path when you want the same run-subdirectory workflow as before.

### From a notebook
You can drive the project from a notebook in three ways.

#### 1. Call the shell wrappers
Use this when you want the old operational pattern with minimal changes:

```python
!bash workflows/ssebop_runner_landsat.sh Llara --workers 40
!bash workflows/sweb_domain_runner.sh Llara --workers 40
```

This is the closest equivalent to the previous `/g/data/ym05/sweb_model/code/operations` workflow.

#### 2. Call the workflow CLIs directly
Use this when you want more explicit control over paths like `gee_config`, `dem`, or output directories:

```python
!python workflows/1_ssebop_prepare_inputs.py \
    --date-range "2024-01-01 to 2024-01-31" \
    --extent "147.20,-35.10,147.30,-35.00" \
    --met-source era5land \
    --gee-config /path/to/base_gee.yaml \
    --dem /path/to/dem.tif \
    --out-dir /path/to/run_inputs

!python workflows/2_ssebop_run_model.py \
    --date-range "2024-01-01 to 2024-01-31" \
    --landsat-dir /path/to/run_inputs/landsat \
    --met-dir /path/to/run_inputs/met/era5land/stack \
    --dem /path/to/dem.tif \
    --output-dir /path/to/ssebop_outputs
```

For the SWB side, the workflow CLIs remain convenient notebook-facing wrappers, but the underlying package APIs are now wired as well.

#### 3. Import `pysweb` directly
Use this when you want notebook-native Python control over the package-backed workflow. See `01_run_pysweb.ipynb` for the canonical run example.

At present:

- `pysweb.ssebop.prepare_inputs` is wired.
- `pysweb.ssebop.run` is wired.
- `pysweb.swb.preprocess` is wired.
- `pysweb.swb.calibrate` is wired.
- `pysweb.swb.run` is wired.

So if you want a pure notebook-driven flow today, you can drive both SSEBop and SWB directly from `pysweb`, while still falling back to the workflow CLIs or shell wrappers when that is more convenient for path-heavy operational runs. For plotting-oriented notebooks and library imports, prefer `pysweb.visualisation`.

### `gee_config` note
If you previously relied on a fixed operations path such as `/g/data/ym05/sweb_model/code/gee_config`, the current package/workflow path does not assume that location automatically. Pass the config path explicitly when using:

- `workflows/1_ssebop_prepare_inputs.py`
- `pysweb.ssebop.prepare_inputs(...)`

## Current notebooks
- `01_run_pysweb.ipynb`: Canonical run example showing package-backed SSEBop plus SWB preprocess/calibrate/run usage from a notebook with `import pysweb`.
- `02_plot_heatmap.ipynb`: Heatmap plotting walkthrough using the plotting modules exposed through `pysweb.visualisation` (legacy `visualisation/plot_heatmap.py` wrapper still available).
- `03_plot_time_series.ipynb`: SSEBop + SWEB time-series plotting walkthrough using the plotting modules exposed through `pysweb.visualisation` (legacy `visualisation/plot_time_series.py` wrapper still available).

## Planned notebooks
- More complete end-to-end and analysis examples built on the package-backed SWB APIs.
