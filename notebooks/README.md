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

GEE_PROJECT=your-gee-project bash workflows/ssebop_runner_landsat.sh Llara --workers 40
bash workflows/sweb_domain_runner.sh Llara --workers 40
```

Those wrapper scripts remain the easiest end-to-end path when you want the same run-subdirectory workflow as before.

### From a notebook
You can drive the project from a notebook in three ways.

#### 1. Call the shell wrappers
Use this when you want the old operational pattern with minimal changes:

```python
!GEE_PROJECT=your-gee-project bash workflows/ssebop_runner_landsat.sh Llara --workers 40
!bash workflows/sweb_domain_runner.sh Llara --workers 40
```

This is the closest equivalent to the previous `/g/data/ym05/sweb_model/code/operations` workflow. `GEE_PROJECT` must be set whenever the SSEBop wrapper runs Step 1.

#### 2. Call the workflow CLIs directly
Use this when you want more explicit control over paths like `gee_project`, prepared DEM location, or output directories:

```python
!python workflows/1_ssebop_prepare_inputs.py \
    --date-range "2024-01-01 to 2024-01-31" \
    --extent "147.20,-35.10,147.30,-35.00" \
    --met-source era5land \
    --gee-project your-gee-project \
    --out-dir /path/to/run_inputs

!python workflows/2_ssebop_run_model.py \
    --date-range "2024-01-01 to 2024-01-31" \
    --landsat-dir /path/to/run_inputs/landsat \
    --met-dir /path/to/run_inputs/met/era5land/stack \
    --dem /path/to/run_inputs/dem/nasadem.tif \
    --output-dir /path/to/ssebop_outputs
```

Step 1 now prepares the DEM artifact under `/path/to/run_inputs/dem/nasadem.tif`, and Step 2 should use that prepared file rather than an unrelated external DEM path.

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

### `gee_project` note
The current package/workflow path requires an explicit Earth Engine project when Landsat, ERA5-Land, NASADEM, OpenLandMap, or reference SSM downloads are involved. Pass `gee_project` explicitly when using:

- `workflows/1_ssebop_prepare_inputs.py`
- `pysweb.ssebop.prepare_inputs(...)`
- `pysweb.swb.preprocess(...)`

## Current notebooks
- `01_run_pysweb.ipynb`: Canonical run example showing package-backed SSEBop plus SWB preprocess/calibrate/run usage from a notebook with `import pysweb`.
- `02_plot_heatmap.ipynb`: Heatmap plotting walkthrough using the plotting modules exposed through `pysweb.visualisation` (legacy `visualisation/plot_heatmap.py` wrapper still available).
- `03_plot_time_series.ipynb`: SSEBop + SWEB time-series plotting walkthrough using the plotting modules exposed through `pysweb.visualisation` (legacy `visualisation/plot_time_series.py` wrapper still available).

## Planned notebooks
- More complete end-to-end and analysis examples built on the package-backed SWB APIs.
