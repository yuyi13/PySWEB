<img src="SWEB_logo.png" alt="SWEB logo" align="right" width="180" />

# The Sydney Soil Water-Energy Balance (SWEB) Model (work in progress)

[![Python](https://img.shields.io/badge/Python-3.12+-306998?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Preprint](https://img.shields.io/badge/Preprint-Release%20Soon-0072BC?style=flat)]()
[![Dataset](https://img.shields.io/badge/Dataset-Release%20Soon-1682D4?style=flat)]()

Python workflows for generating root-zone soil moisture from gridded precipitation, evapotranspiration, and soil hydraulic properties. The current meteorology pathway is ERA5-Land-based and globally usable; soil and SMAP/reference pieces still use the current downstream defaults. The repository is under active development; interfaces and defaults may change.

The package-first refactor is underway. The canonical code layout now lives under `pysweb/`, while `workflows/` keeps thin CLI entrypoints and convenience wrappers around that package code where the refactor is already wired. Some SWB steps still run as workflow-owned scripts during this transition.

## Current repository structure
```
PySWEB/
├── pysweb/                                # Canonical package code
│   ├── io/                                # Shared I/O helpers
│   ├── met/                               # Meteorology path resolution and source-specific helpers
│   │   ├── era5land/
│   │   └── silo/
│   ├── ssebop/                            # Package-backed SSEBop prepare/run logic
│   │   ├── api.py
│   │   └── inputs/
│   └── swb/                               # Package-backed SWB run logic and transitional API surface
│       ├── api.py
│       └── run.py
│
├── workflows/                             # CLI entrypoints and convenience wrappers
│   ├── 1_ssebop_prepare_inputs.py         # Unified first SSEBop step: Landsat + meteorology preparation
│   ├── 1b_download_era5land_daily.py      # Standalone ERA5-Land download utility
│   ├── 1c_stack_era5land_daily.py         # Standalone ERA5-Land stack utility
│   ├── 2_ssebop_run_model.py              # Thin CLI wrapper over the package-backed SSEBop run workflow
│   ├── 3_sweb_preprocess_inputs.py        # Workflow-owned preprocessing CLI
│   ├── 4_sweb_calib_domain.py             # Workflow-owned domain calibration CLI
│   ├── 5_sweb_run_model.py                # Thin CLI wrapper over the package-backed SWB run workflow
│   ├── ssebop_runner_landsat.sh           # Convenience bash wrapper for Steps 1-2
│   └── sweb_domain_runner.sh              # Convenience bash wrapper for Steps 3-5
│
├── core/                                  # Legacy/reused modules still referenced during the refactor
│   ├── era5land_download_config.py        # Build ERA5-Land GEE download configs
│   ├── era5land_refet.py                  # Reference ET and meteorology helpers
│   ├── era5land_stack.py                  # Discover/sort ERA5-Land daily downloads
│   ├── met_input_paths.py                 # Resolve meteorology inputs for SSEBop
│   ├── swb_model_1d.py                    # 1-D layered soil water balance core
│   ├── soil_hydra_funs.py                 # Hydraulic process helpers and Richards-equation pieces
│   ├── ssebop_au.py                       # AU-focused SSEBop geospatial helper functions
│   ├── gee_downloader.py                  # Earth Engine data download utilities
│   └── thomas_solve_tridiagonal_matrix.py # Tridiagonal solver utility
│
├── visualisation/                         # Plotting and visualisation helpers
│   ├── plot_time_series.py                # Time-series extraction + plots for SSEBop and SWEB outputs
│   └── plot_heatmap.py                    # Heatmap plotting for SWEB layers (optionally with SSEBop forcing panel)
│
├── notebooks/                             # Example Jupyter notebooks
│   ├── README.md                          # Notebook index and scope
│   ├── 01_plot_heatmap.ipynb              # Heatmap plotting walkthrough
│   └── 02_plot_time_series.ipynb          # Time-series plotting walkthrough
│
├── README.md
└── SWEB_logo.png
```

Runtime outputs are written under the unified prepared-input layout rooted at `1_ssebop_inputs/` plus `2_ssebop_outputs/`, `3_sweb_inputs/`, and `4_sweb_outputs/`. The legacy `1_era5land_raw/` and `1_era5land_stacks/` folders are still used when the standalone ERA5-Land utilities are run directly.

## Workflow overview
1. `workflows/1_ssebop_prepare_inputs.py`: unified first SSEBop step. It prepares Landsat inputs and meteorology products together, writing Landsat to `out_dir/landsat` and ERA5-Land outputs to `out_dir/met/era5land/{raw,stack}`.
2. `workflows/2_ssebop_run_model.py`: thin CLI wrapper over the package-backed SSEBop run workflow. It consumes the prepared Landsat directory plus a meteorology stack directory (for example `out_dir/met/era5land/stack`).
3. `workflows/3_sweb_preprocess_inputs.py`: align ERA5-Land precipitation, SSEBop `E/T/ET`, soil properties, and optional SMAP SSM to one grid.
4. `workflows/4_sweb_calib_domain.py`: calibrate domain-wide SWEB parameters (`diff_factor`, `sm_max_factor`, `sm_min_factor`, `root_beta`).
5. `workflows/5_sweb_run_model.py`: thin CLI wrapper over the package-backed SWB run workflow.

The standalone `1b_download_era5land_daily.py` and `1c_stack_era5land_daily.py` utilities are still available, but they are no longer the primary first step for the package-backed SSEBop path.
The wrapper handoff follows the same contract: `ssebop_runner_landsat.sh` prepares meteorology under `1_ssebop_inputs/<run_subdir>/met/era5land/stack`, and `sweb_domain_runner.sh` consumes that location by default. `sweb_domain_runner.sh` falls back to `1_era5land_stacks/<run_subdir>` only for legacy standalone `1c_stack_era5land_daily.py` usage.

## Quick start
The primary entrypoints are the workflow CLIs in `workflows/`, which delegate into `pysweb/` where that package wiring exists:

```bash
python workflows/1_ssebop_prepare_inputs.py \
  --date-range "2024-01-01 to 2024-01-31" \
  --extent "147.20,-35.10,147.30,-35.00" \
  --met-source era5land \
  --gee-config /path/to/base_gee.yaml \
  --dem /path/to/dem.tif \
  --out-dir /path/to/run_inputs

python workflows/2_ssebop_run_model.py \
  --date-range "2024-01-01 to 2024-01-31" \
  --landsat-dir /path/to/run_inputs/landsat \
  --met-dir /path/to/run_inputs/met/era5land/stack \
  --dem /path/to/dem.tif \
  --output-dir /path/to/ssebop_outputs
```

Convenience wrappers remain available for environment-specific end-to-end runs, but they are wrappers around the workflow CLIs rather than the primary architecture:

```bash
# Step A: prepare Landsat + meteorology inputs and run SSEBop
bash workflows/ssebop_runner_landsat.sh <run_subdir>

# Step B: preprocess, calibrate, and run SWEB against the same run_subdir
bash workflows/sweb_domain_runner.sh <run_subdir>
```

In that wrapper sequence, the SWEB wrapper reads precipitation from `1_ssebop_inputs/<run_subdir>/met/era5land/stack` by default, so the handoff works without manually copying ERA5-Land stacks. If you still run the standalone ERA5-Land stack workflow, the SWEB wrapper can fall back to `1_era5land_stacks/<run_subdir>`.

Plotting and notebook helpers are unchanged:

```bash
python visualisation/plot_time_series.py \
  --run-subdir <run_subdir> \
  --output /g/data/ym05/sweb_model/figures/<run_subdir>_timeseries.png

python visualisation/plot_heatmap.py \
  --run-subdir <run_subdir> \
  --lat <latitude> --lon <longitude> \
  --output /g/data/ym05/sweb_model/figures/<run_subdir>_heatmap.png

python visualisation/plot_heatmap.py \
  --run-subdir <run_subdir> \
  --domain-mean \
  --output /g/data/ym05/sweb_model/figures/<run_subdir>_heatmap_domain.png

cd notebooks
jupyter notebook
```

Both wrapper scripts currently include environment-specific default paths (for example `/g/data/...`) near the top of each script. Update those values before running on another machine or filesystem.

The meteorology path is now ERA5-Land-based and globally usable. Soil texture/SOC rasters and SMAP/reference inputs still use the current downstream defaults and are not yet globalized.

## Key outputs
- From the unified first SSEBop step (`1_ssebop_prepare_inputs.py`): a prepared run directory containing `landsat/`, `met/era5land/raw/`, and `met/era5land/stack/`. The stack directory holds `precipitation_daily_<start>_<end>.nc`, `tmax_daily_<start>_<end>.nc`, `tmin_daily_<start>_<end>.nc`, `rs_daily_<start>_<end>.nc`, `ea_daily_<start>_<end>.nc`, and `et_short_crop_daily_<start>_<end>.nc`.
- From SSEBop run (`2_ssebop_run_model.py`): `et_daily_ssebop_<start>_<end>.nc` plus intermediate `etf`/`ndvi` products, driven by the prepared meteorology stack directory.
- From SWEB preprocess (`3_sweb_preprocess_inputs.py`): `rain_daily_*.nc`, `effective_precip_daily_*.nc`, `et_daily_*.nc`, `t_daily_*.nc`, `soil_*.nc`, and optionally `smap_ssm_daily_*.nc`. When invoked via `sweb_domain_runner.sh`, precipitation is sourced from the unified prepared stack first and only falls back to the legacy stack directory if needed.
- From calibration (`4_sweb_calib_domain.py`): CSV with calibrated domain parameters.
- From SWEB run (`5_sweb_run_model.py`): consolidated RZSM NetCDF, optionally split into burn-in and post-burn products by `sweb_domain_runner.sh`.
- From visualisation helpers (`visualisation/plot_time_series.py`, `visualisation/plot_heatmap.py`):
  PNG plots and optional extracted CSV tables.

## Requirements
- Python 3.12+ (recommended)
- Core packages: `numpy`, `pandas`, `xarray`, `rioxarray`, `rasterio`, `netCDF4`, `scipy`, `pyproj`, `pyyaml`, `tqdm`
- System libs for geospatial reprojection: GDAL/PROJ
- Access to forcing and ancillary datasets (Landsat, ERA5-Land, soil property rasters, optional SMAP-DS)

## Naming convention
This repository uses:

- `pysweb/` for the canonical package code.
- `workflows/` for runnable CLI entry points and convenience wrappers around package or workflow-owned implementations.
- `core/` for legacy/reused modules that are still being folded into the package layout.

## Development guidance
Use this rule of thumb while the package-first refactor is still in progress:

- Prefer `pysweb/` for new reusable logic and for revisions to already-migrated functionality.
- Keep `workflows/` thin. Update them when CLI arguments, orchestration, or wrapper behavior changes.
- Keep `core/` for now, but treat it as transitional implementation substrate rather than the long-term public interface.

In practice:

- SSEBop prepare/run changes should usually go into `pysweb.ssebop` first.
- SWB run changes should usually go into `pysweb.swb` first.
- SWB preprocess/calibration are still more workflow-owned today, so edits to `workflows/3_sweb_preprocess_inputs.py` and `workflows/4_sweb_calib_domain.py` are still expected until those paths are migrated.
- If `pysweb/` only wraps a `core/` implementation today, either update that `core/` implementation carefully or finish migrating it into `pysweb/` rather than maintaining two divergent implementations.

### Current `core/` status
Keep these `core/` modules for now because they still provide real implementation value:

- `core/gee_downloader.py`: still the backend implementation behind `pysweb.io.gee`.
- `core/swb_model_1d.py`: still used directly by `workflows/4_sweb_calib_domain.py`.
- `core/soil_hydra_funs.py`: still supports `core/swb_model_1d.py`.

These are mostly transitional, compatibility-oriented, or already superseded by `pysweb/` equivalents:

- `core/ssebop_au.py`: now mainly a compatibility shim over `pysweb.ssebop.*`.
- `core/era5land_download_config.py`: largely superseded by `pysweb.met.era5land.download`.
- `core/era5land_refet.py`: largely superseded by `pysweb.met.era5land.refet`.
- `core/era5land_stack.py`: largely superseded by `pysweb.met.era5land.stack`.
- `core/met_input_paths.py`: largely superseded by `pysweb.met.paths`.

These are the next retirement candidates once remaining tests, docs, and callers are updated:

- `core/ssebop_au.py`
- `core/era5land_download_config.py`
- `core/era5land_refet.py`
- `core/era5land_stack.py`
- `core/met_input_paths.py`

Review `core/thomas_solve_tridiagonal_matrix.py` separately before removing it. It looks legacy, but it should be retired only after confirming there are no remaining external consumers.

### Workflow naming note
For the SSEBop first step, keep `workflows/1_ssebop_prepare_inputs.py` as the canonical entrypoint.

- `prepare_inputs` is the current package/API naming.
- `workflows/1_ssebop_preprocess_inputs.py` is a legacy duplicate name and should not be the preferred script for future development.

## Development status
This repository is actively evolving. Verify file paths, date ranges, and spatial settings before running large jobs.

`pysweb.ssebop` and `pysweb.swb.run` are wired today. The `pysweb.swb.preprocess` and `pysweb.swb.calibrate` package APIs are not yet exposed as finished public interfaces, so use `workflows/3_sweb_preprocess_inputs.py` and `workflows/4_sweb_calib_domain.py` directly for those steps.
