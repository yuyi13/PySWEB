<img src="SWEB_logo.png" alt="SWEB logo" align="right" width="180" />

# The Sydney Soil Water-Energy Balance (SWEB) Model (work in progress)

[![Python](https://img.shields.io/badge/Python-3.12+-306998?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Preprint](https://img.shields.io/badge/Preprint-Release%20Soon-0072BC?style=flat)]()
[![Dataset](https://img.shields.io/badge/Dataset-Release%20Soon-1682D4?style=flat)]()

Python workflows for generating root-zone soil moisture from gridded precipitation, evapotranspiration, and soil hydraulic properties. The current meteorology pathway is ERA5-Land-based and globally usable; SWB soil inputs now default to Earth Engine OpenLandMap, and the reference SSM path now defaults to `gssm1km` from `users/qianrswaterr/GlobalSSM1km0509`. The repository is under active development; interfaces and defaults may change.

The package-first refactor is now the main execution path. The canonical code layout lives under `pysweb/`, while `workflows/` keeps thin CLI entrypoints and convenience wrappers around those package modules. Legacy support code remains under `core/`, but the main SSEBop and SWB prepare/preprocess/calibrate/run paths now route through `pysweb/`.

## Current repository structure
```
PySWEB/
├── pysweb/                                # Canonical package code
│   ├── io/                                # Shared I/O helpers
│   ├── met/                               # Meteorology path resolution and source-specific helpers
│   │   ├── era5land/
│   │   └── silo/
│   ├── soil/                              # Canonical soil-source discovery and loading logic
│   ├── ssebop/                            # Package-backed SSEBop prepare/run logic
│   │   ├── api.py
│   │   └── inputs/
│   ├── swb/                               # Package-backed SWB preprocess/calibrate/run logic
│       ├── __init__.py
│       ├── api.py
│       ├── calibrate.py
│       ├── core.py
│       ├── preprocess.py
│       ├── run.py
│       └── solver.py
│   └── visualisation/                     # Canonical plotting modules for notebooks and scripts
│
├── workflows/                             # CLI entrypoints and convenience wrappers
│   ├── 1_ssebop_prepare_inputs.py         # Unified first SSEBop step: Landsat + meteorology preparation
│   ├── 1b_download_era5land_daily.py      # Standalone ERA5-Land download utility
│   ├── 1c_stack_era5land_daily.py         # Standalone ERA5-Land stack utility
│   ├── 2_ssebop_run_model.py              # Thin CLI wrapper over the package-backed SSEBop run workflow
│   ├── 3_sweb_preprocess_inputs.py        # Thin CLI wrapper over the package-backed SWB preprocess workflow
│   ├── 4_sweb_calib_domain.py             # Thin CLI wrapper over the package-backed SWB calibration workflow
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
├── visualisation/                         # Legacy wrapper scripts around pysweb.visualisation during the transition
│   ├── plot_time_series.py                # Time-series extraction + plots for SSEBop and SWEB outputs
│   └── plot_heatmap.py                    # Heatmap plotting for SWEB layers (optionally with SSEBop forcing panel)
│
├── notebooks/                             # Example Jupyter notebooks
│   ├── README.md                          # Notebook index and scope
│   ├── 01_run_pysweb.ipynb                # Canonical notebook run example for SSEBop + SWB
│   ├── 02_plot_heatmap.ipynb              # Heatmap plotting walkthrough
│   └── 03_plot_time_series.ipynb          # Time-series plotting walkthrough
│
├── README.md
└── SWEB_logo.png
```

Runtime outputs are written under the unified prepared-input layout rooted at `1_ssebop_inputs/` plus `2_ssebop_outputs/`, `3_sweb_inputs/`, and `4_sweb_outputs/`. The legacy `1_era5land_raw/` and `1_era5land_stacks/` folders are still used when the standalone ERA5-Land utilities are run directly.

## Workflow overview
1. `workflows/1_ssebop_prepare_inputs.py`: unified first SSEBop step. It prepares Landsat inputs and meteorology products together, writing Landsat to `out_dir/landsat` and ERA5-Land outputs to `out_dir/met/era5land/{raw,stack}`.
2. `workflows/2_ssebop_run_model.py`: thin CLI wrapper over the package-backed SSEBop run workflow. It consumes the prepared Landsat directory plus a meteorology stack directory (for example `out_dir/met/era5land/stack`).
3. `workflows/3_sweb_preprocess_inputs.py`: thin wrapper over `pysweb.swb.preprocess`; it aligns ERA5-Land precipitation, SSEBop `E/T/ET`, OpenLandMap soil properties, and optional `gssm1km` reference SSM to one grid.
4. `workflows/4_sweb_calib_domain.py`: thin wrapper over `pysweb.swb.calibrate`; it calibrates domain-wide SWEB parameters (`diff_factor`, `sm_max_factor`, `sm_min_factor`, `root_beta`).
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

For notebook-driven runs, start with `notebooks/01_run_pysweb.ipynb`. For plotting from Python, prefer `pysweb.visualisation`. The legacy `visualisation/*.py` wrappers remain available during the transition:

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

The canonical package imports for those plotting paths are `pysweb.visualisation.plot_time_series` and `pysweb.visualisation.plot_heatmap`.

Both wrapper scripts currently include environment-specific default paths (for example `/g/data/...`) near the top of each script. Update those values before running on another machine or filesystem.

The meteorology path is now ERA5-Land-based and globally usable. SWB soil texture/SOC inputs now default to Earth Engine OpenLandMap, and the reference SSM input now defaults to `gssm1km` from `users/qianrswaterr/GlobalSSM1km0509`.

## Key outputs
- From the unified first SSEBop step (`1_ssebop_prepare_inputs.py`): a prepared run directory containing `landsat/`, `met/era5land/raw/`, and `met/era5land/stack/`. The stack directory holds `precipitation_daily_<start>_<end>.nc`, `tmax_daily_<start>_<end>.nc`, `tmin_daily_<start>_<end>.nc`, `rs_daily_<start>_<end>.nc`, `ea_daily_<start>_<end>.nc`, and `et_short_crop_daily_<start>_<end>.nc`.
- From SSEBop run (`2_ssebop_run_model.py`): `et_daily_ssebop_<start>_<end>.nc` plus intermediate `etf`/`ndvi` products, driven by the prepared meteorology stack directory.
- From SWEB preprocess (`3_sweb_preprocess_inputs.py`): `rain_daily_*.nc`, `effective_precip_daily_*.nc`, `et_daily_*.nc`, `t_daily_*.nc`, `soil_*.nc`, and optionally `reference_ssm_daily_*.nc`. When invoked via `sweb_domain_runner.sh`, precipitation is sourced from the unified prepared stack first and only falls back to the legacy stack directory if needed.
- From calibration (`4_sweb_calib_domain.py`): CSV with calibrated domain parameters.
- From SWEB run (`5_sweb_run_model.py`): consolidated RZSM NetCDF, optionally split into burn-in and post-burn products by `sweb_domain_runner.sh`.
- From visualisation helpers (`visualisation/plot_time_series.py`, `visualisation/plot_heatmap.py`):
  PNG plots and optional extracted CSV tables.

## Requirements
- Python 3.12+ (recommended)
- Core packages: `numpy`, `pandas`, `xarray`, `rioxarray`, `rasterio`, `netCDF4`, `scipy`, `pyproj`, `pyyaml`, `tqdm`
- System libs for geospatial reprojection: GDAL/PROJ
- Access to forcing and ancillary datasets (Landsat, ERA5-Land, DEM, SSEBop-ready ET products)
- Google Earth Engine access for OpenLandMap and `users/qianrswaterr/GlobalSSM1km0509` (for example with project `yiyu-research`)

## Naming convention
This repository uses:

- `pysweb/` for the canonical package code.
- `workflows/` for runnable CLI entry points and convenience wrappers around package or workflow-owned implementations.
- `core/` for legacy/reused modules that are still being folded into the package layout.

For soil-source logic, `pysweb.soil` is now the canonical package location. Keep new soil backend selection and loading changes there, with workflows and wrappers delegating into that package.

## Development guidance
Use this rule of thumb while the package-first refactor is still in progress:

- Prefer `pysweb/` for new reusable logic and for revisions to already-migrated functionality.
- Keep `workflows/` thin. Update them when CLI arguments, orchestration, or wrapper behavior changes.
- Keep `core/` for now, but treat it as transitional implementation substrate rather than the long-term public interface.

In practice:

- SSEBop prepare/run changes should usually go into `pysweb.ssebop` first.
- SWB run changes should usually go into `pysweb.swb` first.
- SWB preprocess/calibration are now package-backed entry points, so edits should usually go into `pysweb.swb.preprocess` and `pysweb.swb.calibrate` first, with `workflows/3_sweb_preprocess_inputs.py` and `workflows/4_sweb_calib_domain.py` kept as thin wrappers.
- If `pysweb/` only wraps a `core/` implementation today, either update that `core/` implementation carefully or finish migrating it into `pysweb/` rather than maintaining two divergent implementations.

### Current `core/` status
Keep these `core/` modules for now because they still provide real implementation value or compatibility coverage:

- `core/gee_downloader.py`: still the backend implementation behind `pysweb.io.gee`.
- `core/swb_model_1d.py`: now legacy/reference solver code; the package-owned runtime path uses `pysweb.swb.solver`.
- `core/soil_hydra_funs.py`: legacy hydraulic helper module coupled to `core/swb_model_1d.py`.

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
- `core/swb_model_1d.py`
- `core/soil_hydra_funs.py`

Review `core/thomas_solve_tridiagonal_matrix.py` separately before removing it. It looks legacy, but it should be retired only after confirming there are no remaining external consumers.

### Workflow naming note
For the SSEBop first step, keep `workflows/1_ssebop_prepare_inputs.py` as the canonical entrypoint.

- `prepare_inputs` is the current package/API naming.
- `workflows/1_ssebop_preprocess_inputs.py` is a legacy duplicate name and should not be the preferred script for future development.

## Development status
This repository is actively evolving. Verify file paths, date ranges, and spatial settings before running large jobs.

`pysweb.ssebop.prepare_inputs`, `pysweb.ssebop.run`, `pysweb.swb.preprocess`, `pysweb.swb.calibrate`, and `pysweb.swb.run` are wired today. `workflows/2_ssebop_run_model.py`, `workflows/3_sweb_preprocess_inputs.py`, `workflows/4_sweb_calib_domain.py`, and `workflows/5_sweb_run_model.py` are thin wrappers over those package modules.

The `notebooks/` directory currently contains:

- `01_run_pysweb.ipynb`: canonical notebook run example using `import pysweb` for SSEBop plus SWB preprocess/calibrate/run
- `02_plot_heatmap.ipynb`: heatmap plotting walkthrough, with plotting modules exposed through `pysweb.visualisation`
- `03_plot_time_series.ipynb`: SSEBop + SWEB time-series plotting walkthrough, with plotting modules exposed through `pysweb.visualisation`
