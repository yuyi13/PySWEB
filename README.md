<img src="SWEB_logo.png" alt="SWEB logo" align="right" width="180" />

# The Sydney Soil Water-Energy Balance (SWEB) Model (work in progress)

[![Python](https://img.shields.io/badge/Python-3.12+-306998?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Preprint](https://img.shields.io/badge/Preprint-Release%20Soon-0072BC?style=flat)]()
[![Dataset](https://img.shields.io/badge/Dataset-Release%20Soon-1682D4?style=flat)]()

Python workflows for generating root-zone soil moisture from gridded precipitation, evapotranspiration, and soil hydraulic properties. The current meteorology pathway is ERA5-Land-based and globally usable; soil and SMAP/reference pieces still use the current downstream defaults. The repository is under active development; interfaces and defaults may change.

## Current repository structure
```
PySWEB/
├── workflows/                             # Pipeline entry points and orchestration scripts
│   ├── 1_ssebop_prepare_inputs.py         # Prepare GEE config and download Landsat inputs
│   ├── 1b_download_era5land_daily.py      # Download ERA5-Land DAILY_AGGR daily GeoTIFFs
│   ├── 1c_stack_era5land_daily.py         # Stack ERA5-Land rasters and derive daily forcing NetCDFs
│   ├── 2_ssebop_run_model.py              # Build ET/E/T products with Landsat + explicit ERA5-Land forcing
│   ├── 3_sweb_preprocess_inputs.py        # Harmonise rain/ET/soil/SMAP to a common grid
│   ├── 4_sweb_calib_domain.py             # Domain-wide parameter calibration against SMAP-DS SSM
│   ├── 5_sweb_run_model.py                # Spatial SWEB run using preprocessed forcings and soil inputs
│   ├── ssebop_runner_landsat.sh           # Two-step wrapper: download + SSEBop run
│   └── sweb_domain_runner.sh              # Three-step wrapper: preprocess + calibrate + SWEB run
│
├── core/                                  # Reusable model and data-processing modules
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

Runtime outputs are written under `1_ssebop_inputs/`, `1_era5land_raw/`, `1_era5land_stacks/`, `2_ssebop_outputs/`, `3_sweb_inputs/`, and `4_sweb_outputs/`.

## Workflow overview
1. `workflows/1_ssebop_prepare_inputs.py`: build a run-specific GEE config and download Landsat scenes.
2. `workflows/1b_download_era5land_daily.py`: download ERA5-Land DAILY_AGGR daily rasters.
3. `workflows/1c_stack_era5land_daily.py`: stack the ERA5-Land rasters into daily `precipitation`, `tmax`, `tmin`, `rs`, `ea`, and `et_short_crop` NetCDFs.
4. `workflows/2_ssebop_run_model.py`: compute daily `ET`, `E`, `T`, `etf_interp`, `ndvi_interp`, and `Tc` from Landsat plus explicit ERA5-Land meteorology files.
5. `workflows/3_sweb_preprocess_inputs.py`: align ERA5-Land precipitation, SSEBop `E/T/ET`, soil properties, and optional SMAP SSM to one grid.
6. `workflows/4_sweb_calib_domain.py`: calibrate domain-wide SWEB parameters (`diff_factor`, `sm_max_factor`, `sm_min_factor`, `root_beta`).
7. `workflows/5_sweb_run_model.py`: run spatial SWEB and export root-zone soil moisture NetCDF outputs.

## Quick start
The shell wrappers in `workflows/` are the easiest way to run end-to-end workflows:

```bash
cd workflows

# Step A: Landsat + ERA5-Land + SSEBop workflow
bash ssebop_runner_landsat.sh <run_subdir>

# Step B: SWEB workflow (preprocess + calibrate + final run)
bash sweb_domain_runner.sh <run_subdir>

# Step C: Plot extracted time series from SSEBop + SWEB outputs
python ../visualisation/plot_time_series.py \
  --run-subdir <run_subdir> \
  --output /g/data/ym05/sweb_model/figures/<run_subdir>_timeseries.png

# Step D: Plot SWEB layer heatmap (optional SSEBop top panel)
python ../visualisation/plot_heatmap.py \
  --run-subdir <run_subdir> \
  --lat <latitude> --lon <longitude> \
  --output /g/data/ym05/sweb_model/figures/<run_subdir>_heatmap.png

# Optional: domain-wide heatmap instead of point mode
python ../visualisation/plot_heatmap.py \
  --run-subdir <run_subdir> \
  --domain-mean \
  --output /g/data/ym05/sweb_model/figures/<run_subdir>_heatmap_domain.png

# Step E: Run notebook examples
cd ../notebooks
jupyter notebook
```

Both wrapper scripts currently include environment-specific default paths (for example `/g/data/...`) near the top of each script. Update those values before running on another machine or filesystem.

The meteorology path is now ERA5-Land-based and globally usable. Soil texture/SOC rasters and SMAP/reference inputs still use the current downstream defaults and are not yet globalized.

## Key outputs
- From ERA5-Land download/stack steps (`1b_download_era5land_daily.py`, `1c_stack_era5land_daily.py`): daily GeoTIFF downloads plus `precipitation_daily_<start>_<end>.nc`, `tmax_daily_<start>_<end>.nc`, `tmin_daily_<start>_<end>.nc`, `rs_daily_<start>_<end>.nc`, `ea_daily_<start>_<end>.nc`, and `et_short_crop_daily_<start>_<end>.nc`.
- From SSEBop run (`2_ssebop_run_model.py`): `et_daily_ssebop_<start>_<end>.nc` plus intermediate `etf`/`ndvi` products, all driven by explicit ERA5-Land meteorology files.
- From SWEB preprocess (`3_sweb_preprocess_inputs.py`): `rain_daily_*.nc`, `effective_precip_daily_*.nc`, `et_daily_*.nc`, `t_daily_*.nc`, `soil_*.nc`, and optionally `smap_ssm_daily_*.nc`.
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

- `workflows/` for runnable pipeline entry points and orchestration scripts.
- `core/` for reusable model logic, numerical helpers, and processing utilities.

## Development status
This repository is actively evolving. Verify file paths, date ranges, and spatial settings before running large jobs.
