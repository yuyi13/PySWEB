# SWB output storage

Time-dependent gridded SWB output variables use compressed NetCDF chunks of at most 32 days × 64 latitude cells × 64 longitude cells. Chunk lengths are capped by the actual output dimensions, including resampled grids. Final-state variables remain spatial fields.

This storage layout limits repeated decompression during daily/monthly quality checks, point extraction and maps on long records. It changes file layout and binary file hashes, while retaining the existing output precision, variables, dates and numerical results. It does not change the solver or remove the runner’s in-memory output arrays; simulation allocations must still cover those arrays.

## Configured native workflow

Two final products share exactly the same `x`/`y` values, UTM CRS and 30 m affine transform:

| Product | Main variables | Included auxiliaries |
|---|---|---|
| `products/SSEBop_ET.nc` | `ET`, `E_s`, `E_t`, `etf_interp`, `ndvi_interp`, `f_t` on `time,y,x` | Daily reference ET, tmax/tmin, solar radiation, vapour pressure; elevation; development dT climatology; scene-time LST, cold temperature, dT, observed NDVI/ETf, reflectance, QA and validity; optional water mask |
| `products/SWEB_RZSM.nc` | `rzsm_default` and `rzsm_calibrated` on `time,layer,y,x`, in m³ m⁻³ | Raw/effective precipitation, ET/transpiration/soil-evaporation forcing, NDVI, all five hydraulic fields, layer bottoms, scenario-specific profile storage, daily budgets, valid-forcing flags and final states |

Scene retrieval auxiliaries use `scene_time`; dT climatology uses `dayofyear`. These temporal dimensions differ from the daily model time axis without changing the spatial grid. Scenario-specific variables receive `_default`/`_calibrated` suffixes. Calibration-disabled runs have no calibrated variable. Four scalar parameters for each scenario and the configuration are stored as attributes; manifests retain input identity and software provenance.

Intermediate single-scenario SWB files retain the component API's `rzsm_layer_*` variables under `products/intermediate/`. They support diagnosis and are not additional consolidated final products. Native calibration inputs remain separate under `calibration_inputs/`; they intentionally have a different spatial grid. See [the full native-grid contract](native-grids.md).

## Streaming large records

Streamed daily and scene-dependent arrays use temporal chunks of at most
32 entries and spatial chunks of at most 64 pixels. This matches the bounded
copy operations used for final crops, scenario combination and initialization
trimming. Seasonal controls additionally align their y chunks with the row
blocks used to construct the control.

Older source files with large temporal chunks remain readable. Streaming copies
temporarily increase the current variable's read cache to retain chunk slabs,
capped at 512 MiB per variable, then restore its original cache settings.
Daily preprocessing uses the same bounded cache on the NetCDF handle read by
xarray, while retaining xarray's CF decoding and the existing daily reprojection.
Only the selected component's cache changes; global NetCDF defaults stay intact.
These storage choices preserve values, masks, coordinates, units and scientific
provenance; they do not change model or forcing parameters.
