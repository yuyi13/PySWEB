# Landsat quality and pilot reuse

Default Landsat preparation downloads Collection 2 `ST_B10`, `SR_B4`, `SR_B5`,
`QA_PIXEL` and `QA_RADSAT`. It applies the pixel mask before export and retains
QA for local verification. SSEBop requires QA by default and checks it again
before estimating NDVI and ET fraction.

`QA_PIXEL` bits 0–5 and 7 exclude fill, dilated cloud, high-confidence cirrus,
cloud, shadow, snow and water. `QA_RADSAT` must be zero. There is no extra
cloud-distance buffer, scene cloud-cover cutoff or medium-confidence rejection.
Raw thermal, red and NIR values must be positive. Reflectance uses
`DN * 0.0000275 - 0.2`, temperature uses `DN * 0.00341802 + 149`, and NDVI
must lie in [-1, 1]. The reusable `pysweb.ssebop.quality.landsat_c2_qa_mask`
can retain water for other diagnostics through `exclude_water=False`.
The [official Landsat catalogue](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2)
defines the bits and scales. Temporal filling remains separate from masking.

## Download and local-input contracts

- Use `daily_strategy: first` when retaining QA; bitfields cannot be median-composited.
  Multiple same-day scenes need an explicit scene or mosaic design.
- Set `postprocess.maskval_to_na: false`: zero means no saturation in QA_RADSAT.
  The reader preserves raw QA even in older GeoTIFFs declaring nodata=0;
  missing radiometric inputs still exclude those pixels.
- Missing requested bands and missing enabled mask bands fail explicitly.
- Independently masked files without QA require `landsat_qa = "pre_masked"`
  in TOML, the equivalent API argument or `--landsat-qa pre_masked`.
  This declares externally established QA; it does not perform or verify
  cloud masking. Outputs record the mode.
- Custom download templates must retain both QA bands, use `first`, and
  preserve QA zeros to satisfy default scene processing.

## Package use in pilots

Use `pysweb workflow` or its preparation/SSEBop/soil/SWB APIs when their input
contract matches the experiment. The AU-Ync pilot already uses package
ERA5-Land stacking/reference ET, OpenLandMap pedotransfer, SSEBop physics,
temporal reconstruction, preprocessing, calibration and the reference solver.
Exact WRS/scene selection, padded projected grids, development-only climatology,
held-out reconstruction boundaries and extra retrieval-quality filters remain
explicit study orchestration. The generic daily downloader does not yet
reproduce that complete exact-scene experiment.

Use shared QA and scale helpers in study adapters. Preserve code and input
identities when moving operations into the package. Replaying QA against
retained scenes verifies mask equivalence; it is not a new calibration or
an observed-accuracy claim.

The sensor argument to `landsat_c2_qa_mask` now supports `LT05`, `LE07`,
`LC08` (default) and `LC09`. `landsat_c2_l2_bands` returns the corresponding
thermal/red/NIR mapping. Landsat 5/7 uses ST_B6, SR_B3 and SR_B4; QA_PIXEL
bit 2 is unused. Requiring QA_RADSAT zero excludes Landsat 7 dropped pixels.
The scene API recognises ST_B6 as the older-sensor QA convention. These rules
retain missing SLC/thermal support and do not perform cross-sensor spectral
harmonisation. See the official [Landsat 5 catalogue](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2)
and [Landsat 7 catalogue](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2).

`pysweb.io.ee_export.download_geotiff` downloads one explicit image using
an explicit region/CRS and scale or affine grid, preserving supplied band
identities and raw QA zero. It returns grid, source and checksum metadata
and atomically replaces the target only after raster validation. It does not
select a daily composite, persist signed URLs or silently retry authentication.
Use bounded requests within Earth Engine's direct-download limits; callers
remain responsible for immutable scene selection, tiling, cached provenance
and inspecting cross-sensor support.


## Mixed sensors and reference ET

OpenET SSEBop supports Landsat 5, 7, 8 and 9 Collection 2 Level 2.
Its FANO coefficients are shared across sensors. PySWEB uses the same 0.125,
0.9 and 0.4 defaults, with a local multiscale cold-boundary implementation.
The local method has different land-cover conditioning, grid and smoothing
behaviour from the full OpenET workflow. Band support and kernel agreement
establish implementation checks; field validation remains necessary.
Use `lst_band="auto"` to select ST_B6/SR_B3/SR_B4 or ST_B10/SR_B4/SR_B5
from each raw C2 raster. Both thermal bands in one raster are ambiguous and rejected.
Positive DN and QA masks apply before retrieval; Landsat 7 scan gaps remain missing.

Native SSEBop ET fraction is **alfalfa referenced**. The workflow now requires
matching reference ET. ERA5-Land stacking writes both `et_short_crop` (grass)
and `et_tall_crop` (alfalfa). For example, pass
`reference_et=".../et_tall_crop_daily_START_END.nc"`,
`reference_et_var="et_tall_crop"`, `et_reference_type="alfalfa"` in the
Python API or YAML run configuration. Historical short-reference multiplication
requires `allow_legacy_reference_mismatch=True`; it warns and labels the ET
output. It is retained for reproducibility of earlier results.
A grass conversion must use a justified ETr/ETo ratio; no fixed conversion is assumed.

Sources: [OpenET model](https://github.com/Open-ET/openet-ssebop/blob/main/openet/ssebop/model.py),
[OpenET image processing](https://github.com/Open-ET/openet-ssebop/blob/main/openet/ssebop/image.py).
