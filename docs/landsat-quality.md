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
