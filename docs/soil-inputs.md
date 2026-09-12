# Local soil inputs

## Custom hydraulic properties

Set `soil.source="custom"`. Provide exactly one `soil.file` (a combined NetCDF) or `soil.input_dir` (five separate NetCDFs). CLI equivalents are `--soil-source custom`, `--soil-file` and `--soil-input-dir`.

| Variable | Separate filename | Required units |
|---|---|---|
| `porosity` | `soil_porosity.nc` | `m3 m-3` |
| `wilting_point` | `soil_wilting_point.nc` | `m3 m-3` |
| `available_water_capacity` | `soil_available_water_capacity.nc` | `m3 m-3` |
| `b_coefficient` | `soil_b_coefficient.nc` | `dimensionless` or `1` |
| `conductivity_sat` | `soil_conductivity_sat.nc` | `mm day-1` |

Each variable requires a `layer` dimension, spatial coordinates and a CRS readable by rioxarray. Write CRS with `da.rio.write_crs(...)` before exporting. Use `layer_bottoms_mm` variable metadata or `soil.layer_bottoms_mm` in TOML (`--soil-layer-bottoms-mm` on the preprocess CLI). Depths are positive, strictly increasing cumulative bottoms in millimetres. All variables must share the same layers. The demonstration constructs a complete valid file in `pysweb.demo.prepare_demo`.

The adapter retains exact matching grids and otherwise reprojects continuous hydraulic properties bilinearly to the target grid. Values must satisfy `0 <= wilting_point < porosity <= 1`, `0 <= available_water_capacity <= porosity-wilting_point`, `b > 0` and `Ksat >= 0`. Missing soil values exclude the affected profile from simulation. Inputs with missing units, CRS or depth information fail explicitly. Preserve source measurement metadata and document how point or horizon observations became spatial layers; reprojection alone does not solve this representativeness problem.

## MLConstraints adapter

Set `soil.source="mlcons"` and `soil.mlcons_dir` to a local directory containing:

```text
LegendModel_2.0.1_Clay_4layers.tif
LegendModel_2.0.1_Sand_4layers.tif
LegendModel_2.0.1_OC_4layers.tif
```

These are multiband rasters with one depth per band, shallow to deep. Default bottoms are 150, 300, 600 and 1000 mm; override them for your actual data. Clay, sand and organic carbon use percentage inputs. The adapter preserves the historical pedotransfer implementation and records that convention. Its organic-matter conversion is `OM = 1.72 * OC * 0.01` for the existing coefficients. This convention requires a separately benchmarked scientific review before it is changed.

If only `LegendModel_2.0.1_output.RDS` is present, the adapter can extract rasters using `Rscript` with both `raster` and `terra`. It writes extracted products under the output directory's `mlcons_rasters/` directory, preserving the source directory. R is needed only for this conversion; the Python package does not install R. Direct GeoTIFF inputs avoid that dependency.

The MLConstraints adapter produces the same `SoilOutputs` contract as OpenLandMap and custom hydraulic inputs. Precipitation, reference observations, calibration and simulation all use the canonical package workflows. See [spec/README.md](../spec/README.md) for compatibility entrypoints.

## Reference surface moisture

A local reference NetCDF uses `soil.reference_file` / `soil.reference_var` in a configured workflow, or `reference_file` / `reference_var` in preprocessing. Supply daily volumetric moisture in `m3 m-3` with a CRS and timestamps covering the calibration interval. Select a surface depth and state timing appropriate to the observation. Calibration fits a domain-mean objective and does not establish spatial or depth-specific predictive accuracy.
