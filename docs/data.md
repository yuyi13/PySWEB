# Data scope and provenance

The repository redistributes generated synthetic demo inputs only. Full applications depend on external data and their licences.

| Input | Source / version | Use and limitations |
|---|---|---|
| Landsat | [Collection 2 Level 2](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2), LC08/LC09 | Surface temperature and reflectance; QA masking and scaling in `pysweb.ssebop.landsat` |
| Meteorology | [ECMWF/ERA5_LAND/DAILY_AGGR](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR) | Daily forcing and reference ET; package conversion to model units |
| Elevation | [NASA/NASADEM_HGT/001](https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001) | Elevation correction and meteorological calculations; coverage limits apply |
| Global soil | [OpenLandMap](https://openlandmap.org/), clay/sand/SOC `v02` | Predictor assets listed below; hydraulic estimates require pedotransfer assumptions |
| Reference SSM | `users/qianrswaterr/GlobalSSM1km0509` | Current default user-hosted GEE asset; access, coverage and scientific provenance must be checked for the study |
| Local soil/reference | User-supplied NetCDF, GeoTIFF or MLConstraints RDS | Record measurement methods, dates, depths, CRS, units and reuse permissions |

OpenLandMap asset identifiers:

```text
OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02
OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02
OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02
```

ERA5-Land daily flow variables are provided as daily sums; state variables use daily aggregates. The provider documents occasional negative accumulated values from packing artifacts. PySWEB retains its existing precipitation conversion and non-negative handling. Review unusual events against the source data. Source spatial resolution and uncertainty must be reported separately from the chosen output grid.

Prepared precipitation includes full calendar months, even for a shorter requested run. Raw input files remain external; preprocessing and output manifests record checksums of local files and identifiers of remote assets. Remote asset identifiers alone cannot freeze an upstream asset that changes in place: retain the downloaded/prepared inputs with your study archive.

No DOI is asserted for the software or its default user-hosted reference asset. A future release should archive a versioned package, configuration, provenance manifests and legally redistributable validation fixtures under a permanent identifier.
