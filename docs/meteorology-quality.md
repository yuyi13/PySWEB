# Precipitation quality and packing noise

ERA5-Land accumulated fields can contain small negative values caused by GRIB
packing. The [Earth Engine catalogue](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR)
and [ECMWF explanation](https://confluence.ecmwf.int/spaces/UDOC/pages/208501579/Why+are+there+sometimes+small+negative+precipitation+accumulations+-+ecCodes+GRIB+FAQ)
describe the limitation. The appropriate tolerance depends on the source and its
encoding; PySWEB does not assume a universal value or alter stacking defaults.

For an explicitly justified tolerance, apply:

```python
from pysweb.met.era5land.quality import qc_precipitation_mm

corrected, flags = qc_precipitation_mm(
    precipitation_mm,
    small_negative_tolerance_mm=0.001,  # Example study choice, not a product default.
)
```

The helper preserves nonnegative values, converts only negative values within
the tolerance to zero, and masks more negative or nonfinite values. It does not
modify the input array. Flags have the same shape as the data:

| Flag | Meaning |
|---|---|
| 0 | Finite, nonnegative input unchanged |
| 1 | Small negative value reset to zero |
| 2 | Negative value beyond tolerance masked |
| 3 | Nonfinite input masked |

Use millimetres, record the tolerance and retain the flags. Audit correction
depths over the intended analysis period before model use. This bounded
correction does not resolve small positive packing artefacts, establish exact
rain-free days or independently validate unusually high rainfall. Missing data
are never converted to zero. Source files remain unchanged.
