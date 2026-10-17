# Bounded-memory time indexing

`pysweb.io.time_reindex.reindex_netcdf_time` copies a flat numeric NetCDF dataset onto explicitly supplied output dates and source-row indices. It supports repeated and reordered source rows and uses index `-1` for missing rows. It performs no interpolation and makes no scientific choice about warm-up years or calibration dates.

The helper reads one variable in temporal blocks (32 rows by default), preserves numeric storage types, raw packed values, spatial coordinates and variable attributes, and writes compressed NetCDF through the existing netCDF4 dependency. Dask is not required. Output dates must be unique and increasing; time must be the first dimension of every time-dependent variable. Output time uses the proleptic Gregorian calendar in days since 1970-01-01.

```python
from pysweb.io.time_reindex import reindex_netcdf_time

reindex_netcdf_time(
    source_path, destination_path, output_dates, source_indices,
    chunk_days=32,
    attributes={"initialisation_source": "Explicitly declared development-year cycle"},
)
```

A successful write atomically replaces the destination. A failure preserves the previous destination and leaves a uniquely named partial file for diagnosis. Nested groups and nonnumeric variables are rejected. Memory is bounded by the temporal block and each variable's spatial extent; the helper does not provide a bound for arbitrarily large spatial grids.
