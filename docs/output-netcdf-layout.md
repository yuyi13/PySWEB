# SWB output storage

Time-dependent gridded SWB output variables use compressed NetCDF chunks of at most 32 days × 64 latitude cells × 64 longitude cells. Chunk lengths are capped by the actual output dimensions, including resampled grids. Final-state variables remain spatial fields.

This storage layout limits repeated decompression during daily/monthly quality checks, point extraction and maps on long records. It changes file layout and binary file hashes, while retaining the existing output precision, variables, dates and numerical results. It does not change the solver or remove the runner’s in-memory output arrays; simulation allocations must still cover those arrays.
