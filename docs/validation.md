# Validation

## Native grids and paired products, 17 October 2026

The revision was exercised on Gadi with the existing Python 3.12.12 geospatial runtime, under explicit PBS CPU, memory and storage allocations.

- **355 tests passed.** New checks cover native CRS/affine preservation, integer-pixel export windows, mixed-band rejection, complete-footprint aggregation, cell–day errors that cannot cancel spatially, temporal holdout isolation and exact final SEB/SWB grid equality.
- The offline integration prepares native GSSM and 30 m UTM grids, fits two domain scalars, runs both scenarios and writes both consolidated products with soil, forcing and budget auxiliaries. It checks a synthetic European UTM domain; the fixture's zero-iteration candidate tests orchestration, not optimizer convergence or predictive skill.
- The source distribution and wheel built successfully using the installed `uv` and setuptools. The wheel was installed into an isolated job-local target, imported outside the checkout, and completed the two-worker offline demo with a maximum daily budget residual of **2.58 × 10⁻¹⁴ mm**. The shared runtime was unchanged.
- Bounded live Earth Engine downloads passed native-grid verification for Landsat, ERA5-Land, NASADEM, OpenLandMap clay/sand/organic carbon and GSSM. The live OpenLandMap transform used 0.002083333° spacing, illustrating why source metadata must be read rather than inferred from a nominal metre scale.
- Native NASADEM export was tested against its global source origin; explicit pixel dimensions avoid Earth Engine constructing an excessively wide crop. DEM preparation also requires complete native ERA5 pixel footprints.

The live test checks acquisition and georegistration, not a complete regional calibration. A full native-grid Yanco/Narrabri rerun and independent field validation remain separate work. Native-grid calibration can reduce solver cost; these software checks do not establish improved accuracy, exact coarse-to-fine equivalence or a measured GPU speedup. NetCDF dependency, alias-deprecation and TIFF band warnings were recorded; successful numerical and file checks are the evidence reported here.

## September 2026 update

The modernization was checked locally with Python 3.12.12:

- 231 pytest tests passed, including the existing numerical helpers and new solver, local soil, calendar, calibration-support, download-integrity and workflow tests.
- Source distribution and wheel built successfully. The wheel was installed and imported from `site-packages` outside the repository; the two-worker offline demo completed with a maximum daily water-budget residual of 2.58 × 10⁻¹⁴ mm.
- The demo exercised real NetCDF writing/reading, two-layer custom soil preparation, partial-window output using full-month rainfall, and final-state/QC/budget export.
- Wet, dry and saturation tests covered one-, two- and five-layer profiles. Restart and state-timing tests checked the complete direct-solver sequence.
- Real synthetic GeoTIFF tests exercised MLConstraints inputs and download-cache integrity. Invalid custom units, depths, CRS and hydraulic values were rejected.
- Shell launchers ran from an unrelated working directory; notebook Python cells compiled and the offline notebook client executed.
- Local documentation links, retired-namespace imports, undefined names and Git whitespace checks passed. A time-series diagnostic was rendered and visually inspected.

The local test environment emitted one NumPy binary-compatibility warning when loading its existing NetCDF dependency. NetCDF round-trip and installed-wheel demo checks completed successfully. CI performs the tests and wheel/demo checks on fresh Linux/macOS environments.

Live Earth Engine acquisition, a regional SSEBop-to-SWB run, MLConstraints RDS extraction through R, and independent observational validation were not performed as part of this verification. The download transport is mocked in integrity tests; raster and output validation use real local files. No predictive-skill claim follows from these software checks.
