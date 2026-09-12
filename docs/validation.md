# Validation of the September 2026 update

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
