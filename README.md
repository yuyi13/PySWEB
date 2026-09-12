<p align="center"><img src="docs/assets/pysweb-logo.png" alt="PySWEB — Soil Water-Energy Balance" width="700"></p>

# PySWEB · Soil Water-Energy Balance

[![Tests](https://github.com/yuyi13/PySWEB/actions/workflows/tests.yml/badge.svg)](https://github.com/yuyi13/PySWEB/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB)](https://www.python.org/)
[![MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

PySWEB estimates daily, layered soil moisture from precipitation, evapotranspiration and soil hydraulic properties. It links satellite-based SSEBop evapotranspiration with a one-dimensional soil water balance, including root uptake, vertical redistribution and bottom drainage. It continues development of the Sydney Soil Water-Energy Balance model.

**Research software under active development.** The offline example is runnable immediately. Regional applications require external data, appropriate calibration and independent validation. The energy connection enters through the evapotranspiration estimate; the soil solver does not integrate a prognostic soil heat balance.

[Quick start](#quick-start) · [Workflow](#configured-workflow) · [Your soil data](#using-your-own-soil-data) · [Methods](docs/methods.md) · [Migration](docs/migration.md)

## Quick start

Python 3.12 or newer is required. From the repository root:

```bash
git clone https://github.com/yuyi13/PySWEB.git
cd PySWEB
python -m pip install -e ".[dev]"
pysweb demo
```

The demo generates a small synthetic two-layer soil dataset and a full month of forcing, preprocesses a 14-day window, runs four cells and checks water-budget closure. It needs no Earth Engine account or external downloads. Its output is:

```text
examples/outputs/demo/
├── source/       # Synthetic forcing and custom hydraulic properties
├── prepared/     # Aligned daily forcing, soil files and provenance
└── results/      # SWEB_RZSM_2024-01-01_2024-01-14.nc and manifest
```

The same runnable example is available through Python:

```python
from pysweb.demo import run_demo
output = run_demo("examples/outputs/demo")
```

For runtime dependencies only, install `python -m pip install .`. Plotting is available with `python -m pip install ".[plot]"`. Installation currently uses the Git repository; no PyPI publication is claimed.

## Configured workflow

[examples/workflow.toml](examples/workflow.toml) centralizes paths, dates, extent, soil source, calibration and model settings. Copy it to a study configuration and edit the values. Relative paths resolve from the configuration file's directory. The example extent is illustrative.

```bash
pysweb workflow --config examples/workflow.toml --dry-run
# After setting a real study extent and Earth Engine project:
pysweb workflow --config study.toml
```

The five stages share one implementation in `pysweb.workflow`:

| Stage | Role | Inputs / requirements |
|---|---|---|
| `prepare` | Download Landsat, ERA5-Land and NASADEM; stack meteorology | Earth Engine authentication and your project |
| `ssebop` | Estimate daily ET and partition E/T | Prepared satellite, meteorology and DEM files |
| `preprocess` | Align forcing and soil properties; calculate effective rain | Full calendar-month rain; local or online soil inputs |
| `calibrate` | Fit four domain parameters to reference surface soil moisture | Enabled explicitly; local reference or configured Earth Engine asset |
| `run` | Simulate independent soil columns | Prepared forcing and hydraulic properties |

Select stages with `--stages preprocess,calibrate,run`. Calibration is skipped when `calibration.enabled = false`. Its date window can differ from the simulation window. Both windows are prepared when required; calibration and simulation share process settings and state timing. See [the workflow guide](docs/workflows.md) for offline operation, HPC execution and individual APIs.

Every configured run writes `resolved-config.json`. Model output manifests record input SHA-256 checksums, configuration, code identity and output checksums. `skip_existing` validates the model output against this identity before reuse. Unmanifested older outputs require regeneration. Generated data and environments stay outside version control.

## Using your own soil data

**Local soil data are a maintained pathway.** `spec/` provides compatibility entrypoints and guidance for the MLConstraints and custom-soil workflows; their implementations live in `pysweb.soil` and use the shared SWB solver.

| Soil source | Status | Contract |
|---|---|---|
| `openlandmap` | Implemented; online | Versioned global clay, sand and organic-carbon assets; five model layers |
| `custom` | Implemented; offline | NetCDF hydraulic properties with units, CRS and explicit layer depths |
| `mlcons` | Implemented; local data required | Existing MLConstraints Clay/Sand/OC rasters, or RDS extraction using R |
| `slga` | Reserved | Fails explicitly until a backend is implemented |

A custom hydraulic file contains `porosity`, `wilting_point`, `available_water_capacity`, `b_coefficient` and `conductivity_sat` on a common layered grid. Supply your actual layer bottoms in millimetres. The model validates units, geometry, CRS and hydraulic ranges before use. See [the soil input contract](docs/soil-inputs.md), [spec/README.md](spec/README.md) and [examples/custom-soil.toml](examples/custom-soil.toml).

Measured hydraulic parameters can avoid uncertainty from global texture maps and pedotransfer functions. Their benefit still depends on measurement quality, depth support and spatial representativeness.

## Outputs and scientific interpretation

The SWB NetCDF contains individual-layer soil moisture (`rzsm_layer_*`, m³ m⁻³), total profile water storage (`profile_sm`, mm), forcing summaries, daily budget terms, valid-forcing fractions and the final state of each layer. The default `state_timing = "start"` preserves historical timestamps. Select `"end"` to report each day's updated state. The final state is retained in both modes.

Water-budget diagnostics distinguish infiltration, actual soil-limited ET, drainage, saturation overflow, numerical storage adjustments and residuals. Missing forcing freezes a cell's state and marks that day invalid; these stored states must be screened with the QC variable. Calibration excludes cells with incomplete forcing and compares model and observation means over the same support, weighted by geographic cell area. A failed solver candidate receives an infinite objective.

Effective rainfall follows the existing monthly Smith formula, evaluated using complete calendar months before slicing the requested window. Partial-month source data now fail explicitly. These changes can alter historical outputs; [migration notes](docs/migration.md) identify the intentional differences.

This update verifies numerical and software behavior with synthetic tests. It does not establish regional predictive skill. Retained empirical assumptions, reference-data limitations, initialization and validation priorities are documented in [methods and limitations](docs/methods.md).

## Repository map

| Path | Purpose |
|---|---|
| [pysweb/](pysweb/) | Canonical package: `ssebop`, `swb`, `soil`, `met`, `dem`, `io`, `visualisation` |
| [workflows/](workflows/) | Thin numbered CLIs and portable shell launchers |
| [spec/](spec/) | Maintained custom-soil / MLConstraints compatibility entrypoints |
| [examples/](examples/) | Small configurations; ignored generated demo results |
| [notebooks/](notebooks/README.md) | Shared-config run notebook and diagnostic plotting examples |
| [tests/](tests/) | Unit, scientific-contract and integration tests |
| [docs/](docs/) | Methods, data contracts, workflow and migration guidance |
| [.github/workflows/](.github/workflows/) | Linux/macOS tests, wheel build and installed-package demo |

`pysweb/` is the sole runtime package. The former top-level `core/` retirement is complete; no new code should depend on that namespace. Obsolete `superpowers` implementation plans have been replaced by maintained documentation, with historical versions retained in Git.

Notebook entrypoints are [01_run_pysweb.ipynb](notebooks/01_run_pysweb.ipynb), [02_plot_heatmap.ipynb](notebooks/02_plot_heatmap.ipynb) and [03_plot_time_series.ipynb](notebooks/03_plot_time_series.ipynb). Plotting is owned by `pysweb.visualisation`; `workflows/6_plot_results.py` remains available.

## Data and citation

The full workflow uses external datasets with their own access and reuse terms: [Landsat Collection 2](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2), [ERA5-Land daily aggregates](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR), [NASADEM](https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001) and [OpenLandMap](https://openlandmap.org/). The default reference SSM asset is user-hosted and may require access permission. [Data provenance](docs/data.md) lists versions, units and acquisition constraints.

Cite the software version and commit used in an analysis; GitHub can export the [CITATION.cff](CITATION.cff) metadata. No paper DOI or permanent software archive is currently assigned in this repository.

```text
Yu, Y. (2026). PySWEB: Soil Water-Energy Balance (version 0.1.0).
https://github.com/yuyi13/PySWEB [include the commit used].
```

The code is distributed under the [MIT licence](LICENSE). Dataset licences remain those of their providers. For development checks and contribution guidance, see [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).
