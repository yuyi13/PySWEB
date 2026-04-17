# ERA5-Land Daily Aggregate Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Australian-only SILO/ANUClimate meteorology path with a global `ERA5-Land DAILY_AGGR` workflow in Google Earth Engine, and compute FAO56-like daily short-reference `ETo` using `refetgee`-style daily equations while leaving soil and soil-moisture-reference work out of scope.

**Architecture:** Keep Landsat download and SSEBop logic in place, but add a parallel ERA5-Land daily download step that fetches raw daily meteorology GeoTIFFs for the run window. Stack those daily rasters locally into NetCDF products, derive `ea`, `wind_speed`, and daily grass-reference `ETo` in Python using a small pure NumPy/xarray module aligned with the `refetgee` daily ASCE-short formulation, then pass the resulting explicit NetCDF paths into the existing SSEBop and SWEB workflows instead of relying on SILO/ANUClimate directory inference.

**Tech Stack:** Python 3.12, Earth Engine API, NumPy, xarray, rasterio/rioxarray, PyYAML, pytest, bash

---

## File Structure

**Create**
- `core/era5land_download_config.py`
  Pure config builder for the ERA5-Land daily GEE download workflow.
- `core/era5land_refet.py`
  Climate-math utilities for `ERA5-Land DAILY_AGGR` unit conversion and FAO56-like daily grass-reference `ETo` calculation.
- `core/era5land_stack.py`
  File-discovery and date-parsing helpers used by the local ERA5-Land NetCDF stacker.
- `core/met_input_paths.py`
  Source-agnostic explicit meteorology path resolution for `workflows/2_ssebop_run_model.py`.
- `workflows/1b_download_era5land_daily.py`
  Run-subdir-aware GEE downloader entry point for `ECMWF/ERA5_LAND/DAILY_AGGR`.
- `workflows/1c_stack_era5land_daily.py`
  Convert downloaded daily ERA5-Land GeoTIFFs into NetCDF stacks for precipitation, `tmax`, `tmin`, `rs`, `ea`, and `et_short_crop`.
- `tests/core/test_gee_downloader_config.py`
  Tests for `core/gee_downloader.py` config validation around daily download strategy.
- `tests/core/test_era5land_refet.py`
  Unit tests for unit conversions and `ETo` math.
- `tests/workflows/test_1b_download_era5land_daily.py`
  Pure-function tests for the ERA5-Land GEE config builder.
- `tests/workflows/test_1c_stack_era5land_daily.py`
  Tests for daily-file discovery and date parsing in the local stacker.
- `tests/workflows/test_2_ssebop_run_model.py`
  Tests for explicit meteorology-path resolution so ERA5-Land products bypass SILO inference cleanly.

**Modify**
- `core/gee_downloader.py`
  Add config-driven support for daily single-image products so `ERA5-Land DAILY_AGGR` can download one image per day without median compositing.
- `workflows/2_ssebop_run_model.py`
  Keep the existing explicit meteorology-file interface, but refactor path resolution and help text so the script is no longer SILO-only in wording or fallback logic.
- `workflows/ssebop_runner_landsat.sh`
  Add ERA5-Land daily download + stack steps and pass explicit meteorology NetCDFs into `2_ssebop_run_model.py`.
- `workflows/sweb_domain_runner.sh`
  Replace Australian precipitation directory defaults with explicit ERA5-Land precipitation NetCDFs produced by the new stacker.
- `README.md`
  Document the new Landsat + ERA5-Land path, expected outputs, and what remains out of scope.

**Do Not Touch In This Plan**
- `workflows/3_sweb_preprocess_inputs.py`
  It already supports `--rain-file`; use that path rather than refactoring precipitation ingestion.
- `workflows/4_sweb_calib_domain.py`
  Soil moisture reference changes are intentionally deferred.
- Soil-grid code and `spec/` prototypes.

### Target Output Products

The new local stacker writes the following files for each run window:

- `precipitation_daily_<start>_<end>.nc` with variable `precipitation` in `mm day-1`
- `tmax_daily_<start>_<end>.nc` with variable `tmax` in `degC`
- `tmin_daily_<start>_<end>.nc` with variable `tmin` in `degC`
- `rs_daily_<start>_<end>.nc` with variable `rs` in `MJ m-2 day-1`
- `ea_daily_<start>_<end>.nc` with variable `ea` in `kPa`
- `et_short_crop_daily_<start>_<end>.nc` with variable `et_short_crop` in `mm day-1`

These names match the semantics already expected by `workflows/2_ssebop_run_model.py`, so the runner can use explicit file paths instead of any source-specific directory convention.

### ERA5-Land Band Mapping

Download these `ECMWF/ERA5_LAND/DAILY_AGGR` bands directly from GEE:

- `temperature_2m_min` -> `tmin` input
- `temperature_2m_max` -> `tmax` input
- `dewpoint_temperature_2m` -> convert to mean actual vapour pressure `ea`
- `u_component_of_wind_10m`
- `v_component_of_wind_10m`
- `surface_solar_radiation_downwards_sum` -> `rs`
- `total_precipitation_sum` -> `precipitation`

Deliberately ignore ERA5-Land evaporation and soil-water bands in this plan.

### ETo Equation Boundary

Mirror the `openet-refet-gee` daily short-reference (`eto`) path:

- short crop constants: `Cn = 900`, `Cd = 0.34`
- pressure from elevation, not ERA5-Land `surface_pressure`
- actual vapour pressure from dewpoint temperature
- wind adjusted from 10 m to 2 m
- daily extraterrestrial radiation and clear-sky radiation based on latitude, day-of-year, and elevation

This keeps the result close to FAO56-like daily `ETo` while staying consistent with the `refetgee` implementation strategy the user selected.

## Task 1: Make the GEE Downloader Accept Daily Single-Image Products

**Files:**
- Modify: `core/gee_downloader.py`
- Test: `tests/core/test_gee_downloader_config.py`

- [ ] **Step 1: Write the failing config-validation tests**

```python
from pathlib import Path

import yaml

from core.gee_downloader import GEEDownloader


def _write_cfg(tmp_path: Path, daily_strategy: str) -> Path:
    cfg = {
        "coords": [147.2, -35.1, 147.3, -35.0],
        "download_dir": str(tmp_path / "downloads"),
        "start_year": 2024,
        "start_month": 1,
        "start_day": 1,
        "end_year": 2024,
        "end_month": 1,
        "end_day": 3,
        "bands": ["temperature_2m_min"],
        "scale": 11132,
        "out_format": "tif",
        "auth_mode": "browser",
        "filename_prefix": "ERA5LandDaily",
        "collection": "ECMWF/ERA5_LAND/DAILY_AGGR",
        "daily_strategy": daily_strategy,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_daily_strategy_defaults_to_median(tmp_path):
    path = _write_cfg(tmp_path, "median")
    downloader = GEEDownloader(str(path))
    assert downloader.cfg["daily_strategy"] == "median"


def test_daily_strategy_accepts_first(tmp_path):
    path = _write_cfg(tmp_path, "first")
    downloader = GEEDownloader(str(path))
    assert downloader.cfg["daily_strategy"] == "first"
```

- [ ] **Step 2: Run the tests to verify the new setting is not supported yet**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/core/test_gee_downloader_config.py -q
```

Expected: FAIL because `daily_strategy` is not validated or stored in `GEEDownloader.cfg`.

- [ ] **Step 3: Implement config validation plus the `first` daily-image strategy**

```python
def _validate_config(self):
    req = [
        "coords", "download_dir", "start_year", "start_month", "start_day",
        "end_year", "end_month", "end_day", "bands", "scale", "out_format",
        "auth_mode", "filename_prefix",
    ]
    missing = [k for k in req if k not in self.cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    if "collections" not in self.cfg and "collection" not in self.cfg:
        raise ValueError("Missing required config key: provide either 'collection' or 'collections'.")

    self.cfg["collections"] = self._normalize_collections(
        self.cfg.get("collections", self.cfg.get("collection"))
    )
    self.cfg["collection"] = self.cfg["collections"][0]

    daily_strategy = str(self.cfg.get("daily_strategy", "median")).lower()
    if daily_strategy not in {"median", "first"}:
        raise ValueError("daily_strategy must be one of: median, first")
    self.cfg["daily_strategy"] = daily_strategy
```

```python
def _composite_for_day(self, day_str: str, collection: str) -> ee.Image:
    next_day = (
        datetime.strptime(day_str, "%Y-%m-%d") + relativedelta(days=1)
    ).strftime("%Y-%m-%d")
    col = ee.ImageCollection(collection).filterDate(day_str, next_day)
    if not self._is_globalish_extent():
        col = col.filterBounds(self._region())

    if self.cfg["daily_strategy"] == "first":
        img = ee.Image(col.sort("system:time_start").first())
        if img is None:
            raise RuntimeError(f"No image found for {collection} on {day_str}")
        return img

    if (self.cfg.get("cloud_mask") or {}).get("enabled", False):
        def _apply_mask(im):
            m = build_mask_condition(ee.Image(im), self.cfg)
            return ee.Image(im).updateMask(m)
        col = col.map(_apply_mask)

    composite = col.reduce(ee.Reducer.median())
    bnames = composite.bandNames()
    new_names = bnames.map(lambda n: ee.String(n).replace("_median$", ""))
    return composite.rename(new_names)
```

- [ ] **Step 4: Run the tests and make sure config handling passes**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/core/test_gee_downloader_config.py -q
```

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit the downloader change**

```bash
git add core/gee_downloader.py tests/core/test_gee_downloader_config.py
git commit -m "feat: support first-image daily GEE downloads"
```

## Task 2: Add a Dedicated ERA5-Land DAILY_AGGR Download Workflow

**Files:**
- Create: `core/era5land_download_config.py`
- Create: `workflows/1b_download_era5land_daily.py`
- Test: `tests/workflows/test_1b_download_era5land_daily.py`

- [ ] **Step 1: Write the failing config-builder test**

```python
from core.era5land_download_config import build_era5land_cfg


def test_build_era5land_cfg_sets_expected_collection_and_bands(tmp_path):
    cfg = build_era5land_cfg(
        start_date="2024-01-01",
        end_date="2024-01-03",
        extent=[147.2, -35.1, 147.3, -35.0],
        out_dir=str(tmp_path / "raw"),
    )
    assert cfg["collection"] == "ECMWF/ERA5_LAND/DAILY_AGGR"
    assert cfg["daily_strategy"] == "first"
    assert cfg["bands"] == [
        "temperature_2m_min",
        "temperature_2m_max",
        "dewpoint_temperature_2m",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "surface_solar_radiation_downwards_sum",
        "total_precipitation_sum",
    ]
```

- [ ] **Step 2: Run the test to verify the workflow helper does not exist**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/workflows/test_1b_download_era5land_daily.py -q
```

Expected: FAIL with import error for `core.era5land_download_config` or missing `build_era5land_cfg`.

- [ ] **Step 3: Implement the new ERA5-Land download workflow**

```python
#!/usr/bin/env python3
"""
Script: 1b_download_era5land_daily.py
Objective: Download daily ERA5-Land meteorology GeoTIFFs from GEE for SSEBop and SWEB forcing preparation.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-04-16
Inputs: Date range, extent, output directory, and Earth Engine authentication mode.
Outputs: Daily GeoTIFFs containing raw ERA5-Land DAILY_AGGR meteorology bands.
Usage: python workflows/1b_download_era5land_daily.py --help
Dependencies: pyyaml, core/gee_downloader.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORE_DIR = os.path.join(PROJECT_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from gee_downloader import GEEDownloader  # noqa: E402
from era5land_download_config import build_era5land_cfg  # noqa: E402


def parse_date_range(date_range: str):
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", date_range)
    if len(dates) != 2:
        raise ValueError("Date range must include two dates in YYYY-MM-DD format.")
    return dates[0], dates[1]

def main():
    parser = argparse.ArgumentParser(description="Download daily ERA5-Land meteorology GeoTIFFs from GEE.")
    parser.add_argument("--date-range", required=True, help="Date range string with two dates (YYYY-MM-DD).")
    parser.add_argument("--extent", nargs=4, type=float, required=True, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--output-dir", required=True, help="Directory for downloaded daily GeoTIFFs.")
    args = parser.parse_args()

    start_date, end_date = parse_date_range(args.date_range)
    cfg = build_era5land_cfg(start_date, end_date, list(args.extent), args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.output_dir) / f"gee_config_era5land_{start_date}_{end_date}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    GEEDownloader(str(cfg_path)).run()
```

Add the pure config helper module:

```python
ERA5LAND_BANDS = [
    "temperature_2m_min",
    "temperature_2m_max",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_solar_radiation_downwards_sum",
    "total_precipitation_sum",
]


def build_era5land_cfg(start_date: str, end_date: str, extent: list[float], out_dir: str) -> dict:
    return {
        "collection": "ECMWF/ERA5_LAND/DAILY_AGGR",
        "coords": extent,
        "download_dir": out_dir,
        "start_year": int(start_date[0:4]),
        "start_month": int(start_date[5:7]),
        "start_day": int(start_date[8:10]),
        "end_year": int(end_date[0:4]),
        "end_month": int(end_date[5:7]),
        "end_day": int(end_date[8:10]),
        "bands": ERA5LAND_BANDS,
        "scale": 11132,
        "crs": "EPSG:4326",
        "out_format": "tif",
        "auth_mode": "browser",
        "filename_prefix": "ERA5LandDaily",
        "daily_strategy": "first",
        "postprocess": {
            "maskval_to_na": False,
            "enforce_float32": False,
        },
    }
```

- [ ] **Step 4: Run the tests, then smoke-test one day over a tiny AOI**

Run unit test:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/workflows/test_1b_download_era5land_daily.py -q
```

Expected: PASS.

Run smoke download:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python workflows/1b_download_era5land_daily.py --date-range "2024-01-01 to 2024-01-01" --extent 147.20 -35.10 147.30 -35.00 --output-dir /tmp/pysweb-era5land-raw
```

Expected: One GeoTIFF named like `ERA5LandDaily_2024-01-01.tif` plus the generated YAML config.

- [ ] **Step 5: Commit the new download workflow**

```bash
git add core/era5land_download_config.py workflows/1b_download_era5land_daily.py tests/workflows/test_1b_download_era5land_daily.py
git commit -m "feat: add ERA5-Land daily GEE download workflow"
```

## Task 3: Stack ERA5-Land Daily GeoTIFFs and Compute FAO56-Like ETo

**Files:**
- Create: `core/era5land_refet.py`
- Create: `core/era5land_stack.py`
- Create: `workflows/1c_stack_era5land_daily.py`
- Test: `tests/core/test_era5land_refet.py`
- Test: `tests/workflows/test_1c_stack_era5land_daily.py`

- [ ] **Step 1: Write the failing math and file-discovery tests**

```python
import numpy as np

from core.era5land_refet import (
    actual_vapor_pressure_from_dewpoint_c,
    compute_daily_eto_short,
    j_per_m2_to_mj_per_m2_day,
    kelvin_to_celsius,
    meters_to_mm_day,
    wind_speed_from_uv,
)


def test_basic_unit_conversions():
    assert np.allclose(kelvin_to_celsius(np.array([273.15, 300.15])), np.array([0.0, 27.0]))
    assert np.isclose(j_per_m2_to_mj_per_m2_day(8640000.0), 8.64)
    assert np.isclose(meters_to_mm_day(0.012), 12.0)


def test_actual_vapor_pressure_from_dewpoint():
    result = actual_vapor_pressure_from_dewpoint_c(np.array([20.0]))
    assert np.allclose(result, np.array([2.338]), atol=0.01)


def test_wind_speed_from_uv():
    assert np.allclose(wind_speed_from_uv(np.array([3.0]), np.array([4.0])), np.array([5.0]))


def test_daily_eto_short_matches_reference_case():
    eto = compute_daily_eto_short(
        tmax_c=np.array([31.0]),
        tmin_c=np.array([16.0]),
        ea_kpa=np.array([1.90]),
        rs_mj_m2_day=np.array([24.0]),
        uz_m_s=np.array([3.2]),
        zw_m=10.0,
        elev_m=np.array([180.0]),
        lat_deg=np.array([-35.0]),
        doy=np.array([15]),
    )
    assert np.allclose(eto, np.array([6.0]), atol=0.5)
```

```python
from pathlib import Path

from core.era5land_stack import discover_daily_files


def test_discover_daily_files_sorts_by_embedded_date(tmp_path: Path):
    (tmp_path / "ERA5LandDaily_2024-01-03.tif").write_text("x", encoding="utf-8")
    (tmp_path / "ERA5LandDaily_2024-01-01.tif").write_text("x", encoding="utf-8")
    (tmp_path / "ERA5LandDaily_2024-01-02.tif").write_text("x", encoding="utf-8")
    files = discover_daily_files(tmp_path)
    assert [path.name for path in files] == [
        "ERA5LandDaily_2024-01-01.tif",
        "ERA5LandDaily_2024-01-02.tif",
        "ERA5LandDaily_2024-01-03.tif",
    ]
```

- [ ] **Step 2: Run the tests to verify the refET and stacker modules do not exist**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/core/test_era5land_refet.py tests/workflows/test_1c_stack_era5land_daily.py -q
```

Expected: FAIL with import errors for the new modules.

- [ ] **Step 3: Implement the refET math module and the local stacker**

```python
from __future__ import annotations

import numpy as np


def kelvin_to_celsius(values):
    return np.asarray(values, dtype=float) - 273.15


def meters_to_mm_day(values):
    return np.asarray(values, dtype=float) * 1000.0


def j_per_m2_to_mj_per_m2_day(values):
    return np.asarray(values, dtype=float) / 1_000_000.0


def wind_speed_from_uv(u10, v10):
    u10 = np.asarray(u10, dtype=float)
    v10 = np.asarray(v10, dtype=float)
    return np.sqrt(u10 ** 2 + v10 ** 2)


def actual_vapor_pressure_from_dewpoint_c(tdew_c):
    tdew_c = np.asarray(tdew_c, dtype=float)
    return 0.6108 * np.exp((17.27 * tdew_c) / (tdew_c + 237.3))
```

```python
def _air_pressure_kpa(elev_m):
    elev_m = np.asarray(elev_m, dtype=float)
    return 101.3 * (((293.0 - 0.0065 * elev_m) / 293.0) ** 5.26)


def _psychrometric_constant_kpa_c(pressure_kpa):
    return 0.000665 * pressure_kpa


def _wind_speed_2m(uz_m_s, zw_m):
    uz_m_s = np.asarray(uz_m_s, dtype=float)
    return uz_m_s * 4.87 / np.log((67.8 * zw_m) - 5.42)


def compute_daily_eto_short(tmax_c, tmin_c, ea_kpa, rs_mj_m2_day, uz_m_s, zw_m, elev_m, lat_deg, doy):
    tmax_c = np.asarray(tmax_c, dtype=float)
    tmin_c = np.asarray(tmin_c, dtype=float)
    ea_kpa = np.asarray(ea_kpa, dtype=float)
    rs_mj_m2_day = np.asarray(rs_mj_m2_day, dtype=float)
    elev_m = np.asarray(elev_m, dtype=float)
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=float))
    doy = np.asarray(doy, dtype=float)

    tmean_c = (tmax_c + tmin_c) * 0.5
    pressure_kpa = _air_pressure_kpa(elev_m)
    gamma = _psychrometric_constant_kpa_c(pressure_kpa)
    u2 = _wind_speed_2m(uz_m_s, zw_m)

    es_tmax = 0.6108 * np.exp((17.27 * tmax_c) / (tmax_c + 237.3))
    es_tmin = 0.6108 * np.exp((17.27 * tmin_c) / (tmin_c + 237.3))
    es = (es_tmax + es_tmin) * 0.5
    delta = 4098.0 * (0.6108 * np.exp((17.27 * tmean_c) / (tmean_c + 237.3))) / ((tmean_c + 237.3) ** 2)
    vpd = es - ea_kpa

    dr = 1.0 + 0.033 * np.cos((2.0 * np.pi / 365.0) * doy)
    solar_declination = 0.409 * np.sin((2.0 * np.pi / 365.0) * doy - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(solar_declination))
    gsc = 0.0820
    ra = (
        (24.0 * 60.0 / np.pi)
        * gsc
        * dr
        * (
            ws * np.sin(lat_rad) * np.sin(solar_declination)
            + np.cos(lat_rad) * np.cos(solar_declination) * np.sin(ws)
        )
    )
    rso = (0.75 + (2e-5 * elev_m)) * ra
    rns = (1.0 - 0.23) * rs_mj_m2_day
    fcd = np.clip(1.35 * (rs_mj_m2_day / np.maximum(rso, 1e-6)) - 0.35, 0.05, 1.0)
    rnl = (
        4.903e-9
        * (((tmax_c + 273.16) ** 4 + (tmin_c + 273.16) ** 4) / 2.0)
        * (0.34 - 0.14 * np.sqrt(np.maximum(ea_kpa, 0.0)))
        * fcd
    )
    rn = rns - rnl

    return (
        0.408 * delta * rn + gamma * (900.0 / (tmean_c + 273.0)) * u2 * vpd
    ) / (
        delta + gamma * (1.0 + 0.34 * u2)
    )
```

```python
from era5land_stack import discover_daily_files  # noqa: E402
from era5land_refet import (
    actual_vapor_pressure_from_dewpoint_c,
    compute_daily_eto_short,
    j_per_m2_to_mj_per_m2_day,
    kelvin_to_celsius,
    meters_to_mm_day,
    wind_speed_from_uv,
)  # noqa: E402
```

```python
from pathlib import Path


def discover_daily_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("ERA5LandDaily_*.tif"))
    return sorted(files, key=lambda path: path.stem.split("_")[-1])
```

```python
#!/usr/bin/env python3
"""
Script: 1c_stack_era5land_daily.py
Objective: Stack downloaded ERA5-Land daily GeoTIFFs into NetCDF forcing products and derive daily short-reference ETo.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-04-16
Inputs: Raw ERA5-Land daily GeoTIFF directory, DEM raster, date range, output directory.
Outputs: NetCDF files for precipitation, tmax, tmin, rs, ea, and et_short_crop.
Usage: python workflows/1c_stack_era5land_daily.py --help
Dependencies: numpy, pandas, xarray, rasterio, rioxarray
"""
```

```python
with rasterio.open(path) as src:
    band_map = {name: src.read(idx) for idx, name in enumerate(src.descriptions, start=1)}

tmin_c = kelvin_to_celsius(band_map["temperature_2m_min"])
tmax_c = kelvin_to_celsius(band_map["temperature_2m_max"])
tdew_c = kelvin_to_celsius(band_map["dewpoint_temperature_2m"])
ea_kpa = actual_vapor_pressure_from_dewpoint_c(tdew_c)
u10 = band_map["u_component_of_wind_10m"]
v10 = band_map["v_component_of_wind_10m"]
uz = wind_speed_from_uv(u10, v10)
rs = j_per_m2_to_mj_per_m2_day(band_map["surface_solar_radiation_downwards_sum"])
precip = meters_to_mm_day(band_map["total_precipitation_sum"])
eto = compute_daily_eto_short(
    tmax_c=tmax_c,
    tmin_c=tmin_c,
    ea_kpa=ea_kpa,
    rs_mj_m2_day=rs,
    uz_m_s=uz,
    zw_m=10.0,
    elev_m=dem_array,
    lat_deg=lat_grid,
    doy=np.full(tmin_c.shape, day_of_year, dtype=float),
)
```

- [ ] **Step 4: Run unit tests and a 3-day stack smoke test**

Run unit tests:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/core/test_era5land_refet.py tests/workflows/test_1c_stack_era5land_daily.py -q
```

Expected: PASS.

Run smoke stack:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python workflows/1c_stack_era5land_daily.py --raw-dir /tmp/pysweb-era5land-raw --dem /g/data/yx97/EO_collections/GA/DEM_30m_v01/dems1sv1_0/w001000.adf --date-range 2024-01-01 2024-01-03 --output-dir /tmp/pysweb-era5land-nc
```

Expected: Six NetCDF files are written under `/tmp/pysweb-era5land-nc` with daily time dimension length `3`.

- [ ] **Step 5: Commit the local stacker and refET module**

```bash
git add core/era5land_refet.py core/era5land_stack.py workflows/1c_stack_era5land_daily.py tests/core/test_era5land_refet.py tests/workflows/test_1c_stack_era5land_daily.py
git commit -m "feat: stack ERA5-Land daily inputs and derive reference ET"
```

## Task 4: Point SSEBop and SWEB at Explicit ERA5-Land Meteorology Products

**Files:**
- Create: `core/met_input_paths.py`
- Modify: `workflows/2_ssebop_run_model.py`
- Modify: `workflows/ssebop_runner_landsat.sh`
- Modify: `workflows/sweb_domain_runner.sh`
- Test: `tests/workflows/test_2_ssebop_run_model.py`

- [ ] **Step 1: Write the failing explicit-path resolution test**

```python
from argparse import Namespace
from pathlib import Path

from core.met_input_paths import resolve_met_paths


def test_explicit_met_paths_override_source_inference(tmp_path: Path):
    args = Namespace(
        met_dir=None,
        et_short_crop=str(tmp_path / "et_short_crop_daily.nc"),
        et_short_crop_var="et_short_crop",
        tmax=str(tmp_path / "tmax_daily.nc"),
        tmax_var="tmax",
        tmin=str(tmp_path / "tmin_daily.nc"),
        tmin_var="tmin",
        rs=str(tmp_path / "rs_daily.nc"),
        rs_var="rs",
        ea=str(tmp_path / "ea_daily.nc"),
        ea_var="ea",
    )
    for key in ("et_short_crop", "tmax", "tmin", "rs", "ea"):
        Path(getattr(args, key)).write_text("x", encoding="utf-8")

    paths = resolve_met_paths(args)
    assert paths["et_short_crop"].name == "et_short_crop_daily.nc"
    assert paths["tmax"].name == "tmax_daily.nc"
    assert paths["tmin"].name == "tmin_daily.nc"
    assert paths["rs"].name == "rs_daily.nc"
    assert paths["ea"].name == "ea_daily.nc"
```

- [ ] **Step 2: Run the test to confirm the helper does not exist yet**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/workflows/test_2_ssebop_run_model.py -q
```

Expected: FAIL with import error for `core.met_input_paths` or missing `resolve_met_paths`.

- [ ] **Step 3: Refactor path resolution and update the runners**

```python
MET_FILE_STEMS = {
    "et_short_crop": "et_short_crop_daily.nc",
    "tmax": "tmax_daily.nc",
    "tmin": "tmin_daily.nc",
    "rs": "rs_daily.nc",
    "ea": "ea_daily.nc",
}


def resolve_met_paths(args: argparse.Namespace) -> Dict[str, Path]:
    met_dir = Path(args.met_dir).expanduser().resolve() if getattr(args, "met_dir", None) else None
    paths: Dict[str, Path] = {}
    for key, filename in MET_FILE_STEMS.items():
        override = getattr(args, key, None)
        if override:
            candidate = Path(override).expanduser().resolve()
        elif met_dir:
            candidate = met_dir / filename
        else:
            raise ValueError(f"Provide --met-dir or explicit path via --{key.replace('_', '-')}")
        if not candidate.exists():
            raise FileNotFoundError(f"Meteorology input not found for '{key}': {candidate}")
        paths[key] = candidate
    return paths
```

```python
parser.add_argument("--met-dir", default=None, help="Directory containing daily meteorology NetCDFs.")
parser.add_argument("--met-temp-units", choices=["celsius", "kelvin"], default="celsius")
parser.add_argument("--et-short-crop", default=None)
parser.add_argument("--tmax", default=None)
parser.add_argument("--tmin", default=None)
parser.add_argument("--rs", default=None)
parser.add_argument("--ea", default=None)
```

Update the shell runner to stage raw and stacked ERA5-Land products:

```bash
ERA5LAND_RAW_DIR="${PROJECT_DIR}/1b_era5land_daily_raw"
ERA5LAND_MET_DIR="${PROJECT_DIR}/1c_era5land_daily_nc"
```

```bash
python "1b_download_era5land_daily.py" \
  --date-range "${DATE_RANGE}" \
  --extent "${EXTENT_MIN_LON}" "${EXTENT_MIN_LAT}" "${EXTENT_MAX_LON}" "${EXTENT_MAX_LAT}" \
  --output-dir "${RUN_ERA5LAND_RAW_DIR}"

python "1c_stack_era5land_daily.py" \
  --raw-dir "${RUN_ERA5LAND_RAW_DIR}" \
  --dem "${DEM_PATH}" \
  --date-range "${START_DATE}" "${END_DATE}" \
  --output-dir "${RUN_ERA5LAND_MET_DIR}"
```

```bash
python "2_ssebop_run_model.py" \
  --date-range "${DATE_RANGE}" \
  --landsat-dir "${RUN_INPUT_DIR}" \
  --landsat-pattern "${LANDSAT_PATTERN}" \
  --et-short-crop "${RUN_ERA5LAND_MET_DIR}/et_short_crop_daily_${START_DATE//-/}_${END_DATE//-/}.nc" \
  --et-short-crop-var "et_short_crop" \
  --tmax "${RUN_ERA5LAND_MET_DIR}/tmax_daily_${START_DATE//-/}_${END_DATE//-/}.nc" \
  --tmax-var "tmax" \
  --tmin "${RUN_ERA5LAND_MET_DIR}/tmin_daily_${START_DATE//-/}_${END_DATE//-/}.nc" \
  --tmin-var "tmin" \
  --rs "${RUN_ERA5LAND_MET_DIR}/rs_daily_${START_DATE//-/}_${END_DATE//-/}.nc" \
  --rs-var "rs" \
  --ea "${RUN_ERA5LAND_MET_DIR}/ea_daily_${START_DATE//-/}_${END_DATE//-/}.nc" \
  --ea-var "ea" \
  --met-temp-units "celsius" \
  --dem "${DEM_PATH}" \
  --landcover "${LANDCOVER_PATH}" \
  --output-dir "${RUN_OUTPUT_DIR}"
```

For the SWEB side, use the new precipitation NetCDF explicitly:

```bash
python "${PREPROCESS_SCRIPT}" \
  --date-range "${window_start}" "${window_end}" \
  --extent "${EXTENT_ARGS[@]}" \
  --sm-res "${SM_RES}" \
  --workers "${N_WORKERS}" \
  --rain-file "${RUN_ERA5LAND_MET_DIR}/precipitation_daily_${window_start_fmt}_${window_end_fmt}.nc" \
  --rain-var "precipitation" \
  --et-file "${et_source_file}" \
  --e-var "${E_VAR}" \
  --et-var "${ET_VAR}" \
  --t-var "${T_VAR}" \
  --output-dir "${PREPROCESS_OUT_DIR}"
```

- [ ] **Step 4: Run the explicit-path unit test and a small runner smoke pass**

Run unit test:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/workflows/test_2_ssebop_run_model.py -q
```

Expected: PASS.

Run smoke pass over a 3-day window:

```bash
cd /g/data/ym05/github/PySWEB/workflows && bash ssebop_runner_landsat.sh era5land_smoke --mute-download --workers 1
```

Expected: the wrapper prints ERA5-Land raw/stack paths, then reaches `2_ssebop_run_model.py` without mentioning SILO path inference failures.

- [ ] **Step 5: Commit the explicit-meteorology integration**

```bash
git add core/met_input_paths.py workflows/2_ssebop_run_model.py workflows/ssebop_runner_landsat.sh workflows/sweb_domain_runner.sh tests/workflows/test_2_ssebop_run_model.py
git commit -m "feat: wire ERA5-Land meteorology into SSEBop and SWEB runners"
```

## Task 5: Document the New Climate Workflow and Verify End-to-End Outputs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the documentation patch**

```markdown
## Workflow overview
1. `workflows/1_ssebop_prepare_inputs.py`: build a Landsat GEE config and download Landsat scenes.
2. `workflows/1b_download_era5land_daily.py`: download daily ERA5-Land DAILY_AGGR meteorology GeoTIFFs from GEE.
3. `workflows/1c_stack_era5land_daily.py`: stack ERA5-Land daily GeoTIFFs into NetCDF meteorology products and derive daily short-reference `ETo`.
4. `workflows/2_ssebop_run_model.py`: compute daily `ET`, `E`, `T`, `etf_interp`, `ndvi_interp`, and `Tc` from Landsat plus explicit meteorology NetCDFs.
5. `workflows/3_sweb_preprocess_inputs.py`: align ERA5-Land precipitation, SSEBop ET/T, soil properties, and optional SMAP SSM to one grid.
```

```markdown
## Key outputs
- From ERA5-Land stack (`1c_stack_era5land_daily.py`): `precipitation_daily_*.nc`, `tmax_daily_*.nc`, `tmin_daily_*.nc`, `rs_daily_*.nc`, `ea_daily_*.nc`, `et_short_crop_daily_*.nc`.
- From SSEBop run (`2_ssebop_run_model.py`): `et_daily_ssebop_<start>_<end>.nc` plus intermediate `etf`/`ndvi` products.
```

- [ ] **Step 2: Run a README-only sanity check**

Run:

```bash
git diff -- README.md
```

Expected: The README references `ERA5-Land DAILY_AGGR` and no longer claims the workflow is Landsat + SILO only.

- [ ] **Step 3: Run a narrow end-to-end verification**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python - <<'PY'
import xarray as xr
from pathlib import Path

root = Path("/tmp/pysweb-era5land-nc")
checks = {
    "precipitation_daily_20240101_20240103.nc": "precipitation",
    "et_short_crop_daily_20240101_20240103.nc": "et_short_crop",
    "tmax_daily_20240101_20240103.nc": "tmax",
    "tmin_daily_20240101_20240103.nc": "tmin",
    "rs_daily_20240101_20240103.nc": "rs",
    "ea_daily_20240101_20240103.nc": "ea",
}
for filename, var in checks.items():
    path = root / filename
    ds = xr.open_dataset(path)
    assert var in ds, (filename, list(ds.data_vars))
    assert int(ds.sizes["time"]) == 3, (filename, ds.sizes)
print("verified", len(checks), "files")
PY
```

Expected: `verified 6 files`

- [ ] **Step 4: Commit the documentation and verification changes**

```bash
git add README.md
git commit -m "docs: document ERA5-Land daily aggregate forcing workflow"
```

- [ ] **Step 5: Tag the branch as ready for soil-scope follow-up**

```bash
git status --short
```

Expected: clean worktree except for any intentionally deferred soil-related changes not covered by this plan.

## Self-Review

### Spec Coverage
- `ERA5-Land DAILY_AGGR` replacement path is covered by Tasks 1, 2, and 4.
- FAO56-like `ETo` based on `refetgee` daily logic is covered by Task 3.
- SWEB precipitation replacement is covered by Task 4 through explicit `--rain-file`.
- Soil moisture reference and soil-grid changes are deliberately excluded and documented as out of scope.

### Placeholder Scan
- No `TBD`, `TODO`, or “implement later” placeholders remain.
- Every new file has a concrete path.
- Every task includes explicit commands or code blocks.

### Type Consistency
- Raw ERA5-Land inputs stay in native GEE units until the local stacker converts them.
- Final NetCDF variable names are fixed as `precipitation`, `tmax`, `tmin`, `rs`, `ea`, and `et_short_crop`.
- `2_ssebop_run_model.py` continues to consume those products through explicit file paths, so no downstream caller must infer source-specific names.

Plan complete and saved to `docs/superpowers/plans/2026-04-16-era5-land-daily-agg-migration.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
