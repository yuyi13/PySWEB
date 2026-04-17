# PySWEB Package Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor PySWEB into a light `pysweb` package with explicit `ssebop` and `swb` top-level APIs, move meteorology source logic under `pysweb.met`, rename the first SSEBop step to `prepare_inputs`, and keep shell runners only as thin wrappers over Python entrypoints.

**Architecture:** Build a real `pysweb/` package first. Then move reusable code out of numbered workflow scripts and `core/` modules into package modules. Keep workflow scripts as parser-only CLIs and keep shell runners as convenience wrappers. Treat `ERA5-Land` and `SILO` as interchangeable meteorology adapters under `pysweb.met`, not as part of SSEBop physics.

**Tech Stack:** Python 3.12, setuptools, xarray, NumPy, rasterio/rioxarray, Earth Engine helper scripts, pytest, bash wrappers, editable install via `pyproject.toml`

---

## File Structure

### Create

- `pyproject.toml`
- `pysweb/__init__.py`
- `pysweb/ssebop/__init__.py`
- `pysweb/ssebop/api.py`
- `pysweb/ssebop/core.py`
- `pysweb/ssebop/landcover.py`
- `pysweb/ssebop/grid.py`
- `pysweb/ssebop/inputs/__init__.py`
- `pysweb/ssebop/inputs/landsat.py`
- `pysweb/swb/__init__.py`
- `pysweb/swb/api.py`
- `pysweb/swb/preprocess.py`
- `pysweb/swb/calibrate.py`
- `pysweb/swb/run.py`
- `pysweb/met/__init__.py`
- `pysweb/met/paths.py`
- `pysweb/met/era5land/__init__.py`
- `pysweb/met/era5land/download.py`
- `pysweb/met/era5land/stack.py`
- `pysweb/met/era5land/refet.py`
- `pysweb/met/silo/__init__.py`
- `pysweb/met/silo/paths.py`
- `pysweb/met/silo/readers.py`
- `pysweb/io/__init__.py`
- `pysweb/io/gee.py`
- `tests/package/test_pysweb_imports.py`
- `tests/ssebop/test_core.py`
- `tests/met/test_paths.py`
- `tests/ssebop/test_api_prepare_inputs.py`
- `tests/ssebop/test_api_run.py`
- `tests/swb/test_api.py`
- `tests/workflows/test_1_ssebop_prepare_inputs.py`

### Modify

- `core/ssebop_au.py`
- `core/era5land_download_config.py`
- `core/era5land_refet.py`
- `core/era5land_stack.py`
- `core/met_input_paths.py`
- `core/gee_downloader.py`
- `workflows/1b_download_era5land_daily.py`
- `workflows/1c_stack_era5land_daily.py`
- `workflows/2_ssebop_run_model.py`
- `workflows/3_sweb_preprocess_inputs.py`
- `workflows/4_sweb_calib_domain.py`
- `workflows/5_sweb_run_model.py`
- `workflows/ssebop_runner_landsat.sh`
- `workflows/sweb_domain_runner.sh`
- `README.md`

### Rename / Delete

- Rename `workflows/1_ssebop_preprocess_inputs.py` to `workflows/1_ssebop_prepare_inputs.py`
- Delete any lingering imports that reference `workflows/1_ssebop_preprocess_inputs.py`
- Remove `core/` compatibility imports only after all runtime callers use `pysweb.*`

---

## Task 1: Create the `pysweb` Package Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `pysweb/__init__.py`
- Create: `pysweb/ssebop/__init__.py`
- Create: `pysweb/swb/__init__.py`
- Create: `pysweb/met/__init__.py`
- Create: `pysweb/io/__init__.py`
- Create: `pysweb/ssebop/api.py`
- Create: `pysweb/swb/api.py`
- Test: `tests/package/test_pysweb_imports.py`

- [ ] **Step 1: Write the failing package smoke test**

```python
from importlib import import_module


def test_top_level_package_exposes_ssebop_and_swb():
    pysweb = import_module("pysweb")

    assert hasattr(pysweb, "ssebop")
    assert hasattr(pysweb, "swb")


def test_subpackages_import_cleanly():
    assert import_module("pysweb.ssebop").__name__ == "pysweb.ssebop"
    assert import_module("pysweb.swb").__name__ == "pysweb.swb"
    assert import_module("pysweb.met").__name__ == "pysweb.met"
```

- [ ] **Step 2: Run the test to confirm the package does not exist yet**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/package/test_pysweb_imports.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pysweb'`.

- [ ] **Step 3: Add package metadata and the minimal namespace**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "PySWEB"
version = "0.0.0"
description = "Experimental package-first interfaces for SSEBop and SWB workflows."
requires-python = ">=3.12"

[tool.setuptools]
packages = [
  "pysweb",
  "pysweb.ssebop",
  "pysweb.ssebop.inputs",
  "pysweb.swb",
  "pysweb.met",
  "pysweb.met.era5land",
  "pysweb.met.silo",
  "pysweb.io",
]
```

```python
# pysweb/__init__.py
from . import met, ssebop, swb

__all__ = ["met", "ssebop", "swb"]
```

```python
# pysweb/ssebop/__init__.py
from .api import prepare_inputs, run

__all__ = ["prepare_inputs", "run"]
```

```python
# pysweb/swb/__init__.py
from .api import calibrate, preprocess, run

__all__ = ["preprocess", "calibrate", "run"]
```

- [ ] **Step 4: Add minimal API modules that fail clearly until wired**

```python
# pysweb/ssebop/api.py
def prepare_inputs(*args, **kwargs):
    raise NotImplementedError("pysweb.ssebop.prepare_inputs is not wired yet")


def run(*args, **kwargs):
    raise NotImplementedError("pysweb.ssebop.run is not wired yet")
```

```python
# pysweb/swb/api.py
def preprocess(*args, **kwargs):
    raise NotImplementedError("pysweb.swb.preprocess is not wired yet")


def calibrate(*args, **kwargs):
    raise NotImplementedError("pysweb.swb.calibrate is not wired yet")


def run(*args, **kwargs):
    raise NotImplementedError("pysweb.swb.run is not wired yet")
```

- [ ] **Step 5: Run the smoke test again**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/package/test_pysweb_imports.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml pysweb tests/package/test_pysweb_imports.py
git commit -m "feat: add pysweb package skeleton"
```

---

## Task 2: Extract SSEBop Physics and Grid Helpers into `pysweb.ssebop`

**Files:**
- Create: `pysweb/ssebop/core.py`
- Create: `pysweb/ssebop/landcover.py`
- Create: `pysweb/ssebop/grid.py`
- Modify: `workflows/2_ssebop_run_model.py`
- Modify: `core/ssebop_au.py`
- Test: `tests/ssebop/test_core.py`

- [ ] **Step 1: Write the failing SSEBop package tests**

```python
import numpy as np
import pandas as pd
import xarray as xr

from pysweb.ssebop.core import build_doy_climatology, et_fraction_xr
from pysweb.ssebop.landcover import worldcover_masks


def test_et_fraction_xr_clamps_and_masks():
    lst = xr.DataArray(np.array([[305.0]], dtype=float), dims=("y", "x"))
    tcold = xr.DataArray(np.array([[300.0]], dtype=float), dims=("y", "x"))
    dt = xr.DataArray(np.array([[4.0]], dtype=float), dims=("y", "x"))

    result = et_fraction_xr(lst, tcold, dt, clamp_max=1.0, mask_max=2.0)

    np.testing.assert_allclose(result.values, [[0.0]])


def test_build_doy_climatology_groups_by_dayofyear():
    data = xr.DataArray(
        [1.0, 3.0],
        coords={"time": pd.to_datetime(["2024-01-01", "2025-01-01"])},
        dims=("time",),
    )

    result = build_doy_climatology(data)

    assert int(result["dayofyear"].values[0]) == 1
    np.testing.assert_allclose(result.values, [2.0])


def test_worldcover_masks_preserve_expected_classes():
    landcover = xr.DataArray(np.array([[30, 50, 80]], dtype=np.uint8), dims=("y", "x"))
    ag_mask, anomalous_mask, water_mask = worldcover_masks(landcover)

    np.testing.assert_array_equal(ag_mask.values, [[1, 0, 0]])
    np.testing.assert_array_equal(anomalous_mask.values, [[0, 1, 0]])
    np.testing.assert_array_equal(water_mask.values, [[0, 0, 1]])
```

- [ ] **Step 2: Run the test to confirm the new modules do not exist yet**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/ssebop/test_core.py -q
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Copy the SSEBop implementations out of `core/ssebop_au.py`**

Move these functions into `pysweb/ssebop/core.py`:
- `dt_fao56_xr`
- `et_fraction_xr`
- `tcold_fano_simple_xr`
- `build_doy_climatology`
- `compute_dt_daily`
- `daily_et_from_etf`

Move these functions into `pysweb/ssebop/landcover.py`:
- `load_worldcover_landcover`
- `worldcover_masks`

Move these functions into `pysweb/ssebop/grid.py`:
- `reproject_match`
- `reproject_match_crop_first`

Also add clean exports:

```python
__all__ = [
    "build_doy_climatology",
    "compute_dt_daily",
    "daily_et_from_etf",
    "dt_fao56_xr",
    "et_fraction_xr",
    "tcold_fano_simple_xr",
]
```

```python
__all__ = ["load_worldcover_landcover", "worldcover_masks"]
```

```python
__all__ = ["reproject_match", "reproject_match_crop_first"]
```

- [ ] **Step 4: Repoint Workflow 2 to the package modules**

Update `workflows/2_ssebop_run_model.py` imports from the legacy `ssebop_au` module to:
- `pysweb.ssebop.core`
- `pysweb.ssebop.grid`
- `pysweb.ssebop.landcover`

- [ ] **Step 5: Convert `core/ssebop_au.py` into a compatibility shim**

Leave only:
- `AU_SSEBOP_SOURCE_CANDIDATES`
- `SsebopAuConfig`
- re-exports from `pysweb.ssebop.core`
- re-exports from `pysweb.ssebop.grid`
- re-exports from `pysweb.ssebop.landcover`

This keeps old imports alive during the transition while making `pysweb.ssebop.*` the real home for the code.

- [ ] **Step 6: Run the package tests and the existing Workflow 2 tests**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/ssebop/test_core.py tests/workflows/test_2_ssebop_run_model.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pysweb/ssebop core/ssebop_au.py workflows/2_ssebop_run_model.py tests/ssebop/test_core.py
git commit -m "refactor: extract ssebop helpers into package modules"
```

---

## Task 3: Move Meteorology Source Helpers Under `pysweb.met`

**Files:**
- Create: `pysweb/met/paths.py`
- Create: `pysweb/met/era5land/download.py`
- Create: `pysweb/met/era5land/stack.py`
- Create: `pysweb/met/era5land/refet.py`
- Create: `pysweb/met/silo/paths.py`
- Create: `pysweb/met/silo/readers.py`
- Create: `pysweb/io/gee.py`
- Modify: `workflows/1b_download_era5land_daily.py`
- Modify: `workflows/1c_stack_era5land_daily.py`
- Modify: `workflows/2_ssebop_run_model.py`
- Test: `tests/met/test_paths.py`
- Test: `tests/workflows/test_1b_download_era5land_daily.py`
- Test: `tests/workflows/test_1c_stack_era5land_daily.py`
- Test: `tests/workflows/test_2_ssebop_run_model.py`

- [ ] **Step 1: Write the failing meteorology path tests**

```python
from pathlib import Path

import pytest

from pysweb.met.paths import infer_met_var_from_path, resolve_met_input_paths


def test_resolve_met_input_paths_builds_era5land_stack_name(tmp_path: Path):
    met_dir = tmp_path / "met"
    met_dir.mkdir()
    path = met_dir / "tmax_daily_2024-01-01_2024-01-03.nc"
    path.write_text("", encoding="utf-8")

    result = resolve_met_input_paths(
        field="tmax",
        explicit_path=None,
        met_dir=str(met_dir),
        silo_dir=None,
        date_range="2024-01-01 to 2024-01-03",
    )

    assert result == str(path)


def test_infer_met_var_from_custom_file_uses_default_for_non_legacy_names():
    assert infer_met_var_from_path("custom_tmax.nc", default_var="tmax") == "tmax"


def test_resolve_met_input_paths_raises_for_missing_explicit_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_met_input_paths(
            field="ea",
            explicit_path=str(tmp_path / "missing.nc"),
            met_dir=None,
            silo_dir=None,
            date_range=None,
        )
```

- [ ] **Step 2: Run the test to confirm `pysweb.met` is not present yet**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/met/test_paths.py -q
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Move the path and conversion helpers into package modules**

Copy the current implementations from `core/met_input_paths.py` into `pysweb/met/paths.py`:
- `parse_date_range`
- `years_in_date_range`
- `infer_met_var_from_path`
- `resolve_met_input_paths`
- `SILO_FILENAME_SUFFIX`

Copy the current implementations from `core/era5land_refet.py` into `pysweb/met/era5land/refet.py`:
- `kelvin_to_celsius`
- `actual_vapor_pressure_from_dewpoint_c`
- `wind_speed_from_uv`
- `j_per_m2_to_mj_per_m2_day`
- `meters_to_mm_day`
- `compute_daily_eto_short`

Copy the current implementations from `core/era5land_stack.py` into `pysweb/met/era5land/stack.py`:
- `extract_date_from_path`
- `discover_daily_files`

Copy the current implementation from `core/era5land_download_config.py` into `pysweb/met/era5land/download.py`:
- `ERA5LAND_BANDS`
- `build_era5land_cfg`

Add `pysweb/io/gee.py` as a thin package-level export:

```python
from core.gee_downloader import GEEDownloader, _safe_mkdir

__all__ = ["GEEDownloader", "_safe_mkdir"]
```

- [ ] **Step 4: Add callable ERA5-Land package functions used by both CLI and API layers**

In `pysweb/met/era5land/download.py`, add:

```python
def write_era5land_config(start_date: str, end_date: str, extent: list[float], output_dir: str) -> Path:
    cfg = build_era5land_cfg(start_date=start_date, end_date=end_date, extent=extent, out_dir=output_dir)
    cfg_path = Path(output_dir) / f"gee_config_era5land_{start_date}_{end_date}.yaml"
    os.makedirs(output_dir, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as handle:
        if yaml is None:
            json.dump(cfg, handle, indent=2)
        else:
            yaml.safe_dump(cfg, handle, sort_keys=False)
    return cfg_path


def download_era5land_daily(*, date_range: str, extent: list[float], output_dir: str) -> Path:
    start_date, end_date = parse_date_range(date_range)
    cfg_path = write_era5land_config(start_date, end_date, extent, output_dir)
    GEEDownloader(str(cfg_path)).run()
    return cfg_path
```

In `pysweb/met/era5land/stack.py`, move the current body of `workflows/1c_stack_era5land_daily.py` into:
- `parse_args(argv=None)`
- `stack_era5land_daily_inputs(raw_dir, dem, start_date, end_date, output_dir)`
- `main(argv=None)`

Keep all grid reads, date filtering, derived variable creation, and NetCDF writing in this package module. The workflow script should stop owning the implementation.

In `pysweb/met/silo/readers.py`, add:

```python
def open_silo_da(path: str, variable: str) -> xr.DataArray:
    dataset = xr.open_dataset(path)
    if variable not in dataset:
        raise ValueError(f"Variable '{variable}' not found in {path}")
    return dataset[variable]
```

- [ ] **Step 5: Repoint workflow scripts to package imports**

Required import changes:
- `workflows/1b_download_era5land_daily.py` imports from `pysweb.met.era5land.download` and `pysweb.io.gee`
- `workflows/1c_stack_era5land_daily.py` imports from `pysweb.met.era5land.stack`
- `workflows/2_ssebop_run_model.py` imports from `pysweb.met.paths`

Also make both ERA5-Land workflow files parser-only wrappers:
- keep `build_parser()` or `parse_args()`
- keep `main()`
- call the package function
- remove duplicated implementation from the workflow file

- [ ] **Step 6: Run the meteorology tests and workflow tests**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/met/test_paths.py tests/workflows/test_1b_download_era5land_daily.py tests/workflows/test_1c_stack_era5land_daily.py tests/workflows/test_2_ssebop_run_model.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pysweb/met pysweb/io workflows/1b_download_era5land_daily.py workflows/1c_stack_era5land_daily.py workflows/2_ssebop_run_model.py tests/met/test_paths.py tests/workflows/test_1b_download_era5land_daily.py tests/workflows/test_1c_stack_era5land_daily.py tests/workflows/test_2_ssebop_run_model.py
git commit -m "refactor: move meteorology helpers into pysweb.met"
```

---

## Task 4: Unify the First SSEBop Step Around `pysweb.ssebop.prepare_inputs`

**Files:**
- Create: `pysweb/ssebop/inputs/landsat.py`
- Modify: `pysweb/ssebop/api.py`
- Rename: `workflows/1_ssebop_preprocess_inputs.py` to `workflows/1_ssebop_prepare_inputs.py`
- Modify: `workflows/1b_download_era5land_daily.py`
- Modify: `workflows/1c_stack_era5land_daily.py`
- Test: `tests/ssebop/test_api_prepare_inputs.py`
- Test: `tests/workflows/test_1_ssebop_prepare_inputs.py`

- [ ] **Step 1: Write the failing prepare-inputs orchestration tests**

```python
from pathlib import Path

from pysweb.ssebop.api import prepare_inputs


def test_prepare_inputs_calls_landsat_and_era5land_steps(monkeypatch, tmp_path: Path):
    recorded = []

    monkeypatch.setattr(
        "pysweb.ssebop.inputs.landsat.prepare_landsat_inputs",
        lambda **kwargs: recorded.append(("landsat", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.met.era5land.download.download_era5land_daily",
        lambda **kwargs: recorded.append(("era5land_download", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.met.era5land.stack.stack_era5land_daily_inputs",
        lambda **kwargs: recorded.append(("era5land_stack", kwargs)),
    )

    prepare_inputs(
        date_range="2024-01-01 to 2024-01-03",
        extent=[147.2, -35.1, 147.3, -35.0],
        met_source="era5land",
        landsat_dir=str(tmp_path / "landsat"),
        met_raw_dir=str(tmp_path / "raw"),
        met_stack_dir=str(tmp_path / "stack"),
        dem=str(tmp_path / "dem.tif"),
        gee_config="/tmp/gee.yaml",
    )

    assert [name for name, _ in recorded] == ["landsat", "era5land_download", "era5land_stack"]
```

```python
from importlib import util
from pathlib import Path


def test_unified_first_step_cli_exposes_met_source():
    workflow_path = Path(__file__).resolve().parents[2] / "workflows" / "1_ssebop_prepare_inputs.py"
    spec = util.spec_from_file_location("ssebop_prepare_inputs_workflow", workflow_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--date-range", "2024-01-01 to 2024-01-03",
            "--extent", "147.2,-35.1,147.3,-35.0",
            "--met-source", "era5land",
            "--gee-config", "/tmp/gee.yaml",
            "--out-dir", "/tmp/out",
            "--dem", "/tmp/dem.tif",
        ]
    )

    assert args.met_source == "era5land"
```

- [ ] **Step 2: Run the test to confirm the high-level API is not wired yet**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/ssebop/test_api_prepare_inputs.py tests/workflows/test_1_ssebop_prepare_inputs.py -q
```

Expected: FAIL because `prepare_inputs()` still raises `NotImplementedError`.

- [ ] **Step 3: Move Landsat preparation logic into `pysweb.ssebop.inputs.landsat`**

Copy the reusable pieces from `workflows/1_ssebop_preprocess_inputs.py` into `pysweb/ssebop/inputs/landsat.py`:
- `parse_date_range`
- `parse_extent`
- `_write_gee_config`
- `update_gee_config`
- `write_gee_config_from_cfg`

Add one package-facing execution function:

```python
def prepare_landsat_inputs(
    *,
    date_range: str,
    extent: list[float],
    gee_config: str,
    out_dir: str,
) -> str:
    start_date, end_date = parse_date_range(date_range)
    _safe_mkdir(out_dir)
    cfg_path = update_gee_config(gee_config, start_date, end_date, extent, out_dir)
    GEEDownloader(cfg_path).run()
    return cfg_path
```

If inline-dict GEE config support is still needed, add it inside this module before the final wrapper lands, not inside the workflow script.

- [ ] **Step 4: Implement `pysweb.ssebop.prepare_inputs`**

Update `pysweb/ssebop/api.py` to:
- validate `met_source`
- call `prepare_landsat_inputs`
- call `download_era5land_daily`
- call `stack_era5land_daily_inputs`

Concrete structure:

```python
from pysweb.met.paths import parse_date_range
from pysweb.met.era5land.download import download_era5land_daily
from pysweb.met.era5land.stack import stack_era5land_daily_inputs
from pysweb.ssebop.inputs.landsat import prepare_landsat_inputs


def prepare_inputs(
    *,
    date_range: str,
    extent: list[float],
    met_source: str,
    landsat_dir: str,
    met_raw_dir: str,
    met_stack_dir: str,
    dem: str,
    gee_config: str,
) -> None:
    start_date, end_date = parse_date_range(date_range)
    prepare_landsat_inputs(
        date_range=date_range,
        extent=extent,
        gee_config=gee_config,
        out_dir=landsat_dir,
    )
    if met_source != "era5land":
        raise NotImplementedError(f"Unsupported met_source: {met_source}")
    download_era5land_daily(date_range=date_range, extent=extent, output_dir=met_raw_dir)
    stack_era5land_daily_inputs(
        raw_dir=met_raw_dir,
        dem=dem,
        start_date=start_date,
        end_date=end_date,
        output_dir=met_stack_dir,
    )
```

- [ ] **Step 5: Rename the workflow and turn it into a thin wrapper**

Rename:

```bash
git mv workflows/1_ssebop_preprocess_inputs.py workflows/1_ssebop_prepare_inputs.py
```

Then rewrite the file so it contains:
- `build_parser()`
- `parse_extent()` import from `pysweb.ssebop.inputs.landsat`
- `main()`

The wrapper should call:

```python
prepare_inputs(
    date_range=args.date_range,
    extent=parse_extent(args.extent),
    met_source=args.met_source,
    landsat_dir=args.out_dir,
    met_raw_dir=args.met_raw_dir or os.path.join(args.out_dir, "era5land_raw"),
    met_stack_dir=args.met_stack_dir or os.path.join(args.out_dir, "era5land_stack"),
    dem=args.dem,
    gee_config=args.gee_config,
)
```

`--dem` should now be required in the unified first-step CLI because ERA5-Land stacking needs it.

- [ ] **Step 6: Run the package and workflow tests**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/ssebop/test_api_prepare_inputs.py tests/workflows/test_1_ssebop_prepare_inputs.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pysweb/ssebop/api.py pysweb/ssebop/inputs/landsat.py workflows/1_ssebop_prepare_inputs.py tests/ssebop/test_api_prepare_inputs.py tests/workflows/test_1_ssebop_prepare_inputs.py
git commit -m "feat: unify ssebop prepare step under package api"
```

---

## Task 5: Add `pysweb.ssebop.run` and Make Workflow 2 a Thin CLI

**Files:**
- Modify: `pysweb/ssebop/api.py`
- Modify: `workflows/2_ssebop_run_model.py`
- Test: `tests/ssebop/test_api_run.py`
- Test: `tests/workflows/test_2_ssebop_run_model.py`

- [ ] **Step 1: Write the failing run-API test**

```python
import pytest

from pysweb.ssebop.api import run


def test_ssebop_run_requires_explicit_inputs():
    with pytest.raises(ValueError):
        run(
            date_range="2024-01-01 to 2024-01-03",
            landsat_dir="",
            met_dir="",
            dem="",
            output_dir="",
        )
```

- [ ] **Step 2: Run the tests to confirm `run()` is not implemented**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/ssebop/test_api_run.py tests/workflows/test_2_ssebop_run_model.py -q
```

Expected: FAIL because `pysweb.ssebop.run()` still raises `NotImplementedError`.

- [ ] **Step 3: Move the workflow body from `workflows/2_ssebop_run_model.py` into `pysweb.ssebop.api.run`**

Refactor `workflows/2_ssebop_run_model.py` so that:
- all reusable helper functions stay near the top of the file or move into package modules as needed
- the current `main()` body becomes a callable function:
  - `run_ssebop_workflow(date_range, landsat_dir, met_dir, dem, output_dir, **kwargs)`
- `main()` only parses args and forwards them

Then implement `pysweb.ssebop.api.run` as:
- input validation
- a direct call into `run_ssebop_workflow`

Required shape:

```python
def run(
    *,
    date_range: str,
    landsat_dir: str,
    met_dir: str,
    dem: str,
    output_dir: str,
    **kwargs,
) -> None:
    if not landsat_dir or not met_dir or not dem or not output_dir:
        raise ValueError("landsat_dir, met_dir, dem, and output_dir are required")
    run_ssebop_workflow(
        date_range=date_range,
        landsat_dir=landsat_dir,
        met_dir=met_dir,
        dem=dem,
        output_dir=output_dir,
        **kwargs,
    )
```

Use a normal import path by keeping `run_ssebop_workflow` inside `pysweb.ssebop.api` or another package module. Do not import numbered workflow filenames as Python modules.

- [ ] **Step 4: Reduce Workflow 2 to a parser-only wrapper**

The final `workflows/2_ssebop_run_model.py` should:
- expose `build_parser()`
- parse CLI arguments
- call `pysweb.ssebop.run()`

- [ ] **Step 5: Run the run-API and workflow tests**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/ssebop/test_api_run.py tests/workflows/test_2_ssebop_run_model.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pysweb/ssebop/api.py workflows/2_ssebop_run_model.py tests/ssebop/test_api_run.py tests/workflows/test_2_ssebop_run_model.py
git commit -m "feat: add package-backed ssebop run api"
```

---

## Task 6: Add `pysweb.swb` APIs and Repoint Workflows 3, 4, and 5

**Files:**
- Modify: `pysweb/swb/api.py`
- Create: `pysweb/swb/preprocess.py`
- Create: `pysweb/swb/calibrate.py`
- Create: `pysweb/swb/run.py`
- Modify: `workflows/3_sweb_preprocess_inputs.py`
- Modify: `workflows/4_sweb_calib_domain.py`
- Modify: `workflows/5_sweb_run_model.py`
- Test: `tests/swb/test_api.py`

- [ ] **Step 1: Write the failing SWB package API tests**

```python
from pysweb.swb import calibrate, preprocess, run


def test_swb_package_exports_callable_entrypoints():
    assert callable(preprocess)
    assert callable(calibrate)
    assert callable(run)
```

- [ ] **Step 2: Run the test to confirm the SWB API is not wired**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/swb/test_api.py -q
```

Expected: FAIL because the package API still raises `NotImplementedError`.

- [ ] **Step 3: Move the workflow bodies into package modules**

From `workflows/3_sweb_preprocess_inputs.py`, move the current `main()` path into:
- `pysweb/swb/preprocess.py`
- public function: `preprocess_inputs(**kwargs)`

From `workflows/4_sweb_calib_domain.py`, move the current `main()` path into:
- `pysweb/swb/calibrate.py`
- public function: `calibrate_domain(**kwargs)`

From `workflows/5_sweb_run_model.py`, move the current `main()` path into:
- `pysweb/swb/run.py`
- public function: `run_model(**kwargs)`

Do not import numbered workflow filenames from the package. The implementation must live in the package modules.

- [ ] **Step 4: Wire the top-level SWB API**

Use:

```python
# pysweb/swb/api.py
from .calibrate import calibrate_domain as calibrate
from .preprocess import preprocess_inputs as preprocess
from .run import run_model as run

__all__ = ["preprocess", "calibrate", "run"]
```

- [ ] **Step 5: Reduce Workflows 3, 4, and 5 to parser-only wrappers**

Each workflow file should expose:
- `build_parser()` or `parse_args()`
- `main()`

Each wrapper should then call the package function:
- `pysweb.swb.preprocess()`
- `pysweb.swb.calibrate()`
- `pysweb.swb.run()`

- [ ] **Step 6: Run the SWB API test**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/swb/test_api.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pysweb/swb workflows/3_sweb_preprocess_inputs.py workflows/4_sweb_calib_domain.py workflows/5_sweb_run_model.py tests/swb/test_api.py
git commit -m "feat: add package-backed swb apis"
```

---

## Task 7: Update Runners, Deprecate Legacy Core Imports, and Refresh Docs

**Files:**
- Modify: `workflows/ssebop_runner_landsat.sh`
- Modify: `workflows/sweb_domain_runner.sh`
- Modify: `README.md`
- Modify: `core/ssebop_au.py`
- Modify: `core/era5land_download_config.py`
- Modify: `core/era5land_refet.py`
- Modify: `core/era5land_stack.py`
- Modify: `core/met_input_paths.py`

- [ ] **Step 1: Run shell and docs smoke checks before the final cleanup**

Run:

```bash
bash -n workflows/ssebop_runner_landsat.sh
bash -n workflows/sweb_domain_runner.sh
bash workflows/ssebop_runner_landsat.sh --help
bash workflows/sweb_domain_runner.sh --help
```

Expected: wrappers still describe the older script arrangement until this task is complete.

- [ ] **Step 2: Update both bash runners to call the normalized Python CLIs**

Expected SSEBop runner flow:

```bash
python "${SCRIPT_DIR}/1_ssebop_prepare_inputs.py" \
  --date-range "${DATE_RANGE}" \
  --extent "${EXTENT}" \
  --met-source "era5land" \
  --gee-config "${GEE_CONFIG}" \
  --out-dir "${RUN_INPUT_DIR}" \
  --met-raw-dir "${RUN_MET_RAW_DIR}" \
  --met-stack-dir "${RUN_MET_STACK_DIR}" \
  --dem "${DEM_PATH}"
```

```bash
python "${SCRIPT_DIR}/2_ssebop_run_model.py" \
  --date-range "${DATE_RANGE}" \
  --landsat-dir "${RUN_INPUT_DIR}" \
  --met-dir "${RUN_MET_STACK_DIR}" \
  --dem "${DEM_PATH}" \
  --output-dir "${RUN_OUTPUT_DIR}"
```

The SWB runner should likewise call the package-backed workflow scripts, not own core logic itself.

- [ ] **Step 3: Deprecate the old `core/` modules**

Once all tests and runtime callers use `pysweb.*`, update the old modules so they do one of these:
- re-export the new package module with a deprecation note
- raise a focused `ImportError` that tells users to switch to `pysweb.*`

Recommended end-state for the final pass:
- `core/ssebop_au.py` keeps only compatibility re-exports and a clear deprecation note
- `core/era5land_download_config.py`, `core/era5land_refet.py`, `core/era5land_stack.py`, and `core/met_input_paths.py` do the same

Do not hard-delete them in the same patch unless every test and script has already been updated and verified.

- [ ] **Step 4: Update the README so Python package usage is primary**

Replace the old script-first examples with:

```python
import pysweb

pysweb.ssebop.prepare_inputs(
    date_range="2024-01-01 to 2024-01-31",
    extent=[147.2, -35.1, 147.3, -35.0],
    met_source="era5land",
    landsat_dir="/tmp/ssebop_inputs",
    met_raw_dir="/tmp/era5land_raw",
    met_stack_dir="/tmp/era5land_stack",
    dem="/tmp/dem.tif",
    gee_config="/tmp/gee.yaml",
)

pysweb.ssebop.run(
    date_range="2024-01-01 to 2024-01-31",
    landsat_dir="/tmp/ssebop_inputs",
    met_dir="/tmp/era5land_stack",
    dem="/tmp/dem.tif",
    output_dir="/tmp/ssebop_outputs",
)

pysweb.swb.preprocess(config="/tmp/swb_preprocess.yaml")
pysweb.swb.calibrate(config="/tmp/swb_calibration.yaml")
pysweb.swb.run(config="/tmp/swb_run.yaml")
```

Document the bash runners as convenience wrappers around the Python entrypoints.

- [ ] **Step 5: Run the final targeted verification suite**

Run:

```bash
source /g/data/yx97/users_unikey/yiyu0116/softwares/miniconda3/bin/activate && conda activate geo_env && python -m pytest tests/package/test_pysweb_imports.py tests/ssebop/test_core.py tests/met/test_paths.py tests/ssebop/test_api_prepare_inputs.py tests/ssebop/test_api_run.py tests/swb/test_api.py tests/workflows/test_1_ssebop_prepare_inputs.py tests/workflows/test_1b_download_era5land_daily.py tests/workflows/test_1c_stack_era5land_daily.py tests/workflows/test_2_ssebop_run_model.py -q
bash -n workflows/ssebop_runner_landsat.sh
bash -n workflows/sweb_domain_runner.sh
bash workflows/ssebop_runner_landsat.sh --help
bash workflows/sweb_domain_runner.sh --help
```

Expected: PASS, plus clean shell syntax checks.

- [ ] **Step 6: Commit**

```bash
git add README.md workflows/ssebop_runner_landsat.sh workflows/sweb_domain_runner.sh core/ssebop_au.py core/era5land_download_config.py core/era5land_refet.py core/era5land_stack.py core/met_input_paths.py
git commit -m "refactor: finalize package-backed pysweb interfaces"
```

---

## Self-Review Checklist

- Spec coverage:
  - package namespace and naming: Tasks 1, 2, 3
  - explicit `ssebop` vs `swb` separation: Tasks 2 and 6
  - ERA5-Land and SILO as meteorology adapters: Task 3
  - unified first-step SSEBop prepare interface: Task 4
  - Python-first interfaces with bash wrappers retained: Tasks 5 and 7
  - aggressive rename and downstream update: Tasks 4 and 7
- Plan hygiene:
  - no unfinished markers
  - no invented transition modules outside the approved package layout
  - no imports of numbered workflow filenames from the package
- Naming consistency:
  - top-level package name is always `pysweb`
  - SSEBop API uses `prepare_inputs` and `run`
  - SWB API uses `preprocess`, `calibrate`, and `run`
  - meteorology modules live under `pysweb.met`
