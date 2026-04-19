# SWB Global Reference SSM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the SWB package migration by moving preprocess and calibration into `pysweb.swb`, replacing SLGA with Earth Engine `OpenLandMap`, and replacing SMAP-DS-specific calibration inputs with neutral `reference_ssm` outputs backed by `gssm1km`.

**Architecture:** Keep the existing SWB run contract stable while moving preprocess and calibration logic from numbered workflow scripts into package modules. `pysweb.swb.preprocess` owns all Earth Engine interaction for `OpenLandMap` soil properties and `gssm1km` reference SSM, while `pysweb.swb.calibrate` remains file-based and consumes `reference_ssm_daily_*.nc` plus the prepared soil and forcing NetCDFs.

**Tech Stack:** Python 3.12, earthengine-api, requests, numpy, pandas, xarray, rioxarray, rasterio, pyproj, scipy, pytest

---

All new Python files in this plan must use the standard script header from `/home/603/yy4778/.codex/docs/script_header_standard.md`.

## File Structure

### Create

- `pysweb/swb/preprocess.py`
- `pysweb/swb/calibrate.py`
- `tests/swb/test_api.py`
- `tests/swb/test_preprocess.py`
- `tests/swb/test_calibrate.py`
- `tests/workflows/test_3_sweb_preprocess_inputs.py`
- `tests/workflows/test_4_sweb_calib_domain.py`

### Modify

- `pysweb/swb/api.py`
- `tests/package/test_pysweb_imports.py`
- `workflows/3_sweb_preprocess_inputs.py`
- `workflows/4_sweb_calib_domain.py`
- `README.md`

### Keep As-Is

- `pysweb/swb/run.py`
- `pysweb/swb/core.py`
- `pysweb/swb/solver.py`
- `workflows/5_sweb_run_model.py`

---

### Task 1: Promote Real SWB Package Entry Points

**Files:**
- Create: `tests/swb/test_api.py`
- Modify: `tests/package/test_pysweb_imports.py`
- Modify: `pysweb/swb/api.py`
- Create: `pysweb/swb/preprocess.py`
- Create: `pysweb/swb/calibrate.py`

- [ ] **Step 1: Write the failing package API tests**

```python
# tests/swb/test_api.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pysweb.swb.api as swb_api


def test_swb_preprocess_dispatches_to_package_module(monkeypatch):
    recorded = {}

    def fake_preprocess_inputs(**kwargs):
        recorded.update(kwargs)
        return "preprocess-ok"

    monkeypatch.setattr(swb_api, "preprocess_inputs", fake_preprocess_inputs, raising=False)

    result = swb_api.preprocess(output_dir = "/tmp/out", reference_source = "gssm1km")

    assert result == "preprocess-ok"
    assert recorded == {
        "output_dir": "/tmp/out",
        "reference_source": "gssm1km",
    }


def test_swb_calibrate_dispatches_to_package_module(monkeypatch):
    recorded = {}

    def fake_calibrate_domain(**kwargs):
        recorded.update(kwargs)
        return "calibrate-ok"

    monkeypatch.setattr(swb_api, "calibrate_domain", fake_calibrate_domain, raising=False)

    result = swb_api.calibrate(reference_ssm = "/tmp/reference.nc", output = "/tmp/params.csv")

    assert result == "calibrate-ok"
    assert recorded == {
        "reference_ssm": "/tmp/reference.nc",
        "output": "/tmp/params.csv",
    }
```

```python
# tests/package/test_pysweb_imports.py
from importlib import import_module
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_package_api_contracts():
    ssebop = import_module("pysweb.ssebop")
    swb = import_module("pysweb.swb")

    assert callable(ssebop.prepare_inputs)
    assert callable(ssebop.run)
    assert callable(swb.preprocess)
    assert callable(swb.calibrate)

    with __import__("pytest").raises(ValueError, match="Missing required inputs for SWB run"):
        swb.run()
```

- [ ] **Step 2: Run the tests to confirm the current facade still fails**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/package/test_pysweb_imports.py tests/swb/test_api.py -q
```

Expected: FAIL because `pysweb.swb.preprocess` and `pysweb.swb.calibrate` still raise `NotImplementedError` and `tests/swb/test_api.py` does not exist yet.

- [ ] **Step 3: Add minimal package modules and forwarders**

```python
# pysweb/swb/preprocess.py
"""Package-owned SWB preprocess workflow."""

from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = "Preprocess spatial datasets into aligned SWB NetCDF files.")
    parser.add_argument("--output-dir", required = True)
    parser.add_argument("--reference-source", default = "gssm1km")
    parser.add_argument("--reference-ssm-asset", default = "users/qianrswaterr/GlobalSSM1km0509")
    parser.add_argument("--gee-project", default = "yiyu-research")
    parser.add_argument("--skip-reference-ssm", action = "store_true")
    return parser


def preprocess_inputs(**kwargs):
    return kwargs


def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    return preprocess_inputs(**vars(args))
```

```python
# pysweb/swb/calibrate.py
"""Package-owned SWB calibration workflow."""

from __future__ import annotations

import argparse
from typing import Sequence



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = "Domain-wide calibration using reference SSM.")
    parser.add_argument("--reference-ssm", required = True)
    parser.add_argument("--reference-var", default = "reference_ssm")
    parser.add_argument("--output", required = True)
    return parser


def calibrate_domain(**kwargs):
    return kwargs


def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    return calibrate_domain(**vars(args))
```

```python
# pysweb/swb/api.py
from __future__ import annotations

import sys

from pysweb.swb.calibrate import calibrate_domain
from pysweb.swb.preprocess import preprocess_inputs
from pysweb.swb.run import run_swb_workflow


def preprocess(**kwargs):
    return preprocess_inputs(**kwargs)


def calibrate(**kwargs):
    return calibrate_domain(**kwargs)
```

- [ ] **Step 4: Run the tests to confirm the real entry points exist**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/package/test_pysweb_imports.py tests/swb/test_api.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /g/data/ym05/github/PySWEB && git add pysweb/swb/api.py pysweb/swb/preprocess.py pysweb/swb/calibrate.py tests/package/test_pysweb_imports.py tests/swb/test_api.py && git commit -m "feat: expose swb preprocess and calibrate apis"
```

---

### Task 2: Repoint Workflow 3 to a Package-Owned Preprocess CLI

**Files:**
- Create: `tests/workflows/test_3_sweb_preprocess_inputs.py`
- Modify: `workflows/3_sweb_preprocess_inputs.py`
- Modify: `pysweb/swb/preprocess.py`

- [ ] **Step 1: Write the failing wrapper tests for neutral reference arguments**

```python
# tests/workflows/test_3_sweb_preprocess_inputs.py
from importlib import util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_workflow_module():
    workflow_path = ROOT / "workflows" / "3_sweb_preprocess_inputs.py"
    spec = util.spec_from_file_location("sweb_preprocess_workflow", workflow_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_workflow_help_uses_reference_ssm_terms(monkeypatch, capsys):
    workflow_module = _load_workflow_module()
    monkeypatch.setattr(sys, "argv", ["3_sweb_preprocess_inputs.py", "--help"])

    with pytest.raises(SystemExit) as exc:
        workflow_module.main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--reference-source" in captured.out
    assert "--reference-ssm-asset" in captured.out
    assert "--skip-reference-ssm" in captured.out
    assert "--skip-smap" not in captured.out


def test_workflow_main_forwards_to_package_preprocess(monkeypatch):
    workflow_module = _load_workflow_module()
    recorded = {}

    def fake_preprocess_inputs(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(workflow_module, "preprocess_inputs", fake_preprocess_inputs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "3_sweb_preprocess_inputs.py",
            "--date-range", "2024-01-01", "2024-01-03",
            "--extent", "148.0", "-35.5", "148.1", "-35.4",
            "--sm-res", "0.01",
            "--output-dir", "/tmp/prepped",
        ],
    )

    workflow_module.main()

    assert recorded["output_dir"] == "/tmp/prepped"
    assert recorded["reference_source"] == "gssm1km"
    assert recorded["gee_project"] == "yiyu-research"
```

- [ ] **Step 2: Run the workflow wrapper tests to confirm the old script still advertises SMAP arguments**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/workflows/test_3_sweb_preprocess_inputs.py -q
```

Expected: FAIL because the current workflow still exposes `--smap-*` and owns the full implementation body.

- [ ] **Step 3: Replace the workflow implementation with a thin package wrapper**

```python
# workflows/3_sweb_preprocess_inputs.py
#!/usr/bin/env python3
"""
Script: 3_sweb_preprocess_inputs.py
Objective: Provide a thin CLI wrapper around the package-owned SWB preprocess workflow.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-04-19
Inputs: CLI options for forcing preparation, Earth Engine soil inputs, and reference SSM preparation.
Outputs: Delegated package-owned SWB preprocess outputs in the requested output directory.
Usage: python workflows/3_sweb_preprocess_inputs.py --help
Dependencies: pysweb
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pysweb.swb.preprocess import build_parser, preprocess_inputs


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    preprocess_inputs(**vars(args))


if __name__ == "__main__":
    main()
```

```python
# pysweb/swb/preprocess.py
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = "Preprocess spatial datasets into aligned SWB NetCDF files.")
    parser.add_argument("--start-date", type = str)
    parser.add_argument("--end-date", type = str)
    parser.add_argument("--date-range", nargs = 2, metavar = ("START", "END"))
    parser.add_argument("--extent", nargs = 4, type = float, metavar = ("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--sm-res", type = float, required = True)
    parser.add_argument("--output-dir", required = True)
    parser.add_argument("--reference-source", default = "gssm1km")
    parser.add_argument("--reference-ssm-asset", default = "users/qianrswaterr/GlobalSSM1km0509")
    parser.add_argument("--gee-project", default = "yiyu-research")
    parser.add_argument("--skip-reference-ssm", action = "store_true")
    return parser
```

- [ ] **Step 4: Run the wrapper tests again**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/workflows/test_3_sweb_preprocess_inputs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /g/data/ym05/github/PySWEB && git add workflows/3_sweb_preprocess_inputs.py pysweb/swb/preprocess.py tests/workflows/test_3_sweb_preprocess_inputs.py && git commit -m "refactor: route swb preprocess through package"
```

---

### Task 3: Implement OpenLandMap Soil Derivation and GSSM1km Reference SSM Preprocess

**Files:**
- Create: `tests/swb/test_preprocess.py`
- Modify: `pysweb/swb/preprocess.py`

- [ ] **Step 1: Write failing unit tests for the new Earth Engine-specific helpers**

```python
# tests/swb/test_preprocess.py
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysweb.swb.preprocess import (
    GSSM_SCALE_FACTOR,
    OPENLANDMAP_LAYER_SPECS,
    _parse_gssm_band_date,
    _build_layer_bottoms_mm,
    _matching_gssm_image_ids,
    _rename_reference_ssm,
)


def test_parse_gssm_band_date_parses_daily_band_names():
    assert _parse_gssm_band_date("band_2000_03_05_classification") == pd.Timestamp("2000-03-05")
    assert _parse_gssm_band_date("band_2020_12_31_classification") == pd.Timestamp("2020-12-31")


def test_openlandmap_depth_mapping_matches_swb_layer_bottoms():
    assert OPENLANDMAP_LAYER_SPECS == [
        ("b0", 50.0),
        ("b10", 150.0),
        ("b30", 300.0),
        ("b60", 600.0),
        ("b100", 1000.0),
    ]
    np.testing.assert_allclose(_build_layer_bottoms_mm(), np.array([50.0, 150.0, 300.0, 600.0, 1000.0]))


def test_reference_ssm_scaling_and_naming_are_neutral():
    raw = xr.DataArray(
        np.array([[[250.0]], [[500.0]]], dtype = np.float32),
        dims = ("time", "lat", "lon"),
        coords = {
            "time": pd.to_datetime(["2000-03-05", "2000-03-06"]),
            "lat": np.array([-35.0]),
            "lon": np.array([148.0]),
        },
        name = "gssm_raw",
    )

    renamed = _rename_reference_ssm(raw / GSSM_SCALE_FACTOR)

    assert renamed.name == "reference_ssm"
    assert renamed.attrs["units"] == "m3 m-3"
    np.testing.assert_allclose(renamed.values[:, 0, 0], np.array([0.25, 0.5]))


def test_matching_gssm_image_ids_filters_collection_indices_by_year():
    indices = [
        "SM2000Africa1km",
        "SM2000Asia1_1km",
        "SM2001Africa1km",
    ]

    assert _matching_gssm_image_ids(indices, year = 2000) == [
        "SM2000Africa1km",
        "SM2000Asia1_1km",
    ]
```

- [ ] **Step 2: Run the preprocess helper tests to verify the helpers do not exist yet**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/swb/test_preprocess.py -q
```

Expected: FAIL because the helper constants and functions are not implemented.

- [ ] **Step 3: Implement the Earth Engine-aware preprocess helpers and orchestration**

```python
# pysweb/swb/preprocess.py
from __future__ import annotations

import argparse
import io
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import ee
import numpy as np
import pandas as pd
import requests
import rioxarray
import xarray as xr
from rasterio.enums import Resampling

OPENLANDMAP_LAYER_SPECS = [
    ("b0", 50.0),
    ("b10", 150.0),
    ("b30", 300.0),
    ("b60", 600.0),
    ("b100", 1000.0),
]
GSSM_SCALE_FACTOR = 1000.0
OPENLANDMAP_DATASETS = {
    "clay": "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02",
    "sand": "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02",
    "soc": "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02",
}


def _build_layer_bottoms_mm() -> np.ndarray:
    return np.array([bottom_mm for _, bottom_mm in OPENLANDMAP_LAYER_SPECS], dtype = float)


def _parse_gssm_band_date(name: str) -> pd.Timestamp:
    match = re.fullmatch(r"band_(\d{4})_(\d{2})_(\d{2})_classification", name)
    if match is None:
        raise ValueError(f"Unsupported GSSM band name: {name}")
    year, month, day = match.groups()
    return pd.Timestamp(f"{year}-{month}-{day}")


def _initialize_ee(gee_project: str) -> None:
    try:
        ee.Initialize(project = gee_project)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Earth Engine with project '{gee_project}'") from exc


def _rename_reference_ssm(da: xr.DataArray) -> xr.DataArray:
    renamed = da.rename("reference_ssm")
    renamed.attrs.update({
        "long_name": "Reference surface soil moisture",
        "units": "m3 m-3",
        "source": "gssm1km",
    })
    return renamed
```

```python
# pysweb/swb/preprocess.py

def _validate_reference_source(reference_source: str) -> None:
    if reference_source != "gssm1km":
        raise ValueError(
            f"Unsupported reference_source '{reference_source}'. Only 'gssm1km' is supported."
        )


def _matching_gssm_image_ids(indices: Sequence[str], year: int) -> List[str]:
    prefix = f"SM{year}"
    return [index for index in indices if index.startswith(prefix)]


def _select_openlandmap_bands(image_id: str) -> ee.Image:
    return ee.Image(image_id).select([band_name for band_name, _ in OPENLANDMAP_LAYER_SPECS])


def _load_openlandmap_predictors(extent: Tuple[float, float, float, float], gee_project: str) -> Dict[str, xr.DataArray]:
    _initialize_ee(gee_project)
    region = ee.Geometry.Rectangle(list(extent))
    predictors: Dict[str, xr.DataArray] = {}
    for key, image_id in OPENLANDMAP_DATASETS.items():
        image = _select_openlandmap_bands(image_id).clip(region)
        predictors[key] = _download_ee_multiband_image(image, region, key)
    return predictors


def _load_reference_ssm(
    *,
    extent: Tuple[float, float, float, float],
    dates: Sequence[pd.Timestamp],
    reference_ssm_asset: str,
    gee_project: str,
) -> xr.DataArray:
    _initialize_ee(gee_project)
    region = ee.Geometry.Rectangle(list(extent))
    collection = ee.ImageCollection(reference_ssm_asset)
    all_indices = collection.aggregate_array("system:index").getInfo()
    year_stacks = []
    for year in sorted({date.year for date in dates}):
        year_indices = _matching_gssm_image_ids(all_indices, year)
        if not year_indices:
            raise ValueError(f"No gssm1km tile found for year {year}.")
        year_images = [
            ee.Image(collection.filter(ee.Filter.eq("system:index", image_id)).first()).clip(region)
            for image_id in year_indices
        ]
        year_image = ee.ImageCollection(year_images).mosaic()
        selected = []
        for band_name in year_image.bandNames().getInfo():
            band_date = _parse_gssm_band_date(band_name)
            if dates[0] <= band_date <= dates[-1]:
                selected.append((band_name, band_date))
        if selected:
            raw_year = _download_ee_multiband_image(
                year_image.select([band_name for band_name, _ in selected]),
                region,
                f"gssm_raw_{year}",
            )
            raw_year = raw_year.assign_coords(time = [band_date for _, band_date in selected]).rename({"band": "time"})
            year_stacks.append(raw_year)
    if not year_stacks:
        raise ValueError("Requested dates do not overlap any gssm1km daily bands.")
    raw = xr.concat(year_stacks, dim = "time").sortby("time")
    return _rename_reference_ssm(raw / GSSM_SCALE_FACTOR)
```

```python
# pysweb/swb/preprocess.py

def preprocess_inputs(**kwargs):
    args = _build_args(kwargs)
    _validate_reference_source(args.reference_source)
    start, end = _ensure_date_inputs(args)
    dates = pd.date_range(start = start, end = end, freq = "D")
    grid = _build_target_grid(args)

    rain = process_precipitation(args, grid, start, end)
    effective_precip = compute_effective_precipitation_smith(rain, args.dtype)
    et_components = process_et(args, grid, dates)

    soil_predictors = _load_openlandmap_predictors(tuple(args.extent), args.gee_project)
    soil_arrays = process_soil_properties_from_openlandmap(args, grid, soil_predictors)

    outputs = {
        f"rain_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc": rain,
        f"effective_precip_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc": effective_precip,
        f"et_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc": et_components["et"],
        f"t_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc": et_components["t"],
    }
    if not args.skip_reference_ssm:
        outputs[f"reference_ssm_daily_{start:%Y%m%d}_{end:%Y%m%d}.nc"] = _load_reference_ssm(
            extent = tuple(args.extent),
            dates = dates,
            reference_ssm_asset = args.reference_ssm_asset,
            gee_project = args.gee_project,
        )
    return _write_preprocess_outputs(args.output_dir, outputs, soil_arrays)
```

- [ ] **Step 4: Run the new preprocess unit tests**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/swb/test_preprocess.py tests/workflows/test_3_sweb_preprocess_inputs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /g/data/ym05/github/PySWEB && git add pysweb/swb/preprocess.py tests/swb/test_preprocess.py tests/workflows/test_3_sweb_preprocess_inputs.py && git commit -m "feat: add gee-backed swb preprocess inputs"
```

---

### Task 4: Move Calibration Into `pysweb.swb.calibrate` and Neutralize Reference Inputs

**Files:**
- Create: `tests/swb/test_calibrate.py`
- Create: `tests/workflows/test_4_sweb_calib_domain.py`
- Modify: `pysweb/swb/calibrate.py`
- Modify: `workflows/4_sweb_calib_domain.py`

- [ ] **Step 1: Write the failing calibration tests for neutral `reference_ssm` naming**

```python
# tests/swb/test_calibrate.py
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysweb.swb.calibrate import build_parser


def test_calibration_parser_uses_reference_ssm_names():
    parser = build_parser()
    help_text = parser.format_help()

    assert "--reference-ssm" in help_text
    assert "--reference-var" in help_text
    assert "--smap-ssm" not in help_text
    assert "--smap-var" not in help_text


def test_calibration_parser_defaults_reference_var_to_reference_ssm():
    parser = build_parser()
    args = parser.parse_args([
        "--effective-precip", "/tmp/effective.nc",
        "--et", "/tmp/et.nc",
        "--t", "/tmp/t.nc",
        "--soil-dir", "/tmp/soil",
        "--reference-ssm", "/tmp/reference.nc",
        "--output", "/tmp/calibration.csv",
    ])

    assert args.reference_var == "reference_ssm"
```

```python
# tests/workflows/test_4_sweb_calib_domain.py
from importlib import util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_workflow_module():
    workflow_path = ROOT / "workflows" / "4_sweb_calib_domain.py"
    spec = util.spec_from_file_location("sweb_calibrate_workflow", workflow_path)
    module = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_workflow_main_forwards_reference_ssm_args(monkeypatch):
    workflow_module = _load_workflow_module()
    recorded = {}

    def fake_calibrate_domain(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(workflow_module, "calibrate_domain", fake_calibrate_domain)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "4_sweb_calib_domain.py",
            "--effective-precip", "/tmp/effective.nc",
            "--et", "/tmp/et.nc",
            "--t", "/tmp/t.nc",
            "--soil-dir", "/tmp/soil",
            "--reference-ssm", "/tmp/reference.nc",
            "--output", "/tmp/calibration.csv",
        ],
    )

    workflow_module.main()

    assert recorded["reference_ssm"] == "/tmp/reference.nc"
    assert recorded["reference_var"] == "reference_ssm"
```

- [ ] **Step 2: Run the calibration tests to confirm the old script and parser still use SMAP names**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/swb/test_calibrate.py tests/workflows/test_4_sweb_calib_domain.py -q
```

Expected: FAIL because the current parser still requires `--smap-ssm` and the workflow owns the implementation body.

- [ ] **Step 3: Move the implementation into `pysweb.swb.calibrate` and swap to `reference_ssm` naming**

```python
# pysweb/swb/calibrate.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import differential_evolution

from pysweb.swb.solver import soil_water_balance_1d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = "Domain-wide calibration using reference SSM.")
    parser.add_argument("--effective-precip", required = True)
    parser.add_argument("--effective-precip-var", default = "effective_precipitation")
    parser.add_argument("--et", required = True)
    parser.add_argument("--et-var", default = "et")
    parser.add_argument("--t", required = True)
    parser.add_argument("--t-var", default = "t")
    parser.add_argument("--soil-dir", required = True)
    parser.add_argument("--reference-ssm", required = True)
    parser.add_argument("--reference-var", default = "reference_ssm")
    parser.add_argument("--output", required = True)
    return parser
```

```python
# pysweb/swb/calibrate.py

def calibrate_domain(**kwargs):
    args = _build_args(kwargs)
    start, end = _parse_dates(args)
    effective_precip = _load_single_variable(Path(args.effective_precip), args.effective_precip_var)
    et = _load_single_variable(Path(args.et), args.et_var)
    t = _load_single_variable(Path(args.t), args.t_var)
    reference_ssm = _load_single_variable(Path(args.reference_ssm), args.reference_var)

    effective_precip = effective_precip.sel(time = slice(start, end)).transpose("time", args.lat_dim, args.lon_dim)
    et = et.sel(time = effective_precip.coords["time"]).transpose("time", args.lat_dim, args.lon_dim)
    t = t.sel(time = effective_precip.coords["time"]).transpose("time", args.lat_dim, args.lon_dim)
    reference_ssm = reference_ssm.sel(time = effective_precip.coords["time"]).transpose("time", args.lat_dim, args.lon_dim)

    return _run_domain_calibration(args, effective_precip, et, t, reference_ssm)
```

```python
# workflows/4_sweb_calib_domain.py
#!/usr/bin/env python3
"""
Script: 4_sweb_calib_domain.py
Objective: Provide a thin CLI wrapper around the package-owned SWB calibration workflow.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-04-19
Inputs: CLI options for prepared forcing, soil NetCDFs, and reference SSM NetCDF.
Outputs: Delegated package-owned calibration CSV.
Usage: python workflows/4_sweb_calib_domain.py --help
Dependencies: pysweb
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pysweb.swb.calibrate import build_parser, calibrate_domain


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    calibrate_domain(**vars(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the calibration tests again**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/swb/test_calibrate.py tests/workflows/test_4_sweb_calib_domain.py tests/swb/test_api.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /g/data/ym05/github/PySWEB && git add pysweb/swb/calibrate.py workflows/4_sweb_calib_domain.py tests/swb/test_calibrate.py tests/workflows/test_4_sweb_calib_domain.py tests/swb/test_api.py && git commit -m "refactor: move swb calibration into package"
```

---

### Task 5: Update Documentation and Run End-to-End Verification

**Files:**
- Modify: `README.md`
- Modify: `tests/package/test_pysweb_imports.py`
- Test: `tests/package/test_pysweb_imports.py`
- Test: `tests/swb/test_api.py`
- Test: `tests/swb/test_preprocess.py`
- Test: `tests/swb/test_calibrate.py`
- Test: `tests/workflows/test_3_sweb_preprocess_inputs.py`
- Test: `tests/workflows/test_4_sweb_calib_domain.py`

- [ ] **Step 1: Write the failing README assertions in package/import tests**

```python
# tests/package/test_pysweb_imports.py
from importlib import import_module
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_swb_modules_import_cleanly():
    assert import_module("pysweb.swb.preprocess").__name__ == "pysweb.swb.preprocess"
    assert import_module("pysweb.swb.calibrate").__name__ == "pysweb.swb.calibrate"
```

- [ ] **Step 2: Run the focused regression suite before README updates**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/package/test_pysweb_imports.py tests/swb/test_api.py tests/swb/test_preprocess.py tests/swb/test_calibrate.py tests/workflows/test_3_sweb_preprocess_inputs.py tests/workflows/test_4_sweb_calib_domain.py -q
```

Expected: PASS.

- [ ] **Step 3: Update the README to document the packaged SWB path and the new data sources**

```markdown
## Development status

`pysweb.ssebop`, `pysweb.swb.preprocess`, `pysweb.swb.calibrate`, and `pysweb.swb.run` are wired today.

For SWB preprocessing and calibration:

- Soil inputs now default to Earth Engine `OpenLandMap` products rather than local SLGA rasters.
- The default calibration reference is `gssm1km` from `users/qianrswaterr/GlobalSSM1km0509`.
- Preprocess writes `reference_ssm_daily_*.nc` instead of `smap_ssm_daily_*.nc`.
- `workflows/3_sweb_preprocess_inputs.py` and `workflows/4_sweb_calib_domain.py` are thin wrappers over `pysweb.swb.preprocess` and `pysweb.swb.calibrate`.
```

- [ ] **Step 4: Run the full SWB verification suite and inspect output**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/package/test_pysweb_imports.py tests/swb/test_api.py tests/swb/test_preprocess.py tests/swb/test_calibrate.py tests/swb/test_core.py tests/workflows/test_3_sweb_preprocess_inputs.py tests/workflows/test_4_sweb_calib_domain.py tests/workflows/test_5_sweb_run_model.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /g/data/ym05/github/PySWEB && git add README.md tests/package/test_pysweb_imports.py tests/swb/test_api.py tests/swb/test_preprocess.py tests/swb/test_calibrate.py tests/workflows/test_3_sweb_preprocess_inputs.py tests/workflows/test_4_sweb_calib_domain.py && git commit -m "docs: describe swb global reference ssm workflow"
```
