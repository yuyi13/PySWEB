# PySWEB SSEBop `gee_project`, DEM Package, and Landsat Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `gee_project` explicit in the SSEBop package workflow, prepare NASADEM during `prepare_inputs`, move canonical Landsat helpers to `pysweb.ssebop.landsat`, and update the notebook/docs to the new package-backed contract.

**Architecture:** Add a new top-level `pysweb.dem` package with a NASADEM backend and keep `pysweb.ssebop.run(...)` stable by pointing it at a prepared local DEM artifact. Thread `gee_project` through Landsat, ERA5-Land, and DEM acquisition, keep the legacy downloader in place, and use one-transition-round shims where import paths change.

**Tech Stack:** Python 3.12, earthengine-api, requests, rasterio, xarray, numpy, pytest, stdlib `json`, stdlib `pathlib`

---

All new Python files in this plan must use the standard script header from `/home/603/yy4778/.codex/docs/script_header_standard.md`.

## File Structure

### Create

- `pysweb/dem/__init__.py`
- `pysweb/dem/api.py`
- `pysweb/dem/nasadem.py`
- `pysweb/ssebop/landsat.py`
- `tests/dem/__init__.py`
- `tests/dem/test_api.py`
- `tests/dem/test_nasadem.py`
- `tests/ssebop/test_landsat.py`

### Modify

- `pysweb/__init__.py`
- `pysweb/met/era5land/download.py`
- `pysweb/ssebop/api.py`
- `pysweb/ssebop/inputs/landsat.py`
- `core/gee_downloader.py`
- `workflows/1_ssebop_prepare_inputs.py`
- `tests/package/test_docs_and_notebooks.py`
- `tests/package/test_pysweb_imports.py`
- `tests/ssebop/test_api_prepare_inputs.py`
- `tests/workflows/test_1_ssebop_prepare_inputs.py`
- `tests/workflows/test_1b_download_era5land_daily.py`
- `README.md`
- `notebooks/README.md`
- `notebooks/01_run_pysweb.ipynb`

### Keep As-Is

- `pysweb/ssebop/run` entry points and run-step tests
- `pysweb/met/era5land/stack.py` ET0 math and grid validation logic
- `tests/workflows/test_1c_stack_era5land_daily.py`
- `tests/workflows/test_2_ssebop_run_model.py`

---

### Task 1: Scaffold `pysweb.dem` and expose it at package level

**Files:**
- Create: `pysweb/dem/__init__.py`
- Create: `pysweb/dem/api.py`
- Create: `pysweb/dem/nasadem.py`
- Create: `tests/dem/__init__.py`
- Create: `tests/dem/test_api.py`
- Modify: `pysweb/__init__.py`
- Modify: `tests/package/test_pysweb_imports.py`

- [ ] **Step 1: Write the failing import and dispatcher tests**

```python
# tests/dem/test_api.py
#!/usr/bin/env python3
"""
Script: test_api.py
Objective: Verify the DEM package exposes a stable dispatcher surface and explicit source validation.
Author: Yi Yu
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: Pytest execution against the package-level DEM API.
Outputs: Test assertions.
Usage: python -m pytest tests/dem/test_api.py -q
Dependencies: pathlib, sys, pytest
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysweb.dem.api import SUPPORTED_DEM_SOURCES, prepare_dem


def test_supported_dem_sources_are_explicit():
    assert SUPPORTED_DEM_SOURCES == ("nasadem",)


def test_prepare_dem_requires_gee_project_for_gge_backed_sources():
    with pytest.raises(ValueError, match = "gee_project is required"):
        prepare_dem(
            dem_source = "nasadem",
            gee_project = "",
            extent = [147.2, -35.1, 147.3, -35.0],
            output_path = "/tmp/nasadem.tif",
        )


def test_prepare_dem_rejects_unknown_source():
    with pytest.raises(ValueError, match = "Unsupported dem_source 'bogus'"):
        prepare_dem(
            dem_source = "bogus",
            gee_project = "demo-project",
            extent = [147.2, -35.1, 147.3, -35.0],
            output_path = "/tmp/out.tif",
        )
```

```python
# tests/package/test_pysweb_imports.py
def test_top_level_package_exposes_dem():
    pysweb = import_module("pysweb")

    assert hasattr(pysweb, "dem")


def test_dem_subpackage_imports_cleanly():
    assert import_module("pysweb.dem").__name__ == "pysweb.dem"
    assert import_module("pysweb.dem.nasadem").__name__ == "pysweb.dem.nasadem"
```

- [ ] **Step 2: Run the tests to verify the DEM surface does not exist yet**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/package/test_pysweb_imports.py tests/dem/test_api.py -q
```

Expected: FAIL because `pysweb.dem` and the new test module do not exist yet.

- [ ] **Step 3: Add the package scaffolding with a strict dispatcher**

```python
# pysweb/__init__.py
__all__ = ["dem", "io", "met", "ssebop", "soil", "swb", "visualisation"]
```

```python
# pysweb/dem/api.py
#!/usr/bin/env python3
"""
Script: api.py
Objective: Provide the public DEM-backend dispatch surface for package-owned SSEBop preparation.
Author: Yi Yu
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: DEM source selection, GEE project, geographic extent, and local output path.
Outputs: Prepared DEM raster path or explicit selection/validation errors.
Usage: Imported as `pysweb.dem.api`
Dependencies: pathlib
"""
from __future__ import annotations

SUPPORTED_DEM_SOURCES = ("nasadem",)


def prepare_dem(*, dem_source: str, gee_project: str, extent: list[float], output_path: str) -> str:
    if not gee_project:
        raise ValueError("gee_project is required for GEE-backed DEM preparation.")

    if dem_source == "nasadem":
        from pysweb.dem.nasadem import prepare_dem as backend_prepare_dem

        return backend_prepare_dem(
            gee_project = gee_project,
            extent = extent,
            output_path = output_path,
        )

    supported = ", ".join(SUPPORTED_DEM_SOURCES)
    raise ValueError(f"Unsupported dem_source '{dem_source}'. Supported values: {supported}")
```

```python
# pysweb/dem/__init__.py
#!/usr/bin/env python3
"""
Script: __init__.py
Objective: Expose DEM backend modules and the package-level DEM dispatcher.
Author: Yi Yu
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: Package attribute access.
Outputs: Lazily imported DEM modules and dispatcher attributes.
Usage: Imported as `pysweb.dem`
Dependencies: importlib
"""
from importlib import import_module

__all__ = ["api", "nasadem", "prepare_dem"]

_SUBMODULES = {
    "api": "pysweb.dem.api",
    "nasadem": "pysweb.dem.nasadem",
}


def __getattr__(name):
    if name == "prepare_dem":
        value = import_module("pysweb.dem.api").prepare_dem
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'pysweb.dem' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
```

```python
# pysweb/dem/nasadem.py
#!/usr/bin/env python3
"""
Script: nasadem.py
Objective: Hold the NASADEM backend implementation for package-owned SSEBop preparation.
Author: Yi Yu
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: GEE project, target extent, and local output path.
Outputs: Prepared NASADEM raster path.
Usage: Imported as `pysweb.dem.nasadem`
Dependencies: none at scaffold stage
"""
from __future__ import annotations


def prepare_dem(*, gee_project: str, extent: list[float], output_path: str) -> str:
    raise NotImplementedError("pysweb.dem.nasadem.prepare_dem is implemented in Task 3.")
```

- [ ] **Step 4: Run the package-surface tests again**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/package/test_pysweb_imports.py tests/dem/test_api.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the scaffold**

```bash
cd /g/data/ym05/github/PySWEB && git add pysweb/__init__.py pysweb/dem tests/dem tests/package/test_pysweb_imports.py && git commit -m "feat: scaffold pysweb dem package"
```

---

### Task 2: Canonicalize `pysweb.ssebop.landsat` and thread `gee_project` into GEE configs

**Files:**
- Create: `pysweb/ssebop/landsat.py`
- Create: `tests/ssebop/test_landsat.py`
- Modify: `pysweb/ssebop/inputs/landsat.py`
- Modify: `pysweb/met/era5land/download.py`
- Modify: `core/gee_downloader.py`
- Modify: `tests/workflows/test_1b_download_era5land_daily.py`

- [ ] **Step 1: Write the failing canonical-import and config-plumbing tests**

```python
# tests/ssebop/test_landsat.py
#!/usr/bin/env python3
"""
Script: test_landsat.py
Objective: Verify canonical Landsat helpers live under pysweb.ssebop.landsat and still support the legacy shim path.
Author: Yi Yu
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: Temporary config templates and direct module imports.
Outputs: Test assertions.
Usage: python -m pytest tests/ssebop/test_landsat.py -q
Dependencies: importlib, pathlib, sys, json
"""
from importlib import import_module
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysweb.ssebop.landsat import prepare_landsat_inputs, write_gee_config_from_cfg


def test_canonical_landsat_module_imports_cleanly():
    assert import_module("pysweb.ssebop.landsat").__name__ == "pysweb.ssebop.landsat"


def test_legacy_landsat_shim_re_exports_canonical_helpers():
    legacy_module = import_module("pysweb.ssebop.inputs.landsat")
    canonical_module = import_module("pysweb.ssebop.landsat")

    assert legacy_module.prepare_landsat_inputs is canonical_module.prepare_landsat_inputs
    assert legacy_module.parse_extent is canonical_module.parse_extent


def test_write_gee_config_from_cfg_injects_gee_project(tmp_path):
    cfg_path = write_gee_config_from_cfg(
        gee_cfg = {
            "collection": "LANDSAT/LC08/C02/T1_L2",
            "bands": ["SR_B4", "SR_B5", "ST_B10"],
            "scale": 30,
            "out_format": "tif",
            "auth_mode": "browser",
            "filename_prefix": "Landsat_30m",
        },
        start_date = "2024-01-01",
        end_date = "2024-01-03",
        extent = [147.2, -35.1, 147.3, -35.0],
        gee_project = "demo-project",
        out_dir = str(tmp_path),
    )

    payload = json.loads(Path(cfg_path).read_text(encoding = "utf-8"))
    assert payload["gee_project"] == "demo-project"
```

```python
# tests/workflows/test_1b_download_era5land_daily.py
def test_build_era5land_cfg_records_explicit_gee_project(tmp_path):
    cfg = build_era5land_cfg(
        start_date = "2024-01-01",
        end_date = "2024-01-03",
        extent = [147.2, -35.1, 147.3, -35.0],
        gee_project = "demo-project",
        out_dir = str(tmp_path / "raw"),
    )

    assert cfg["gee_project"] == "demo-project"
```

```python
# tests/workflows/test_1b_download_era5land_daily.py
def test_legacy_downloader_initialize_uses_configured_gee_project(monkeypatch):
    import core.gee_downloader as gee_downloader

    calls = {
        "initialize_projects": [],
    }

    class FakeEE:
        @staticmethod
        def Initialize(*args, **kwargs):
            calls["initialize_projects"].append(kwargs.get("project"))

        @staticmethod
        def Authenticate():
            raise AssertionError("Authenticate should not be called in this unit test")

    monkeypatch.setattr(gee_downloader, "ee", FakeEE)

    downloader = gee_downloader.GEEDownloader.__new__(gee_downloader.GEEDownloader)
    downloader.cfg = {"auth_mode": "browser", "gee_project": "demo-project"}
    downloader.initialize()

    assert calls["initialize_projects"] == ["demo-project"]
```

- [ ] **Step 2: Run the tests and confirm the new import/config contract fails**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/ssebop/test_landsat.py tests/workflows/test_1b_download_era5land_daily.py -q
```

Expected: FAIL because `pysweb.ssebop.landsat` does not exist and neither Landsat nor ERA5-Land config builders carry `gee_project`.

- [ ] **Step 3: Implement the canonical Landsat module, shim, and config-project plumbing**

```python
# pysweb/ssebop/landsat.py
#!/usr/bin/env python3
"""
Script: landsat.py
Objective: Provide canonical Landsat preparation helpers for pysweb.ssebop.
Author: Yi Yu
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: Date ranges, extents, Earth Engine project ids, and optional config templates.
Outputs: Derived Landsat downloader configs and downloaded GeoTIFF scenes.
Usage: Imported as `pysweb.ssebop.landsat`
Dependencies: json, re, datetime, pathlib, pysweb.io.gee
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from pysweb.io.gee import GEEDownloader, _safe_mkdir

DEFAULT_LANDSAT_GEE_CFG = {
    "filename_prefix": "Landsat_30m",
    "collections": [
        "LANDSAT/LC08/C02/T1_L2",
        "LANDSAT/LC09/C02/T1_L2",
    ],
    "collection": "LANDSAT/LC08/C02/T1_L2",
    "bands": ["SR_B4", "SR_B5", "ST_B10"],
    "scale": 30,
    "out_format": "tif",
    "auth_mode": "browser",
    "cloud_mask": {
        "enabled": True,
        "band": "QA_PIXEL",
        "type": "bits_any",
        "bits": [0, 1, 2, 3, 4, 5],
        "keep": False,
    },
    "postprocess": {
        "maskval_to_na": True,
        "enforce_float32": False,
    },
}


def _load_yaml_module():
    try:
        import yaml
    except ModuleNotFoundError:
        return None
    return yaml


def parse_date_range(date_range: str) -> tuple[str, str]:
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", date_range)
    if len(dates) != 2:
        raise ValueError("Date range must include two dates in YYYY-MM-DD format.")
    return dates[0], dates[1]


def parse_extent(extent_str: str) -> list[float]:
    parts = [part for part in re.split(r"[,\s]+", extent_str.strip()) if part]
    if len(parts) != 4:
        raise ValueError("Extent must be four numbers: min_lon,min_lat,max_lon,max_lat")
    return [float(part) for part in parts]


def _read_gee_payload(config_path: str) -> dict:
    payload = Path(config_path).read_text(encoding = "utf-8")
    yaml = _load_yaml_module()
    if yaml is not None:
        return yaml.safe_load(payload) or {}
    return json.loads(payload or "{}")


def _write_gee_payload(cfg_path: Path, cfg: dict) -> None:
    yaml = _load_yaml_module()
    if yaml is not None:
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys = False), encoding = "utf-8")
        return
    cfg_path.write_text(json.dumps(cfg, indent = 2), encoding = "utf-8")


def _write_gee_config(cfg: dict, start_date: str, end_date: str, extent: list[float], gee_project: str, out_dir: str) -> str:
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    cfg["coords"] = extent
    cfg["download_dir"] = out_dir
    cfg["gee_project"] = gee_project
    cfg["start_year"] = start_dt.year
    cfg["start_month"] = start_dt.month
    cfg["start_day"] = start_dt.day
    cfg["end_year"] = end_dt.year
    cfg["end_month"] = end_dt.month
    cfg["end_day"] = end_dt.day
    cfg_path = Path(out_dir) / f"gee_config_{start_date}_{end_date}.yaml"
    _write_gee_payload(cfg_path, cfg)
    return str(cfg_path)


def write_gee_config_from_cfg(*, gee_cfg: dict, start_date: str, end_date: str, extent: list[float], gee_project: str, out_dir: str) -> str:
    return _write_gee_config(dict(gee_cfg), start_date, end_date, extent, gee_project, out_dir)


def prepare_landsat_inputs(*, date_range: str, extent: list[float], gee_project: str, out_dir: str, gee_config_template: str | None = None) -> str:
    start_date, end_date = parse_date_range(date_range)
    _safe_mkdir(out_dir)
    if gee_config_template is None:
        cfg = dict(DEFAULT_LANDSAT_GEE_CFG)
    else:
        cfg = _read_gee_payload(gee_config_template)
    cfg_path = write_gee_config_from_cfg(
        gee_cfg = cfg,
        start_date = start_date,
        end_date = end_date,
        extent = extent,
        gee_project = gee_project,
        out_dir = out_dir,
    )
    GEEDownloader(cfg_path).run()
    return cfg_path
```

```python
# pysweb/ssebop/inputs/landsat.py
#!/usr/bin/env python3
"""
Script: landsat.py
Objective: Preserve the legacy Landsat import path during the one-transition-round move to pysweb.ssebop.landsat.
Author: Yi Yu
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: Legacy imports from pysweb.ssebop.inputs.landsat.
Outputs: Re-exported canonical helpers from pysweb.ssebop.landsat.
Usage: Imported as `pysweb.ssebop.inputs.landsat`
Dependencies: pysweb.ssebop.landsat
"""
from pysweb.ssebop.landsat import *  # noqa: F401,F403
```

```python
# pysweb/met/era5land/download.py
def build_era5land_cfg(
    start_date: str,
    end_date: str,
    extent: Iterable[float],
    gee_project: str,
    out_dir: str,
) -> dict:
    ...
    return {
        "collection": "ECMWF/ERA5_LAND/DAILY_AGGR",
        "coords": coords,
        "download_dir": out_dir,
        "gee_project": gee_project,
        ...
    }


def write_era5land_config(..., gee_project: str, output_dir: str) -> Path:
    cfg = build_era5land_cfg(
        start_date = start_date,
        end_date = end_date,
        extent = extent,
        gee_project = gee_project,
        out_dir = output_dir,
    )


def download_era5land_daily(..., gee_project: str, output_dir: str, downloader_cls=None) -> Path:
    cfg_path = write_era5land_config(
        start_date = start_date,
        end_date = end_date,
        extent = extent,
        gee_project = gee_project,
        output_dir = output_dir,
    )
```

```python
# core/gee_downloader.py
    def initialize(self):
        mode = str(self.cfg["auth_mode"]).lower()
        project = str(self.cfg.get("gee_project", "")).strip()
        if mode == "browser":
            if not project:
                raise ValueError("Missing required config key: gee_project")
            try:
                ee.Initialize(project = project)
            except Exception:
                ee.Authenticate()
                ee.Initialize(project = project)
        elif mode == "service":
            email = self.cfg.get("service_account_email")
            key = self.cfg.get("service_account_key")
            if not email or not key:
                raise ValueError("For auth_mode=service, provide service_account_email and service_account_key.")
            if not project:
                raise ValueError("Missing required config key: gee_project")
            creds = ee.ServiceAccountCredentials(email, key)
            ee.Initialize(creds, project = project)
```

- [ ] **Step 4: Run the Landsat/config tests again**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/ssebop/test_landsat.py tests/workflows/test_1b_download_era5land_daily.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the canonical Landsat move and `gee_project` config plumbing**

```bash
cd /g/data/ym05/github/PySWEB && git add pysweb/ssebop/landsat.py pysweb/ssebop/inputs/landsat.py pysweb/met/era5land/download.py core/gee_downloader.py tests/ssebop/test_landsat.py tests/workflows/test_1b_download_era5land_daily.py && git commit -m "refactor: canonicalize ssebop landsat and gee project config"
```

---

### Task 3: Implement NASADEM preparation and wire it into `prepare_inputs`

**Files:**
- Modify: `pysweb/dem/nasadem.py`
- Modify: `pysweb/ssebop/api.py`
- Modify: `workflows/1_ssebop_prepare_inputs.py`
- Modify: `tests/dem/test_nasadem.py`
- Modify: `tests/ssebop/test_api_prepare_inputs.py`
- Modify: `tests/workflows/test_1_ssebop_prepare_inputs.py`

- [ ] **Step 1: Write the failing backend and orchestration tests**

```python
# tests/dem/test_nasadem.py
#!/usr/bin/env python3
"""
Script: test_nasadem.py
Objective: Verify the NASADEM backend initializes Earth Engine with the requested project and writes the prepared raster path.
Author: Yi Yu
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: Monkeypatched Earth Engine and requests helpers plus temporary file paths.
Outputs: Test assertions.
Usage: python -m pytest tests/dem/test_nasadem.py -q
Dependencies: pathlib, sys, types, pytest
"""
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysweb.dem.api import prepare_dem


def test_prepare_dem_dispatches_to_nasadem(monkeypatch, tmp_path):
    recorded = {}

    def fake_backend_prepare_dem(*, gee_project, extent, output_path):
        recorded.update(
            gee_project = gee_project,
            extent = extent,
            output_path = output_path,
        )
        Path(output_path).write_bytes(b"dem")
        return output_path

    monkeypatch.setattr("pysweb.dem.nasadem.prepare_dem", fake_backend_prepare_dem)

    output_path = prepare_dem(
        dem_source = "nasadem",
        gee_project = "demo-project",
        extent = [147.2, -35.1, 147.3, -35.0],
        output_path = str(tmp_path / "nasadem.tif"),
    )

    assert output_path == str(tmp_path / "nasadem.tif")
    assert recorded["gee_project"] == "demo-project"


def test_nasadem_backend_writes_downloaded_content(monkeypatch, tmp_path):
    import pysweb.dem.nasadem as nasadem

    calls = {
        "projects": [],
        "urls": [],
    }

    class FakeImage:
        def select(self, band_name):
            assert band_name == "elevation"
            return self

        def getDownloadURL(self, params):
            calls["urls"].append(params)
            return "https://example.invalid/nasadem.tif"

    class FakeEE:
        class Geometry:
            @staticmethod
            def Rectangle(extent, proj, geodesic):
                return {"extent": extent, "proj": proj, "geodesic": geodesic}

        @staticmethod
        def Initialize(*, project):
            calls["projects"].append(project)

        @staticmethod
        def Image(collection):
            assert collection == "NASA/NASADEM_HGT/001"
            return FakeImage()

    class FakeResponse:
        content = b"FAKE_TIF"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(nasadem, "ee", FakeEE)
    monkeypatch.setattr(nasadem.requests, "get", lambda url, timeout: FakeResponse())

    output_path = tmp_path / "nasadem.tif"
    result = nasadem.prepare_dem(
        gee_project = "demo-project",
        extent = [147.2, -35.1, 147.3, -35.0],
        output_path = str(output_path),
    )

    assert result == str(output_path)
    assert output_path.read_bytes() == b"FAKE_TIF"
    assert calls["projects"] == ["demo-project"]
```

```python
# tests/ssebop/test_api_prepare_inputs.py
def test_prepare_inputs_calls_landsat_dem_and_era5land_steps(monkeypatch, tmp_path: Path):
    recorded = []

    monkeypatch.setattr(
        "pysweb.ssebop.landsat.prepare_landsat_inputs",
        lambda **kwargs: recorded.append(("landsat", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.met.era5land.download.download_era5land_daily",
        lambda **kwargs: recorded.append(("era5land_download", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.dem.api.prepare_dem",
        lambda **kwargs: (recorded.append(("dem", kwargs)), kwargs["output_path"])[1],
    )
    monkeypatch.setattr(
        "pysweb.met.era5land.stack.stack_era5land_daily_inputs",
        lambda **kwargs: recorded.append(("era5land_stack", kwargs)),
    )

    prepare_inputs(
        date_range = "2024-01-01 to 2024-01-03",
        extent = [147.2, -35.1, 147.3, -35.0],
        met_source = "era5land",
        gee_project = "demo-project",
        landsat_dir = str(tmp_path / "landsat"),
        met_raw_dir = str(tmp_path / "raw"),
        met_stack_dir = str(tmp_path / "stack"),
        dem_dir = str(tmp_path / "dem"),
        dem_source = "nasadem",
    )

    assert recorded == [
        (
            "landsat",
            {
                "date_range": "2024-01-01 to 2024-01-03",
                "extent": [147.2, -35.1, 147.3, -35.0],
                "gee_project": "demo-project",
                "out_dir": str(tmp_path / "landsat"),
                "gee_config_template": None,
            },
        ),
        (
            "era5land_download",
            {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "extent": [147.2, -35.1, 147.3, -35.0],
                "gee_project": "demo-project",
                "output_dir": str(tmp_path / "raw"),
            },
        ),
        (
            "dem",
            {
                "dem_source": "nasadem",
                "gee_project": "demo-project",
                "extent": [147.2, -35.1, 147.3, -35.0],
                "output_path": str(tmp_path / "dem" / "nasadem.tif"),
            },
        ),
        (
            "era5land_stack",
            {
                "raw_dir": str(tmp_path / "raw"),
                "dem": str(tmp_path / "dem" / "nasadem.tif"),
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "output_dir": str(tmp_path / "stack"),
            },
        ),
    ]
```

```python
# tests/workflows/test_1_ssebop_prepare_inputs.py
def test_unified_first_step_cli_exposes_gee_project():
    ...
    args = parser.parse_args(
        [
            "--date-range", "2024-01-01 to 2024-01-03",
            "--extent", "147.2,-35.1,147.3,-35.0",
            "--met-source", "era5land",
            "--gee-project", "demo-project",
            "--out-dir", "/tmp/out",
        ]
    )

    assert args.gee_project == "demo-project"


def test_unified_first_step_cli_calls_package_api(monkeypatch, tmp_path: Path):
    ...
    module.main(
        [
            "--date-range", "2024-01-01 to 2024-01-03",
            "--extent", "147.2,-35.1,147.3,-35.0",
            "--met-source", "era5land",
            "--gee-project", "demo-project",
            "--out-dir", str(tmp_path / "out"),
        ]
    )

    assert recorded["gee_project"] == "demo-project"
    assert recorded["dem_dir"] == str(tmp_path / "out" / "dem")
    assert recorded["dem_source"] == "nasadem"
```

- [ ] **Step 2: Run the targeted tests to confirm the orchestration contract is not implemented yet**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/dem/test_nasadem.py tests/ssebop/test_api_prepare_inputs.py tests/workflows/test_1_ssebop_prepare_inputs.py -q
```

Expected: FAIL because NASADEM is still a stub and `prepare_inputs` plus the CLI still require `gee_config` and `dem`.

- [ ] **Step 3: Implement NASADEM and wire the new preparation contract**

```python
# pysweb/dem/nasadem.py
#!/usr/bin/env python3
"""
Script: nasadem.py
Objective: Download NASADEM rasters from Earth Engine for package-owned SSEBop preparation.
Author: Yi Yu
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: GEE project, target extent, and local output path.
Outputs: Prepared NASADEM GeoTIFF path.
Usage: Imported as `pysweb.dem.nasadem`
Dependencies: pathlib, requests, ee
"""
from __future__ import annotations

from pathlib import Path

import ee
import requests

NASADEM_COLLECTION = "NASA/NASADEM_HGT/001"


def _initialize_ee(gee_project: str) -> None:
    try:
        ee.Initialize(project = gee_project)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Earth Engine with project '{gee_project}'.") from exc


def prepare_dem(*, gee_project: str, extent: list[float], output_path: str) -> str:
    _initialize_ee(gee_project)
    image = ee.Image(NASADEM_COLLECTION).select("elevation")
    region = ee.Geometry.Rectangle(extent, proj = "EPSG:4326", geodesic = False)
    url = image.getDownloadURL(
        {
            "region": region,
            "scale": 30,
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
        }
    )
    response = requests.get(url, timeout = 300)
    response.raise_for_status()
    output = Path(output_path)
    output.parent.mkdir(parents = True, exist_ok = True)
    output.write_bytes(response.content)
    if output.stat().st_size == 0:
        raise RuntimeError("NASADEM download produced an empty raster.")
    return str(output)
```

```python
# pysweb/ssebop/api.py
from pysweb.dem import prepare_dem
from pysweb.ssebop import landsat


def prepare_inputs(
    *,
    date_range: str,
    extent: list[float],
    met_source: str,
    gee_project: str,
    landsat_dir: str,
    met_raw_dir: str,
    met_stack_dir: str,
    dem_dir: str,
    dem_source: str = "nasadem",
    gee_config_template: str | None = None,
) -> None:
    start_date, end_date = landsat.parse_date_range(date_range)
    prepared_dem_path = os.path.join(dem_dir, "nasadem.tif")

    if met_source != "era5land":
        raise NotImplementedError(f"Unsupported met_source: {met_source}")

    landsat.prepare_landsat_inputs(
        date_range = date_range,
        extent = extent,
        gee_project = gee_project,
        out_dir = landsat_dir,
        gee_config_template = gee_config_template,
    )
    era5land_download.download_era5land_daily(
        start_date = start_date,
        end_date = end_date,
        extent = extent,
        gee_project = gee_project,
        output_dir = met_raw_dir,
    )
    prepare_dem(
        dem_source = dem_source,
        gee_project = gee_project,
        extent = extent,
        output_path = prepared_dem_path,
    )
    era5land_stack.stack_era5land_daily_inputs(
        raw_dir = met_raw_dir,
        dem = prepared_dem_path,
        start_date = start_date,
        end_date = end_date,
        output_dir = met_stack_dir,
    )
```

```python
# workflows/1_ssebop_prepare_inputs.py
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = "Prepare Landsat, NASADEM, and meteorology inputs for SSEBop")
    parser.add_argument("--date-range", required = True, help = "Date range string like '2024-01-01 to 2024-01-03'")
    parser.add_argument("--extent", required = True, help = "min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--met-source", default = "era5land", choices = ["era5land"])
    parser.add_argument("--gee-project", required = True, help = "Earth Engine project id used for Landsat and DEM preparation")
    parser.add_argument("--out-dir", required = True, help = "Base output directory for prepared inputs")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = os.path.abspath(args.out_dir)
    prepare_inputs(
        date_range = args.date_range,
        extent = parse_extent(args.extent),
        met_source = args.met_source,
        gee_project = args.gee_project,
        landsat_dir = os.path.join(out_dir, "landsat"),
        met_raw_dir = os.path.join(out_dir, "met", args.met_source, "raw"),
        met_stack_dir = os.path.join(out_dir, "met", args.met_source, "stack"),
        dem_dir = os.path.join(out_dir, "dem"),
        dem_source = "nasadem",
    )
```

- [ ] **Step 4: Run the backend and preparation tests again**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/dem/test_nasadem.py tests/ssebop/test_api_prepare_inputs.py tests/workflows/test_1_ssebop_prepare_inputs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the NASADEM preparation flow**

```bash
cd /g/data/ym05/github/PySWEB && git add pysweb/dem/nasadem.py pysweb/ssebop/api.py workflows/1_ssebop_prepare_inputs.py tests/dem/test_nasadem.py tests/ssebop/test_api_prepare_inputs.py tests/workflows/test_1_ssebop_prepare_inputs.py && git commit -m "feat: prepare nasadem during ssebop input prep"
```

---

### Task 4: Update the run notebook and READMEs to the explicit `GEE_PROJECT` workflow

**Files:**
- Modify: `notebooks/01_run_pysweb.ipynb`
- Modify: `notebooks/README.md`
- Modify: `README.md`
- Modify: `tests/package/test_docs_and_notebooks.py`

- [ ] **Step 1: Write the failing docs/notebook assertions**

```python
# tests/package/test_docs_and_notebooks.py
def test_run_notebook_uses_explicit_gee_project_and_prepared_dem():
    markdown_text, code_text = _read_notebook_sections("notebooks/01_run_pysweb.ipynb")

    assert 'GEE_PROJECT = "your-ee-project"' in code_text
    assert "GEE_CONFIG =" not in code_text
    assert "DEM = Path(" not in code_text
    assert 'PREPARED_DEM = DEM_DIR / "nasadem.tif"' in code_text
    assert "gee_project = GEE_PROJECT" in code_text
    assert "dem = str(PREPARED_DEM)" in code_text
    assert "SM_RES is the SWB preprocess target grid" in markdown_text


def test_readmes_describe_gee_project_prepare_flow():
    readme_text = _read_text("README.md")
    notebooks_readme_text = _read_text("notebooks/README.md")

    assert "pysweb.dem" in readme_text
    assert "--gee-project" in notebooks_readme_text
    assert "--gee-config" not in notebooks_readme_text
    assert "--dem /path/to/dem.tif" not in notebooks_readme_text
    assert "prepared DEM" in notebooks_readme_text
```

- [ ] **Step 2: Run the docs test to confirm the current notebook and README text is stale**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/package/test_docs_and_notebooks.py -q
```

Expected: FAIL because the notebook still uses `GEE_CONFIG` and a user-supplied `DEM`, and the README text still documents the old preparation path.

- [ ] **Step 3: Update the notebook variables and README guidance**

```python
# notebooks/01_run_pysweb.ipynb code cell content
GEE_PROJECT = "your-ee-project"
DEM_DIR = RUN_DIR / "inputs" / "dem"
PREPARED_DEM = DEM_DIR / "nasadem.tif"

for path in [
    LANDSAT_DIR,
    MET_RAW_DIR,
    MET_STACK_DIR,
    DEM_DIR,
    SSEBOP_OUTPUT_DIR,
    SWB_PREPROCESS_DIR,
    SWB_CALIBRATION_OUTPUT.parent,
    SWB_OUTPUT_DIR,
]:
    path.mkdir(parents = True, exist_ok = True)
```

```python
# notebooks/01_run_pysweb.ipynb prepare_inputs call
if RUN_PREPARE_INPUTS:
    pysweb.ssebop.prepare_inputs(
        date_range = DATE_RANGE,
        extent = EXTENT,
        met_source = "era5land",
        gee_project = GEE_PROJECT,
        landsat_dir = str(LANDSAT_DIR),
        met_raw_dir = str(MET_RAW_DIR),
        met_stack_dir = str(MET_STACK_DIR),
        dem_dir = str(DEM_DIR),
    )
```

```python
# notebooks/01_run_pysweb.ipynb run() and SWB preprocess calls
pysweb.ssebop.run(
    date_range = DATE_RANGE,
    landsat_dir = str(LANDSAT_DIR),
    met_dir = str(MET_STACK_DIR),
    dem = str(PREPARED_DEM),
    ...
)

pysweb.swb.preprocess(
    ...
    gee_project = GEE_PROJECT,
    ...
)
```

```markdown
<!-- notebooks/README.md -->
#### 2. Call the workflow CLIs directly
Use this when you want explicit control over the date range, extent, Earth Engine project, and run directory:

```python
!python workflows/1_ssebop_prepare_inputs.py \
    --date-range "2024-01-01 to 2024-01-31" \
    --extent "147.20,-35.10,147.30,-35.00" \
    --met-source era5land \
    --gee-project your-ee-project \
    --out-dir /path/to/run_inputs
```

Step 1 now prepares a DEM under `/path/to/run_inputs/dem/nasadem.tif`, and later SSEBop run steps should point `--dem` or `dem=` at that prepared artifact.
```

```markdown
<!-- README.md -->
- `pysweb.dem`: package-owned DEM preparation backends used by SSEBop input preparation
- `pysweb.ssebop.prepare_inputs(...)`: package-first Landsat + ERA5-Land + NASADEM preparation with explicit `gee_project`
```

- [ ] **Step 4: Run the docs test again**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/package/test_docs_and_notebooks.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the notebook and README migration**

```bash
cd /g/data/ym05/github/PySWEB && git add notebooks/01_run_pysweb.ipynb notebooks/README.md README.md tests/package/test_docs_and_notebooks.py && git commit -m "docs: switch ssebop examples to gee project and prepared dem"
```

---

### Task 5: Run the focused regression sweep and close the feature cleanly

**Files:**
- Modify only if the verification run exposes regressions in:
  - `tests/package/test_pysweb_imports.py`
  - `tests/dem/test_api.py`
  - `tests/dem/test_nasadem.py`
  - `tests/ssebop/test_landsat.py`
  - `tests/ssebop/test_api_prepare_inputs.py`
  - `tests/workflows/test_1_ssebop_prepare_inputs.py`
  - `tests/workflows/test_1b_download_era5land_daily.py`
  - `tests/package/test_docs_and_notebooks.py`

- [ ] **Step 1: Run the full focused regression suite**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest \
  tests/package/test_pysweb_imports.py \
  tests/dem/test_api.py \
  tests/dem/test_nasadem.py \
  tests/ssebop/test_landsat.py \
  tests/ssebop/test_api_prepare_inputs.py \
  tests/workflows/test_1_ssebop_prepare_inputs.py \
  tests/workflows/test_1b_download_era5land_daily.py \
  tests/package/test_docs_and_notebooks.py -q
```

Expected: PASS.

- [ ] **Step 2: If the sweep exposes regressions, make the smallest possible follow-up fixes**

Use the same TDD pattern as above: add or tighten the failing assertion first, patch only the file named in the traceback, then rerun the exact failing test before rerunning the full command.

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest <failing-test-node> -q
```

Expected: PASS for the targeted fix, then PASS for the full regression command.

- [ ] **Step 3: Commit the final verified state**

```bash
cd /g/data/ym05/github/PySWEB && git status --short
cd /g/data/ym05/github/PySWEB && git add pysweb core workflows tests notebooks README.md
cd /g/data/ym05/github/PySWEB && git commit -m "feat: add explicit gee project and nasadem prepare flow"
```

- [ ] **Step 4: Record the final verification evidence in the handoff**

Include these exact points in the execution handoff:

- the final regression command that was run
- whether the repo was executed on `main` or a worktree
- the prepared DEM contract: `dem/nasadem.tif`
- the canonical Landsat import path: `pysweb.ssebop.landsat`
- the compatibility shim path: `pysweb.ssebop.inputs.landsat`
