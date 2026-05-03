# Core Retirement Before Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove top-level `core/` as a runtime dependency for supported package paths before PySWEB packaging.

**Architecture:** Start with the install-breaking dependency by making GEE downloader implementation package-owned. Then retire already-superseded ERA5-Land, meteorology path, SSEBop, and SWB core imports into compatibility shims while preserving useful scientific comments in canonical `pysweb` modules.

**Tech Stack:** Python 3.12, setuptools package discovery, Earth Engine downloader helpers, pytest.

---

### Task 1: Make GEE Downloader Package-Owned

**Files:**
- Create: `pysweb/io/gee_downloader.py`
- Modify: `pysweb/io/gee.py`
- Modify: `core/gee_downloader.py`
- Modify: `tests/core/test_gee_downloader_config.py`
- Modify: `tests/workflows/test_1b_download_era5land_daily.py`

- [ ] **Step 1: Add failing no-core import test**

Add a test that blocks `core.gee_downloader` import and verifies `pysweb.io.gee` still exposes the downloader.

```python
def test_pysweb_gee_does_not_import_core_geedownloader(monkeypatch):
    import builtins
    import importlib
    import sys

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "core.gee_downloader":
            raise AssertionError("pysweb.io.gee must not import core.gee_downloader")
        return original_import(name, globals, locals, fromlist, level)

    sys.modules.pop("pysweb.io.gee", None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module("pysweb.io.gee")
    assert module.GEEDownloader.__name__ == "GEEDownloader"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/core/test_gee_downloader_config.py::test_pysweb_gee_does_not_import_core_geedownloader -q
```

Expected: fail because `pysweb.io.gee` currently loads `core.gee_downloader`.

- [ ] **Step 3: Move implementation into `pysweb/io/gee_downloader.py`**

Copy the current implementation body from `core/gee_downloader.py` into `pysweb/io/gee_downloader.py`, preserving comments that explain Earth Engine request limits, tiled download fallback, band metadata, raster post-processing, date handling, and retry behavior.

Ensure the new script header is:

```python
#!/usr/bin/env python3
"""
Script: gee_downloader.py
Objective: Download and post-process Google Earth Engine composites used by PySWEB preprocessing workflows, including tiled fallback for oversized requests.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-05-03
Inputs: YAML configuration file, Earth Engine authentication, date/extent/collection settings.
Outputs: Downloaded GeoTIFF composites with standardized band metadata and post-processing updates.
Usage: python -m pysweb.io.gee_downloader <config.yaml>
Dependencies: earthengine-api, requests, rasterio, numpy, pyyaml, python-dateutil
"""
```

- [ ] **Step 4: Repoint package export**

Replace `pysweb/io/gee.py` with imports from the package-owned implementation:

```python
"""Package-level exports for Google Earth Engine download helpers."""

from __future__ import annotations

from pysweb.io.gee_downloader import GEEDownloader, _safe_mkdir

__all__ = ["GEEDownloader", "_safe_mkdir"]
```

- [ ] **Step 5: Convert old core file to compatibility shim**

Replace `core/gee_downloader.py` with a deprecated wrapper that imports package-owned names and preserves CLI behavior:

```python
#!/usr/bin/env python3
"""
Script: gee_downloader.py
Objective: Provide a deprecated compatibility wrapper for the package-owned Earth Engine downloader.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-05-03
Inputs: YAML configuration file and CLI arguments forwarded to `pysweb.io.gee_downloader`.
Outputs: Delegated GeoTIFF downloads from the package entrypoint.
Usage: python core/gee_downloader.py <config.yaml>
Dependencies: pysweb.io.gee_downloader
"""
from __future__ import annotations

from pysweb.io.gee_downloader import *  # noqa: F401,F403
from pysweb.io.gee_downloader import main


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Repoint tests to package module where they test canonical behavior**

Tests that validate downloader behavior should import `pysweb.io.gee_downloader` or `pysweb.io.gee`. Keep only a small compatibility test for `core.gee_downloader`.

- [ ] **Step 7: Run focused GEE tests**

Run:

```bash
python -m pytest tests/core/test_gee_downloader_config.py tests/workflows/test_1b_download_era5land_daily.py tests/ssebop/test_landsat.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add pysweb/io/gee.py pysweb/io/gee_downloader.py core/gee_downloader.py tests/core/test_gee_downloader_config.py tests/workflows/test_1b_download_era5land_daily.py tests/ssebop/test_landsat.py
git commit -m "Move GEE downloader into package"
```

### Task 2: Retire Superseded ERA5-Land And Meteorology Core Imports

**Files:**
- Modify: `core/era5land_download_config.py`
- Modify: `core/era5land_refet.py`
- Modify: `core/era5land_stack.py`
- Modify: `core/met_input_paths.py`
- Modify: `tests/core/test_era5land_refet.py`
- Modify: `tests/workflows/test_1b_download_era5land_daily.py`
- Modify: `tests/workflows/test_2_ssebop_run_model.py`
- Modify: `tests/met/test_paths.py`

- [ ] **Step 1: Repoint tests to canonical modules**

Replace `core.era5land_refet` imports with `pysweb.met.era5land.refet`, `core.era5land_stack` imports with `pysweb.met.era5land.stack`, and `core.met_input_paths` imports with `pysweb.met.paths`.

- [ ] **Step 2: Convert old core modules to compatibility shims**

Each old module should import from its canonical package equivalent with a standard header and no duplicated implementation:

```python
from pysweb.met.era5land.refet import *  # noqa: F401,F403
```

Use the matching canonical module for each shim.

- [ ] **Step 3: Preserve useful comments**

Compare each old module against its package equivalent before replacing it. Merge comments into package code only when they explain units, ERA5-Land variable conversion, date discovery rules, path naming contracts, or FAO-56/refET assumptions.

- [ ] **Step 4: Run focused meteorology tests**

Run:

```bash
python -m pytest tests/core/test_era5land_refet.py tests/met/test_paths.py tests/workflows/test_1b_download_era5land_daily.py tests/workflows/test_2_ssebop_run_model.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add core/era5land_download_config.py core/era5land_refet.py core/era5land_stack.py core/met_input_paths.py tests/core/test_era5land_refet.py tests/met/test_paths.py tests/workflows/test_1b_download_era5land_daily.py tests/workflows/test_2_ssebop_run_model.py pysweb/met
git commit -m "Retire legacy meteorology core modules"
```

### Task 3: Retire SSEBop And SWB Core Imports

**Files:**
- Modify: `core/ssebop_au.py`
- Modify: `core/swb_model_1d.py`
- Modify: `core/soil_hydra_funs.py`
- Modify: `core/thomas_solve_tridiagonal_matrix.py`
- Modify: `tests/ssebop/test_core.py`
- Modify: `tests/swb/test_core.py`
- Modify: `README.md`

- [ ] **Step 1: Repoint SSEBop tests**

Move remaining `core.ssebop_au` imports in tests to canonical package modules:

```python
from pysweb.ssebop.core import SsebopAuConfig
from pysweb.met.silo.readers import open_silo_et_short_crop, open_silo_variable
```

If a symbol is not exported in those modules yet, add the minimal package export rather than keeping test dependence on `core`.

- [ ] **Step 2: Repoint SWB tests**

Keep SWB runtime tests on `pysweb.swb.core` and `pysweb.swb.solver`. Add compatibility-shim coverage only where needed.

- [ ] **Step 3: Compare and merge useful SWB comments**

Compare `core/swb_model_1d.py`, `core/soil_hydra_funs.py`, and `core/thomas_solve_tridiagonal_matrix.py` against `pysweb/swb/solver.py`. Preserve valid comments that explain the Jackson root beta profile, layer boundary handling, diffusion limiting, tridiagonal solve assumptions, and soil-moisture bounds.

- [ ] **Step 4: Convert old core files to shims or remove redundant standalone utility**

Use package-forwarding shims for `core/ssebop_au.py`, `core/swb_model_1d.py`, and `core/soil_hydra_funs.py` if compatibility is useful. Remove or shim `core/thomas_solve_tridiagonal_matrix.py` only after confirming no internal supported callers remain.

- [ ] **Step 5: Run focused SSEBop and SWB tests**

Run:

```bash
python -m pytest tests/ssebop/test_core.py tests/swb/test_core.py tests/ssebop/test_api_run.py tests/swb/test_api.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add core/ssebop_au.py core/swb_model_1d.py core/soil_hydra_funs.py core/thomas_solve_tridiagonal_matrix.py tests/ssebop/test_core.py tests/swb/test_core.py tests/ssebop/test_api_run.py tests/swb/test_api.py pysweb/ssebop pysweb/swb README.md
git commit -m "Retire legacy model core modules"
```

### Task 4: Verify Package No Longer Depends On Core

**Files:**
- Modify: `tests/package/test_pysweb_imports.py`
- Modify: `README.md`
- Modify: `pyproject.toml` only if package metadata smoke tests require it.

- [ ] **Step 1: Add package import smoke test**

Add a test that scans `pysweb/**/*.py` for direct `core.` imports:

```python
def test_pysweb_package_has_no_core_imports():
    root = Path(__file__).resolve().parents[2] / "pysweb"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "from core" in text or "import core" in text:
            offenders.append(path.relative_to(root.parent).as_posix())

    assert offenders == []
```

- [ ] **Step 2: Run test to verify it passes after migration**

Run:

```bash
python -m pytest tests/package/test_pysweb_imports.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Update README core status**

Update README so `core/` is described only as deprecated compatibility shims, not runtime implementation substrate.

- [ ] **Step 4: Run full suite**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/package/test_pysweb_imports.py README.md pyproject.toml
git commit -m "Verify package is independent of core"
```

