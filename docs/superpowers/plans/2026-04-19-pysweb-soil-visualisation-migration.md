# PySWEB Soil and Visualisation Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate SWB soil sourcing in `pysweb.soil`, move plotting into `pysweb.visualisation`, and align the run notebook and README files with the current package-first workflow while keeping one-transition-round compatibility shims.

**Architecture:** Keep `pysweb.swb.preprocess` as the orchestration layer, but make it call a new `pysweb.soil` dispatcher instead of embedding OpenLandMap logic directly. Move the plotting implementations into `pysweb.visualisation`, leave the legacy `visualisation/*.py` files as thin wrappers, and add lightweight regression checks that keep the package/import boundaries and notebook/docs inventory honest.

**Tech Stack:** Python 3.12, earthengine-api, numpy, pandas, xarray, rioxarray, rasterio, matplotlib, pytest, stdlib `json`

---

All new Python files in this plan must use the standard script header from `/home/603/yy4778/.codex/docs/script_header_standard.md`.

## File Structure

### Create

- `pysweb/soil/__init__.py`
- `pysweb/soil/api.py`
- `pysweb/soil/openlandmap.py`
- `pysweb/soil/mlcons.py`
- `pysweb/soil/slga.py`
- `pysweb/soil/custom.py`
- `pysweb/visualisation/__init__.py`
- `pysweb/visualisation/plot_time_series.py`
- `pysweb/visualisation/plot_heatmap.py`
- `tests/soil/__init__.py`
- `tests/soil/test_api.py`
- `tests/soil/test_openlandmap.py`
- `tests/visualisation/__init__.py`
- `tests/visualisation/test_cli_wrappers.py`
- `tests/package/test_docs_and_notebooks.py`

### Modify

- `pysweb/__init__.py`
- `pysweb/swb/preprocess.py`
- `tests/package/test_pysweb_imports.py`
- `tests/swb/test_preprocess.py`
- `tests/workflows/test_3_sweb_preprocess_inputs.py`
- `visualisation/plot_time_series.py`
- `visualisation/plot_heatmap.py`
- `README.md`
- `notebooks/README.md`
- `notebooks/01_run_pysweb.ipynb`

### Keep As-Is

- `pysweb/swb/calibrate.py`
- `pysweb/swb/run.py`
- `workflows/3_sweb_preprocess_inputs.py`
- `workflows/4_sweb_calib_domain.py`
- `notebooks/02_plot_heatmap.ipynb`
- `notebooks/03_plot_time_series.ipynb`

---

### Task 1: Scaffold `pysweb.soil` and `pysweb.visualisation` Package Surfaces

**Files:**
- Create: `pysweb/soil/__init__.py`
- Create: `pysweb/soil/api.py`
- Create: `pysweb/soil/openlandmap.py`
- Create: `pysweb/soil/mlcons.py`
- Create: `pysweb/soil/slga.py`
- Create: `pysweb/soil/custom.py`
- Create: `pysweb/visualisation/__init__.py`
- Create: `tests/soil/__init__.py`
- Create: `tests/soil/test_api.py`
- Modify: `pysweb/__init__.py`
- Modify: `tests/package/test_pysweb_imports.py`

- [ ] **Step 1: Write the failing package-surface tests**

```python
# tests/soil/test_api.py
#!/usr/bin/env python3
"""
Script: test_api.py
Objective: Verify the soil package exposes explicit backend selection and placeholder failures.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Pytest execution against the package-level soil API.
Outputs: Test assertions.
Usage: python -m pytest tests/soil/test_api.py -q
Dependencies: pytest
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysweb.soil.api import SUPPORTED_SOIL_SOURCES, load_soil_properties


def test_supported_soil_sources_are_explicit():
    assert SUPPORTED_SOIL_SOURCES == ("openlandmap", "mlcons", "slga", "custom")


def test_unknown_soil_source_fails_early():
    with pytest.raises(ValueError, match="Unsupported soil_source 'bogus'"):
        load_soil_properties(soil_source = "bogus", args = None, grid = None)


def test_placeholder_backends_fail_explicitly():
    for soil_source in ("mlcons", "slga", "custom"):
        with pytest.raises(NotImplementedError, match = soil_source):
            load_soil_properties(soil_source = soil_source, args = None, grid = None)
```

```python
# tests/package/test_pysweb_imports.py
def test_top_level_package_exposes_soil_and_visualisation():
    pysweb = import_module("pysweb")

    assert hasattr(pysweb, "soil")
    assert hasattr(pysweb, "visualisation")


def test_new_subpackages_import_cleanly():
    assert import_module("pysweb.soil").__name__ == "pysweb.soil"
    assert import_module("pysweb.visualisation").__name__ == "pysweb.visualisation"
```

- [ ] **Step 2: Run the tests to verify the new package surfaces do not exist yet**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/package/test_pysweb_imports.py tests/soil/test_api.py -q
```

Expected: FAIL because `pysweb.soil`, `pysweb.visualisation`, and `tests/soil/test_api.py` do not exist yet.

- [ ] **Step 3: Add the package scaffolding and placeholder backend dispatcher**

```python
# pysweb/__init__.py
from importlib import import_module

__all__ = ["io", "met", "soil", "ssebop", "swb", "visualisation"]

_SUBMODULES = {name: f"pysweb.{name}" for name in __all__}
```

```python
# pysweb/soil/api.py
"""
Script: api.py
Objective: Provide the public soil-backend dispatch surface for package-owned SWB preprocessing.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Soil backend selection plus backend-specific keyword arguments.
Outputs: Standardized soil-output bundles or explicit selection errors.
Usage: Imported as `pysweb.soil.api`
Dependencies: dataclasses, numpy, xarray
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import xarray as xr

SUPPORTED_SOIL_SOURCES = ("openlandmap", "mlcons", "slga", "custom")


@dataclass(frozen = True)
class SoilOutputs:
    arrays: Dict[str, xr.DataArray]
    layer_bottoms_mm: np.ndarray


def load_soil_properties(*, soil_source: str, args, grid, **kwargs) -> SoilOutputs:
    if soil_source == "openlandmap":
        from pysweb.soil.openlandmap import load_soil_properties as loader

        return loader(args = args, grid = grid, **kwargs)

    if soil_source in ("mlcons", "slga", "custom"):
        raise NotImplementedError(f"Soil backend '{soil_source}' is declared but not implemented yet.")

    supported = ", ".join(SUPPORTED_SOIL_SOURCES)
    raise ValueError(f"Unsupported soil_source '{soil_source}'. Supported values: {supported}")
```

```python
# pysweb/soil/__init__.py
"""
Script: __init__.py
Objective: Expose soil backend modules and the package-level soil dispatcher.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Package attribute access.
Outputs: Lazily imported backend modules and dispatcher attributes.
Usage: Imported as `pysweb.soil`
Dependencies: importlib
"""
from importlib import import_module

__all__ = ["api", "custom", "mlcons", "openlandmap", "slga", "load_soil_properties"]

_SUBMODULES = {
    "api": "pysweb.soil.api",
    "custom": "pysweb.soil.custom",
    "mlcons": "pysweb.soil.mlcons",
    "openlandmap": "pysweb.soil.openlandmap",
    "slga": "pysweb.soil.slga",
}


def __getattr__(name):
    if name == "load_soil_properties":
        value = import_module("pysweb.soil.api").load_soil_properties
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'pysweb.soil' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
```

```python
# pysweb/soil/openlandmap.py
"""
Script: openlandmap.py
Objective: Hold the OpenLandMap soil backend implementation for SWB preprocessing.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: SWB preprocess arguments, target grid, and Earth Engine access.
Outputs: Standardized SWB soil-property arrays.
Usage: Imported as `pysweb.soil.openlandmap`
Dependencies: pysweb.soil.api
"""
from __future__ import annotations


def load_soil_properties(*, args, grid, **kwargs):
    raise NotImplementedError("Soil backend 'openlandmap' has not been moved out of pysweb.swb.preprocess yet.")
```

```python
# pysweb/soil/mlcons.py
# pysweb/soil/slga.py
# pysweb/soil/custom.py
"""
Script: mlcons.py
Objective: Reserve the future MLConstraints soil backend namespace.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Future SWB preprocess arguments and grid definitions.
Outputs: Explicit not-implemented errors until the backend exists.
Usage: Imported as `pysweb.soil.mlcons`
Dependencies: none
"""
from __future__ import annotations


def load_soil_properties(*, args, grid, **kwargs):
    raise NotImplementedError("Soil backend 'mlcons' is declared but not implemented yet.")
```

```python
# pysweb/visualisation/__init__.py
"""
Script: __init__.py
Objective: Expose package-backed visualisation modules.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Package attribute access.
Outputs: Lazily imported plotting modules.
Usage: Imported as `pysweb.visualisation`
Dependencies: importlib
"""
from importlib import import_module

__all__ = ["plot_heatmap", "plot_time_series"]

_SUBMODULES = {name: f"pysweb.visualisation.{name}" for name in __all__}


def __getattr__(name):
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'pysweb.visualisation' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
```

- [ ] **Step 4: Run the tests to verify the new package surfaces exist and fail correctly**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/package/test_pysweb_imports.py tests/soil/test_api.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /g/data/ym05/github/PySWEB && git add pysweb/__init__.py pysweb/soil pysweb/visualisation/__init__.py tests/package/test_pysweb_imports.py tests/soil && git commit -m "feat: scaffold soil and visualisation packages"
```

---

### Task 2: Move OpenLandMap Soil Logic Behind the `pysweb.soil` Dispatcher

**Files:**
- Modify: `pysweb/soil/api.py`
- Modify: `pysweb/soil/openlandmap.py`
- Modify: `pysweb/swb/preprocess.py`
- Create: `tests/soil/test_openlandmap.py`
- Modify: `tests/soil/test_api.py`
- Modify: `tests/swb/test_preprocess.py`
- Modify: `tests/workflows/test_3_sweb_preprocess_inputs.py`

- [ ] **Step 1: Write the failing backend and orchestration tests**

```python
# tests/soil/test_api.py
from pysweb.soil.api import SoilOutputs, load_soil_properties


def test_openlandmap_dispatch_calls_backend(monkeypatch):
    recorded = {}

    def fake_loader(*, args, grid, **kwargs):
        recorded["args"] = args
        recorded["grid"] = grid
        return SoilOutputs(arrays = {}, layer_bottoms_mm = __import__("numpy").array([50.0]))

    monkeypatch.setattr("pysweb.soil.openlandmap.load_soil_properties", fake_loader)

    result = load_soil_properties(soil_source = "openlandmap", args = "ARGS", grid = "GRID")

    assert isinstance(result, SoilOutputs)
    assert recorded == {"args": "ARGS", "grid": "GRID"}
```

```python
# tests/soil/test_openlandmap.py
#!/usr/bin/env python3
"""
Script: test_openlandmap.py
Objective: Verify the OpenLandMap backend owns the current depth mapping and soil-property derivation helpers.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Pytest fixtures, fake Earth Engine objects, and in-memory xarray arrays.
Outputs: Test assertions.
Usage: python -m pytest tests/soil/test_openlandmap.py -q
Dependencies: numpy, pandas, pytest, xarray
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysweb.soil.api import SoilOutputs
from pysweb.soil.openlandmap import (
    OPENLANDMAP_EXPORT_SCALE_M,
    OPENLANDMAP_LAYER_SPECS,
    _build_layer_bottoms_mm,
    load_soil_properties,
)


def test_openlandmap_depth_mapping_matches_swb_profile():
    assert OPENLANDMAP_LAYER_SPECS == [
        ("b0", 50.0),
        ("b10", 150.0),
        ("b30", 300.0),
        ("b60", 600.0),
        ("b100", 1000.0),
    ]
    np.testing.assert_allclose(_build_layer_bottoms_mm(), np.array([50.0, 150.0, 300.0, 600.0, 1000.0]))
    assert OPENLANDMAP_EXPORT_SCALE_M == 250.0


def test_openlandmap_dispatch_returns_soil_outputs(monkeypatch):
    monkeypatch.setattr(
        "pysweb.soil.openlandmap._load_openlandmap_predictors",
        lambda extent, gee_project: {"clay": "clay", "sand": "sand", "soc": "soc"},
    )
    monkeypatch.setattr(
        "pysweb.soil.openlandmap.process_soil_properties_from_openlandmap",
        lambda args, grid, soil_predictors, reproject_to_template: {
            "porosity": "porosity",
            "wilting_point": "wilting_point",
            "available_water_capacity": "awc",
            "b_coefficient": "bcoef",
            "conductivity_sat": "ksat",
        },
    )

    result = load_soil_properties(args = type("Args", (), {"extent": [0, 0, 1, 1], "gee_project": "proj"})(), grid = "GRID")

    assert isinstance(result, SoilOutputs)
    assert set(result.arrays) == {
        "porosity",
        "wilting_point",
        "available_water_capacity",
        "b_coefficient",
        "conductivity_sat",
    }
```

```python
# tests/swb/test_preprocess.py
from pysweb.soil.api import SoilOutputs


def test_preprocess_inputs_delegates_soil_loading_to_soil_api(monkeypatch, tmp_path):
    recorded = {}

    monkeypatch.setattr(preprocess_module, "process_precipitation", lambda *args, **kwargs: _forcing_array(np.ones((2, 1, 1)), name = "rain", dates = ["2024-01-01", "2024-01-02"]))
    monkeypatch.setattr(preprocess_module, "process_et", lambda *args, **kwargs: {"et": _forcing_array(np.ones((2, 1, 1)), name = "et", dates = ["2024-01-01", "2024-01-02"])})
    monkeypatch.setattr(preprocess_module, "_load_reference_ssm", lambda **kwargs: _forcing_array(np.ones((2, 1, 1)), name = "reference_ssm", dates = ["2024-01-01", "2024-01-02"]))
    monkeypatch.setattr(preprocess_module, "_reproject_to_template", lambda da, grid, **kwargs: da)

    def fake_soil_loader(*, soil_source, args, grid, **kwargs):
        recorded["soil_source"] = soil_source
        return SoilOutputs(
            arrays = {"porosity": _soil_array("porosity", np.full((5, 1, 1), 0.4))},
            layer_bottoms_mm = np.array([50.0, 150.0, 300.0, 600.0, 1000.0]),
        )

    monkeypatch.setattr(preprocess_module.soil_api, "load_soil_properties", fake_soil_loader)

    preprocess_inputs(
        date_range = ["2024-01-01", "2024-01-02"],
        extent = [148.0, -35.1, 148.1, -35.0],
        sm_res = 0.1,
        output_dir = str(tmp_path),
        soil_source = "openlandmap",
        skip_reference_ssm = False,
    )

    assert recorded["soil_source"] == "openlandmap"
    assert (tmp_path / "soil_porosity.nc").exists()
```

```python
# tests/workflows/test_3_sweb_preprocess_inputs.py
def test_workflow_help_exposes_soil_source(monkeypatch, capsys):
    workflow_module = _load_workflow_module()
    monkeypatch.setattr(sys, "argv", ["3_sweb_preprocess_inputs.py", "--help"])

    with pytest.raises(SystemExit) as exc:
        workflow_module.main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--soil-source" in captured.out
```

- [ ] **Step 2: Run the targeted tests to verify the current OpenLandMap code is still embedded in `pysweb.swb.preprocess`**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/soil/test_api.py tests/soil/test_openlandmap.py tests/swb/test_preprocess.py tests/workflows/test_3_sweb_preprocess_inputs.py -q
```

Expected: FAIL because `pysweb.soil.openlandmap` is still a stub, `pysweb.swb.preprocess` does not accept `soil_source`, and the orchestration test cannot patch `preprocess_module.soil_api`.

- [ ] **Step 3: Move the OpenLandMap helpers into `pysweb.soil.openlandmap` and call them through the dispatcher**

```python
# pysweb/soil/openlandmap.py
"""
Script: openlandmap.py
Objective: Load OpenLandMap predictors and derive SWB soil-property layers on the target grid.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: SWB preprocess arguments, target grid, and Earth Engine access.
Outputs: Standardized SWB soil-property arrays.
Usage: Imported as `pysweb.soil.openlandmap`
Dependencies: earthengine-api, numpy, xarray, rasterio
"""
from __future__ import annotations

from typing import Dict

import ee
import numpy as np
import xarray as xr
from rasterio.enums import Resampling

from pysweb.soil.api import SoilOutputs

OPENLANDMAP_LAYER_SPECS = [
    ("b0", 50.0),
    ("b10", 150.0),
    ("b30", 300.0),
    ("b60", 600.0),
    ("b100", 1000.0),
]
OPENLANDMAP_EXPORT_SCALE_M = 250.0
OPENLANDMAP_DATASETS = {
    "clay": "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02",
    "sand": "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02",
    "soc": "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02",
}
OPENLANDMAP_SOC_SCALE = 5.0


def _build_layer_bottoms_mm() -> np.ndarray:
    return np.array([bottom_mm for _, bottom_mm in OPENLANDMAP_LAYER_SPECS], dtype = float)


# Move these existing functions from pysweb/swb/preprocess.py into this file
# without renaming or changing their numeric behavior:
# - _select_openlandmap_bands
# - _load_openlandmap_predictors
# - process_soil_properties_from_openlandmap
#
# Change process_soil_properties_from_openlandmap so it accepts
# reproject_to_template as a keyword-only callback instead of importing
# that helper from pysweb.swb.preprocess.


def load_soil_properties(*, args, grid, reproject_to_template, **kwargs) -> SoilOutputs:
    soil_predictors = _load_openlandmap_predictors(tuple(args.extent), args.gee_project)
    soil_arrays = process_soil_properties_from_openlandmap(
        args,
        grid,
        soil_predictors,
        reproject_to_template = reproject_to_template,
    )
    return SoilOutputs(arrays = soil_arrays, layer_bottoms_mm = _build_layer_bottoms_mm())
```

```python
# pysweb/soil/api.py
def load_soil_properties(*, soil_source: str, args, grid, **kwargs) -> SoilOutputs:
    if soil_source == "openlandmap":
        from pysweb.soil.openlandmap import load_soil_properties as loader

        return loader(args = args, grid = grid, **kwargs)
```

```python
# pysweb/swb/preprocess.py
import pysweb.soil.api as soil_api


# Add this parser argument:
parser.add_argument(
    "--soil-source",
    default = "openlandmap",
    help = "Soil backend for SWB preprocessing. Supported: openlandmap, mlcons, slga, custom.",
)


# Replace the embedded OpenLandMap calls inside preprocess_inputs with:
soil_outputs = soil_api.load_soil_properties(
    soil_source = args.soil_source,
    args = args,
    grid = grid,
    reproject_to_template = _reproject_to_template,
)
soil_arrays = soil_outputs.arrays


# Keep the final write path, but pass the dispatched arrays:
return _write_preprocess_outputs(output_dir, outputs, soil_arrays)
```

```python
# tests/swb/test_preprocess.py
# Remove direct imports of OpenLandMap-specific helpers from this file.
# Keep SWB preprocess tests focused on orchestration, output writing, and
# reference_ssm behavior; move the backend-specific tests to tests/soil/test_openlandmap.py.
```

- [ ] **Step 4: Run the targeted tests to confirm the soil package owns the OpenLandMap path**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/soil/test_api.py tests/soil/test_openlandmap.py tests/swb/test_preprocess.py tests/workflows/test_3_sweb_preprocess_inputs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /g/data/ym05/github/PySWEB && git add pysweb/soil/api.py pysweb/soil/openlandmap.py pysweb/swb/preprocess.py tests/soil/test_api.py tests/soil/test_openlandmap.py tests/swb/test_preprocess.py tests/workflows/test_3_sweb_preprocess_inputs.py && git commit -m "refactor: move openlandmap soil loading into pysweb.soil"
```

---

### Task 3: Move Plotting Into `pysweb.visualisation` and Leave Thin Legacy Wrappers

**Files:**
- Modify: `pysweb/visualisation/__init__.py`
- Create: `pysweb/visualisation/plot_time_series.py`
- Create: `pysweb/visualisation/plot_heatmap.py`
- Modify: `visualisation/plot_time_series.py`
- Modify: `visualisation/plot_heatmap.py`
- Create: `tests/visualisation/__init__.py`
- Create: `tests/visualisation/test_cli_wrappers.py`
- Modify: `tests/package/test_pysweb_imports.py`

- [ ] **Step 1: Write the failing visualisation import and wrapper tests**

```python
# tests/visualisation/test_cli_wrappers.py
#!/usr/bin/env python3
"""
Script: test_cli_wrappers.py
Objective: Verify package-backed plotting modules exist and legacy visualisation scripts stay as thin shims.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Module imports, wrapper file contents, and subprocess help invocations.
Outputs: Test assertions.
Usage: python -m pytest tests/visualisation/test_cli_wrappers.py -q
Dependencies: pathlib, subprocess, sys, pytest
"""
from importlib import import_module
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_package_visualisation_modules_import_cleanly():
    assert import_module("pysweb.visualisation.plot_time_series").__name__ == "pysweb.visualisation.plot_time_series"
    assert import_module("pysweb.visualisation.plot_heatmap").__name__ == "pysweb.visualisation.plot_heatmap"


def test_legacy_time_series_wrapper_targets_package_module():
    text = (ROOT / "visualisation" / "plot_time_series.py").read_text(encoding = "utf-8")
    assert "from pysweb.visualisation.plot_time_series import main" in text


def test_legacy_heatmap_wrapper_targets_package_module():
    text = (ROOT / "visualisation" / "plot_heatmap.py").read_text(encoding = "utf-8")
    assert "from pysweb.visualisation.plot_heatmap import main" in text


@pytest.mark.parametrize(
    ("command", "needle"),
    [
        ([sys.executable, "-m", "pysweb.visualisation.plot_time_series", "--help"], "Plot time series from SSEBop and SWEB NetCDF outputs."),
        ([sys.executable, "-m", "pysweb.visualisation.plot_heatmap", "--help"], "Plot SWEB layer heatmaps with optional SSEBop panel"),
        ([sys.executable, "visualisation/plot_time_series.py", "--help"], "Plot time series from SSEBop and SWEB NetCDF outputs."),
        ([sys.executable, "visualisation/plot_heatmap.py", "--help"], "Plot SWEB layer heatmaps with optional SSEBop panel"),
    ],
)
def test_visualisation_entrypoints_exit_cleanly(command, needle):
    result = subprocess.run(command, cwd = ROOT, capture_output = True, text = True, check = False)

    assert result.returncode == 0, result.stderr
    assert needle in result.stdout
```

```python
# tests/package/test_pysweb_imports.py
def test_visualisation_modules_import_cleanly():
    assert import_module("pysweb.visualisation.plot_time_series").__name__ == "pysweb.visualisation.plot_time_series"
    assert import_module("pysweb.visualisation.plot_heatmap").__name__ == "pysweb.visualisation.plot_heatmap"
```

- [ ] **Step 2: Run the visualisation tests to verify the package modules do not exist yet**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/package/test_pysweb_imports.py tests/visualisation/test_cli_wrappers.py -q
```

Expected: FAIL because `pysweb.visualisation.plot_time_series` and `pysweb.visualisation.plot_heatmap` do not exist yet, and the legacy scripts still contain full implementations instead of wrapper imports.

- [ ] **Step 3: Copy the plotting implementations into `pysweb.visualisation` and reduce the old scripts to wrappers**

```bash
cd /g/data/ym05/github/PySWEB && cp visualisation/plot_time_series.py pysweb/visualisation/plot_time_series.py && cp visualisation/plot_heatmap.py pysweb/visualisation/plot_heatmap.py
```

```python
# pysweb/visualisation/plot_time_series.py
# After the copy, keep the implementation unchanged except for the header and
# one added font line directly after importing pyplot:
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
```

```python
# pysweb/visualisation/plot_heatmap.py
# After the copy, change the sibling import to the package path and add the
# same font default directly after importing pyplot:
from pysweb.visualisation.plot_time_series import (
    DEFAULT_SSEBOP_ROOT,
    DEFAULT_SWEB_ROOT,
    SSEBOP_FILE_PATTERN,
    SWEB_FILE_PATTERN,
    _format_var_label,
    _parse_timestamp,
    _resolve_product_path,
    extract_product_series,
)

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
```

```python
# visualisation/plot_time_series.py
#!/usr/bin/env python3
"""
Script: plot_time_series.py
Objective: Preserve the legacy CLI entrypoint while delegating to pysweb.visualisation.plot_time_series.
Author: Yi Yu
Created: 2026-02-20
Last updated: 2026-04-19
Inputs: CLI arguments for time-series plotting.
Outputs: Delegated plotting outputs.
Usage: python visualisation/plot_time_series.py --help
Dependencies: pysweb
"""
from pysweb.visualisation.plot_time_series import main


if __name__ == "__main__":
    main()
```

```python
# visualisation/plot_heatmap.py
#!/usr/bin/env python3
"""
Script: plot_heatmap.py
Objective: Preserve the legacy CLI entrypoint while delegating to pysweb.visualisation.plot_heatmap.
Author: Yi Yu
Created: 2026-02-20
Last updated: 2026-04-19
Inputs: CLI arguments for heatmap plotting.
Outputs: Delegated plotting outputs.
Usage: python visualisation/plot_heatmap.py --help
Dependencies: pysweb
"""
from pysweb.visualisation.plot_heatmap import main


if __name__ == "__main__":
    main()
```

```python
# pysweb/visualisation/__init__.py
__all__ = ["plot_heatmap", "plot_time_series"]
```

- [ ] **Step 4: Run the visualisation tests to confirm the package modules are canonical and wrappers stay thin**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/package/test_pysweb_imports.py tests/visualisation/test_cli_wrappers.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /g/data/ym05/github/PySWEB && git add pysweb/visualisation visualisation/plot_time_series.py visualisation/plot_heatmap.py tests/package/test_pysweb_imports.py tests/visualisation && git commit -m "refactor: migrate plotting into pysweb.visualisation"
```

---

### Task 4: Update the Run Notebook and README Files to Match the New Canonical Paths

**Files:**
- Create: `tests/package/test_docs_and_notebooks.py`
- Modify: `README.md`
- Modify: `notebooks/README.md`
- Modify: `notebooks/01_run_pysweb.ipynb`

- [ ] **Step 1: Write the failing docs and notebook regression tests**

```python
# tests/package/test_docs_and_notebooks.py
#!/usr/bin/env python3
"""
Script: test_docs_and_notebooks.py
Objective: Verify the run notebook and README files describe the current package-backed workflow and notebook inventory.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Repository documentation files and notebook JSON.
Outputs: Test assertions.
Usage: python -m pytest tests/package/test_docs_and_notebooks.py -q
Dependencies: json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding = "utf-8"))


def test_run_notebook_uses_package_backed_swb_workflow():
    notebook = _load_notebook(ROOT / "notebooks" / "01_run_pysweb.ipynb")
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "pysweb.swb.preprocess(" in code
    assert "pysweb.swb.calibrate(" in code
    assert "pysweb.swb.run(" in code
    assert "workflow scripts" not in markdown


def test_readmes_list_current_notebooks_and_package_paths():
    readme = (ROOT / "README.md").read_text(encoding = "utf-8")
    notebook_readme = (ROOT / "notebooks" / "README.md").read_text(encoding = "utf-8")

    for text in (readme, notebook_readme):
        assert "01_run_pysweb.ipynb" in text
        assert "02_plot_heatmap.ipynb" in text
        assert "03_plot_time_series.ipynb" in text

    assert "pysweb.soil" in readme
    assert "pysweb.visualisation" in readme
    assert "pysweb.visualisation" in notebook_readme
```

- [ ] **Step 2: Run the docs test to capture the current stale notebook and README state**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/package/test_docs_and_notebooks.py -q
```

Expected: FAIL because `01_run_pysweb.ipynb` still says SWB preprocess/calibration are workflow-driven and the README files still list the old notebook names and `visualisation/` path as canonical.

- [ ] **Step 3: Update the notebook narrative and both README files**

```text
# notebooks/01_run_pysweb.ipynb
# Replace the opening markdown title cell with:
#
# PySWEB example: run SSEBop and SWB from a notebook with `import pysweb`
#
# This notebook shows the package-backed workflow for preparing Landsat + ERA5-Land inputs,
# running SSEBop directly from Python, and then running the SWB preprocess, calibration,
# and final model stages through the `pysweb` package.
```

```python
# notebooks/01_run_pysweb.ipynb
# Add or replace the SWB configuration cell with these source variables:
SWB_RAIN_FILE = Path("/path/to/rain_daily.nc")
SWB_ET_FILE = Path("/path/to/et_inputs.nc")
SWB_PARAM_CSV = SWB_PREPROCESS_DIR / "calibration_params.csv"

RUN_SWB_PREPROCESS = False
RUN_SWB_CALIBRATE = False
RUN_SWB = False
```

```python
# notebooks/01_run_pysweb.ipynb
# Replace the old "Optional SWB final run" section with package-backed SWB cells:
if RUN_SWB_PREPROCESS:
    pysweb.swb.preprocess(
        date_range = [START_DATE, END_DATE],
        extent = EXTENT,
        sm_res = 0.00025,
        rain_file = str(SWB_RAIN_FILE),
        et_file = str(SWB_ET_FILE),
        output_dir = str(SWB_PREPROCESS_DIR),
        soil_source = "openlandmap",
        gee_project = "yiyu-research",
    )
else:
    print("Set RUN_SWB_PREPROCESS = True after pointing SWB_RAIN_FILE and SWB_ET_FILE at prepared inputs.")


reference_ssm_nc = SWB_PREPROCESS_DIR / f"reference_ssm_daily_{START_TAG}_{END_TAG}.nc"

if RUN_SWB_CALIBRATE:
    pysweb.swb.calibrate(
        effective_precip = str(SWB_PREPROCESS_DIR / f"effective_precip_daily_{START_TAG}_{END_TAG}.nc"),
        et = str(SWB_PREPROCESS_DIR / f"et_daily_{START_TAG}_{END_TAG}.nc"),
        t = str(SWB_PREPROCESS_DIR / f"t_daily_{START_TAG}_{END_TAG}.nc"),
        soil_dir = str(SWB_PREPROCESS_DIR),
        reference_ssm = str(reference_ssm_nc),
        output = str(SWB_PARAM_CSV),
    )
else:
    print("Set RUN_SWB_CALIBRATE = True after SWB preprocess outputs exist.")


if RUN_SWB:
    pysweb.swb.run(
        precip = str(SWB_PREPROCESS_DIR / f"rain_daily_{START_TAG}_{END_TAG}.nc"),
        effective_precip = str(SWB_PREPROCESS_DIR / f"effective_precip_daily_{START_TAG}_{END_TAG}.nc"),
        et = str(SWB_PREPROCESS_DIR / f"et_daily_{START_TAG}_{END_TAG}.nc"),
        t = str(SWB_PREPROCESS_DIR / f"t_daily_{START_TAG}_{END_TAG}.nc"),
        soil_dir = str(SWB_PREPROCESS_DIR),
        param_grid = str(SWB_PARAM_CSV),
        output_dir = str(SWB_OUTPUT_DIR),
        start_date = START_DATE,
        end_date = END_DATE,
        workers = 4,
    )
else:
    print("Set RUN_SWB = True after preprocess and calibration outputs are ready.")
```

```text
# README.md
# Update the repository tree and usage sections so they mention:
# - pysweb/soil/ as the canonical soil-source package
# - pysweb/visualisation/ as the canonical plotting package
# - visualisation/ as a legacy wrapper directory
# - the actual notebook file names:
#   01_run_pysweb.ipynb
#   02_plot_heatmap.ipynb
#   03_plot_time_series.ipynb
#
# Update any example plotting commands and prose to prefer module entrypoints such as:
# - python -m pysweb.visualisation.plot_time_series --help
# - python -m pysweb.visualisation.plot_heatmap --help
# while noting that the legacy wrapper paths still work during the transition.
```

```text
# notebooks/README.md
# Update the notebook inventory so it lists:
# - 01_run_pysweb.ipynb
# - 02_plot_heatmap.ipynb
# - 03_plot_time_series.ipynb
#
# Update the run notebook description to say that 01_run_pysweb.ipynb demonstrates
# package-backed SSEBop plus SWB preprocess/calibrate/run usage.
# Update plotting references to mention pysweb.visualisation as the canonical path.
```

- [ ] **Step 4: Run the docs and notebook tests to confirm the content matches the new package-first story**

Run:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest tests/package/test_docs_and_notebooks.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /g/data/ym05/github/PySWEB && git add README.md notebooks/README.md notebooks/01_run_pysweb.ipynb tests/package/test_docs_and_notebooks.py && git commit -m "docs: align notebook and readmes with pysweb packages"
```

---

## Final Verification

After all four tasks are complete, run the combined regression slice for the touched areas:

```bash
cd /g/data/ym05/github/PySWEB && python -m pytest \
  tests/package/test_pysweb_imports.py \
  tests/package/test_docs_and_notebooks.py \
  tests/soil/test_api.py \
  tests/soil/test_openlandmap.py \
  tests/swb/test_preprocess.py \
  tests/visualisation/test_cli_wrappers.py \
  tests/workflows/test_3_sweb_preprocess_inputs.py \
  -q
```

Expected: PASS.
