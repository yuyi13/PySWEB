# Task 4 SWB Calibration Package Move Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move SWB domain calibration into `pysweb.swb.calibrate`, rename the public calibration interface to neutral reference SSM terms, and reduce Workflow 4 to a thin wrapper.

**Architecture:** Keep the current workflow implementation largely intact, but move it into the package module so `pysweb.swb.calibrate` owns the parser, data loading, optimization, and CSV writing. Switch the solver import to `pysweb.swb.solver.soil_water_balance_1d`, update public argument names from `smap_*` to `reference_*`, and leave the workflow script as a wrapper that parses args and forwards them into the package implementation.

**Tech Stack:** Python, argparse, numpy, pandas, xarray, scipy, pytest

---

### Task 1: Add parser and workflow regression tests

**Files:**
- Create: `tests/swb/test_calibrate.py`
- Create: `tests/workflows/test_4_sweb_calib_domain.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path
import sys

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
from importlib import util
from pathlib import Path
import sys

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

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/swb/test_calibrate.py tests/workflows/test_4_sweb_calib_domain.py -q`
Expected: FAIL because `pysweb.swb.calibrate.build_parser()` does not yet expose the full calibration CLI and Workflow 4 still uses `smap_*` names and owns the implementation.

- [ ] **Step 3: Write minimal implementation**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Domain-wide calibration using reference SSM.")
    parser.add_argument("--effective-precip", required=True)
    parser.add_argument("--effective-precip-var", default="effective_precipitation")
    parser.add_argument("--et", required=True)
    parser.add_argument("--et-var", default="et")
    parser.add_argument("--t", required=True)
    parser.add_argument("--t-var", default="t")
    parser.add_argument("--soil-dir", required=True)
    parser.add_argument("--reference-ssm", required=True)
    parser.add_argument("--reference-var", default="reference_ssm")
    parser.add_argument("--output", required=True)
    return parser
```

```python
from pysweb.swb.calibrate import build_parser, calibrate_domain


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    calibrate_domain(**vars(args))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/swb/test_calibrate.py tests/workflows/test_4_sweb_calib_domain.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/swb/test_calibrate.py tests/workflows/test_4_sweb_calib_domain.py pysweb/swb/calibrate.py workflows/4_sweb_calib_domain.py
git commit -m "refactor: move swb calibration into package"
```

### Task 2: Move calibration implementation into the package module

**Files:**
- Modify: `pysweb/swb/calibrate.py`
- Modify: `workflows/4_sweb_calib_domain.py`

- [ ] **Step 1: Write the failing integration-oriented regression**

Use the same tests from Task 1 as the public contract while replacing the stub package implementation and workflow-owned calibration body.

- [ ] **Step 2: Run test to verify it fails if the implementation is absent**

Run: `python -m pytest tests/swb/test_calibrate.py tests/workflows/test_4_sweb_calib_domain.py -q`
Expected: FAIL if the moved implementation is incomplete.

- [ ] **Step 3: Write minimal implementation**

```python
from pysweb.swb.solver import soil_water_balance_1d


def calibrate_domain(**kwargs):
    args = argparse.Namespace(**kwargs)
    # Move the existing helper functions and main-body calibration logic
    # from workflows/4_sweb_calib_domain.py into this module with neutral
    # `reference_ssm` and `reference_var` naming.
    # Preserve file-based NetCDF loading, alignment, optimization, and CSV output.
```

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/swb/test_calibrate.py tests/workflows/test_4_sweb_calib_domain.py tests/swb/test_api.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pysweb/swb/calibrate.py workflows/4_sweb_calib_domain.py tests/swb/test_api.py
git commit -m "refactor: move swb calibration into package"
```
