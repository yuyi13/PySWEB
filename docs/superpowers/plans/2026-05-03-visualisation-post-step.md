# Visualisation Post-Step Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional Step 6 plotting workflow that dispatches to the canonical `pysweb.visualisation` modules.

**Architecture:** Keep plotting implementation in `pysweb.visualisation`. Update both plotting CLIs to accept `argv=None`, then add `workflows/6_plot_results.py` as a thin subcommand dispatcher that forwards arguments without duplicating parser definitions. Keep top-level `visualisation/` wrappers as deprecated compatibility shims.

**Tech Stack:** Python 3.12, argparse, subprocess tests, pytest, existing PySWEB plotting modules.

---

### Task 1: Make Plotting CLIs Reusable By Wrappers

**Files:**
- Modify: `pysweb/visualisation/plot_heatmap.py`
- Modify: `pysweb/visualisation/plot_time_series.py`
- Test: `tests/visualisation/test_cli_wrappers.py`

- [ ] **Step 1: Add failing tests for argv forwarding**

Add tests that call `parse_args([...])` directly and verify validation still works without patching `sys.argv`.

```python
def test_time_series_parse_args_accepts_explicit_argv(tmp_path):
    module = import_module("pysweb.visualisation.plot_time_series")
    output = tmp_path / "ts.png"

    args = module.parse_args([
        "--ssebop-path",
        str(tmp_path / "input.nc"),
        "--output",
        str(output),
    ])

    assert args.ssebop_path == str(tmp_path / "input.nc")
    assert args.output == str(output)


def test_heatmap_parse_args_accepts_explicit_argv(tmp_path):
    module = import_module("pysweb.visualisation.plot_heatmap")
    output = tmp_path / "heatmap.png"

    args = module.parse_args([
        "--sweb-path",
        str(tmp_path / "input.nc"),
        "--domain-mean",
        "--output",
        str(output),
    ])

    assert args.sweb_path == str(tmp_path / "input.nc")
    assert args.domain_mean is True
    assert args.output == str(output)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/visualisation/test_cli_wrappers.py::test_time_series_parse_args_accepts_explicit_argv tests/visualisation/test_cli_wrappers.py::test_heatmap_parse_args_accepts_explicit_argv -q
```

Expected: failure because `parse_args()` currently does not accept an `argv` argument.

- [ ] **Step 3: Update CLI function signatures**

In `pysweb/visualisation/plot_time_series.py`:

```python
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ...
    args = parser.parse_args(argv)
    ...


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
```

In `pysweb/visualisation/plot_heatmap.py`:

```python
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ...
    args = parser.parse_args(argv)
    ...


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
```

Both modules already import `Optional` and `Sequence`.

- [ ] **Step 4: Run focused visualisation wrapper tests**

Run:

```bash
python -m pytest tests/visualisation/test_cli_wrappers.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pysweb/visualisation/plot_heatmap.py pysweb/visualisation/plot_time_series.py tests/visualisation/test_cli_wrappers.py
git commit -m "Make visualisation CLIs reusable"
```

### Task 2: Add Workflow Step 6 Dispatcher

**Files:**
- Create: `workflows/6_plot_results.py`
- Test: `tests/workflows/test_6_plot_results.py`

- [ ] **Step 1: Add failing workflow dispatcher tests**

Create `tests/workflows/test_6_plot_results.py` with subprocess help coverage and module-level forwarding coverage.

```python
#!/usr/bin/env python3
"""
Script: test_6_plot_results.py
Objective: Verify the Step 6 plotting workflow dispatches to package visualisation entrypoints.
Author: Yi Yu
Created: 2026-05-03
Last updated: 2026-05-03
Inputs: Workflow script imports and subprocess CLI help invocations.
Outputs: Test assertions.
Usage: pytest tests/workflows/test_6_plot_results.py
Dependencies: pytest, subprocess
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "workflows" / "6_plot_results.py"


def load_workflow_module():
    spec = importlib.util.spec_from_file_location("workflow_6_plot_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)

    assert spec is not None
    assert spec.loader is not None

    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, str(SCRIPT), "--help"],
        [sys.executable, str(SCRIPT), "heatmap", "--help"],
        [sys.executable, str(SCRIPT), "time-series", "--help"],
    ],
)
def test_plot_results_help_exits_cleanly(command):
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert result.stderr == ""


def test_plot_results_forwards_heatmap_args(monkeypatch):
    module = load_workflow_module()
    calls = []

    monkeypatch.setattr(module.plot_heatmap, "main", lambda argv=None: calls.append(("heatmap", argv)))

    assert module.main(["heatmap", "--domain-mean", "--output", "plot.png"]) == 0
    assert calls == [("heatmap", ["--domain-mean", "--output", "plot.png"])]


def test_plot_results_forwards_time_series_args(monkeypatch):
    module = load_workflow_module()
    calls = []

    monkeypatch.setattr(module.plot_time_series, "main", lambda argv=None: calls.append(("time-series", argv)))

    assert module.main(["time-series", "--output", "plot.png"]) == 0
    assert calls == [("time-series", ["--output", "plot.png"])]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/workflows/test_6_plot_results.py -q
```

Expected: failure because `workflows/6_plot_results.py` does not exist.

- [ ] **Step 3: Create dispatcher script**

Create `workflows/6_plot_results.py` with the standard script header and forwarding logic:

```python
#!/usr/bin/env python3
"""
Script: 6_plot_results.py
Objective: Dispatch optional post-run plotting commands to canonical PySWEB visualisation modules.
Author: Yi Yu
Created: 2026-05-03
Last updated: 2026-05-03
Inputs: CLI subcommand and plotting arguments forwarded to package visualisation modules.
Outputs: Delegated plotting side effects from the selected visualisation entrypoint.
Usage: python workflows/6_plot_results.py {heatmap,time-series} --help
Dependencies: argparse, pysweb.visualisation
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pysweb.visualisation import plot_heatmap, plot_time_series


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run optional PySWEB post-processing plots."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    heatmap = subparsers.add_parser(
        "heatmap",
        help="Forward arguments to pysweb.visualisation.plot_heatmap.",
        add_help=False,
    )
    heatmap.add_argument("plot_args", nargs=argparse.REMAINDER)

    time_series = subparsers.add_parser(
        "time-series",
        help="Forward arguments to pysweb.visualisation.plot_time_series.",
        add_help=False,
    )
    time_series.add_argument("plot_args", nargs=argparse.REMAINDER)

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    forwarded = list(args.plot_args)

    if args.command == "heatmap":
        plot_heatmap.main(forwarded)
        return 0
    if args.command == "time-series":
        plot_time_series.main(forwarded)
        return 0

    raise ValueError(f"Unsupported plotting command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run dispatcher tests**

Run:

```bash
python -m pytest tests/workflows/test_6_plot_results.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add workflows/6_plot_results.py tests/workflows/test_6_plot_results.py
git commit -m "Add plotting workflow post-step"
```

### Task 3: Update Documentation For Step 6

**Files:**
- Modify: `README.md`
- Modify: `notebooks/README.md`
- Modify: `visualisation/plot_heatmap.py`
- Modify: `visualisation/plot_time_series.py`
- Test: `tests/package/test_docs_and_notebooks.py`
- Test: `tests/visualisation/test_cli_wrappers.py`
- Test: `tests/workflows/test_6_plot_results.py`

- [ ] **Step 1: Update legacy wrapper headers**

Set `Last updated: 2026-05-03` and update the objective or usage text to call them deprecated compatibility wrappers.

- [ ] **Step 2: Update README workflow overview**

Add Step 6:

```text
6. `workflows/6_plot_results.py`: optional post-processing wrapper over `pysweb.visualisation` for heatmaps and time-series plots.
```

Add examples:

```bash
python workflows/6_plot_results.py time-series \
  --run-subdir <run_subdir> \
  --output /g/data/ym05/sweb_model/figures/<run_subdir>_timeseries.png

python workflows/6_plot_results.py heatmap \
  --run-subdir <run_subdir> \
  --domain-mean \
  --output /g/data/ym05/sweb_model/figures/<run_subdir>_heatmap_domain.png
```

- [ ] **Step 3: Update notebook README**

Describe `02_plot_heatmap.ipynb` and `03_plot_time_series.ipynb` as post-run visualisation walkthroughs corresponding to Step 6.

- [ ] **Step 4: Run docs and visualisation tests**

Run:

```bash
python -m pytest tests/package/test_docs_and_notebooks.py tests/visualisation/test_cli_wrappers.py tests/workflows/test_6_plot_results.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add README.md notebooks/README.md visualisation/plot_heatmap.py visualisation/plot_time_series.py tests/package/test_docs_and_notebooks.py tests/visualisation/test_cli_wrappers.py tests/workflows/test_6_plot_results.py
git commit -m "Document plotting workflow post-step"
```

