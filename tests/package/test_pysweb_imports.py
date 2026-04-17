#!/usr/bin/env python3
"""
Script: test_pysweb_imports.py
Objective: Verify the package exposes importable subpackages and the current facade entry-point contracts.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: Package imports and direct facade calls exercised under pytest.
Outputs: Test assertions.
Usage: pytest tests/package/test_pysweb_imports.py
Dependencies: pytest
"""
from importlib import import_module
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_top_level_package_exposes_ssebop_and_swb():
    pysweb = import_module("pysweb")

    assert hasattr(pysweb, "ssebop")
    assert hasattr(pysweb, "swb")


def test_subpackages_and_swb_modules_import_cleanly():
    assert import_module("pysweb.ssebop").__name__ == "pysweb.ssebop"
    assert import_module("pysweb.swb").__name__ == "pysweb.swb"
    assert import_module("pysweb.swb.core").__name__ == "pysweb.swb.core"
    assert import_module("pysweb.swb.run").__name__ == "pysweb.swb.run"
    assert import_module("pysweb.met").__name__ == "pysweb.met"


def test_package_api_contracts():
    ssebop = import_module("pysweb.ssebop")
    swb = import_module("pysweb.swb")

    assert callable(ssebop.prepare_inputs)
    assert callable(ssebop.run)

    for func_name in ("preprocess", "calibrate"):
        func = getattr(swb, func_name)
        with pytest.raises(NotImplementedError) as exc_info:
            func()
        assert str(exc_info.value) == f"pysweb.swb.{func_name} is not wired yet"

    with pytest.raises(ValueError, match="Missing required inputs for SWB run"):
        swb.run()
