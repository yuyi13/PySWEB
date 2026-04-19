#!/usr/bin/env python3
"""
Script: test_pysweb_imports.py
Objective: Verify the package exposes importable subpackages and callable SWB facade entry points.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-19
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
    assert callable(swb.preprocess)
    assert callable(swb.calibrate)

    with pytest.raises(ValueError, match="Missing required inputs for SWB run"):
        swb.run()


@pytest.mark.parametrize(
    ("submodule_name", "attribute_name"),
    [
        ("pysweb.swb.preprocess", "preprocess"),
        ("pysweb.swb.calibrate", "calibrate"),
    ],
)
def test_submodule_first_import_keeps_package_entry_points_callable(submodule_name, attribute_name):
    for module_name in list(sys.modules):
        if module_name == "pysweb" or module_name.startswith("pysweb.swb"):
            sys.modules.pop(module_name, None)

    import_module(submodule_name)
    swb = import_module("pysweb.swb")

    assert callable(getattr(swb, attribute_name))
