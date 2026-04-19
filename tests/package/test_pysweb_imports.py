#!/usr/bin/env python3
"""
Script: test_pysweb_imports.py
Objective: Verify the package exposes importable subpackages and callable facade entry points.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-19
Inputs: Package imports and direct facade calls exercised under pytest.
Outputs: Test assertions.
Usage: pytest tests/package/test_pysweb_imports.py
Dependencies: pytest, subprocess
"""
from importlib import import_module
import inspect
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_top_level_package_exposes_ssebop_and_swb():
    pysweb = import_module("pysweb")

    assert hasattr(pysweb, "ssebop")
    assert hasattr(pysweb, "soil")
    assert hasattr(pysweb, "swb")
    assert hasattr(pysweb, "visualisation")


def test_subpackages_and_swb_modules_import_cleanly():
    assert import_module("pysweb.ssebop").__name__ == "pysweb.ssebop"
    assert import_module("pysweb.soil").__name__ == "pysweb.soil"
    assert import_module("pysweb.swb").__name__ == "pysweb.swb"
    assert import_module("pysweb.swb.core").__name__ == "pysweb.swb.core"
    assert import_module("pysweb.swb.run").__name__ == "pysweb.swb.run"
    assert import_module("pysweb.met").__name__ == "pysweb.met"
    assert import_module("pysweb.visualisation").__name__ == "pysweb.visualisation"


def test_swb_modules_import_cleanly():
    assert import_module("pysweb.swb.preprocess").__name__ == "pysweb.swb.preprocess"
    assert import_module("pysweb.swb.calibrate").__name__ == "pysweb.swb.calibrate"


def test_package_api_contracts():
    for module_name in list(sys.modules):
        if module_name == "pysweb" or module_name.startswith("pysweb.swb"):
            sys.modules.pop(module_name, None)

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


@pytest.mark.parametrize(
    ("import_statement", "module_name"),
    [
        ("import pysweb.swb.preprocess as preprocess_module", "preprocess_module"),
        ("import pysweb.swb.calibrate as calibrate_module", "calibrate_module"),
    ],
)
def test_submodule_import_alias_remains_a_module(import_statement, module_name):
    for module_name_key in list(sys.modules):
        if module_name_key == "pysweb" or module_name_key.startswith("pysweb.swb"):
            sys.modules.pop(module_name_key, None)

    namespace = {}
    exec(import_statement, namespace, namespace)

    assert inspect.ismodule(namespace[module_name])


@pytest.mark.parametrize(
    ("attribute_name",),
    [
        ("preprocess",),
        ("calibrate",),
    ],
)
def test_api_first_import_keeps_package_entry_points_as_callable_modules(attribute_name):
    for module_name in list(sys.modules):
        if module_name == "pysweb" or module_name.startswith("pysweb.swb"):
            sys.modules.pop(module_name, None)

    import_module("pysweb.swb.api")
    swb = import_module("pysweb.swb")
    value = getattr(swb, attribute_name)

    assert inspect.ismodule(value)
    assert callable(value)


@pytest.mark.parametrize(
    ("module_name", "arguments"),
    [
        ("pysweb.swb.preprocess", ["--help"]),
        (
            "pysweb.swb.calibrate",
            ["--help"],
        ),
    ],
)
def test_module_cli_execution_exits_cleanly(module_name, arguments):
    result = subprocess.run(
        [sys.executable, "-m", module_name, *arguments],
        capture_output = True,
        text = True,
        check = False,
    )

    assert result.returncode == 0
    assert result.stderr == ""
