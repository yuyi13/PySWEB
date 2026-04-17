from importlib import import_module
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_top_level_package_exposes_ssebop_and_swb():
    pysweb = import_module("pysweb")

    assert hasattr(pysweb, "ssebop")
    assert hasattr(pysweb, "swb")


def test_subpackages_import_cleanly():
    assert import_module("pysweb.ssebop").__name__ == "pysweb.ssebop"
    assert import_module("pysweb.swb").__name__ == "pysweb.swb"
    assert import_module("pysweb.met").__name__ == "pysweb.met"


def test_placeholder_api_contract():
    ssebop = import_module("pysweb.ssebop")
    swb = import_module("pysweb.swb")

    for func_name in ("prepare_inputs", "run"):
        func = getattr(ssebop, func_name)
        try:
            func()
        except NotImplementedError as exc:
            assert str(exc) == f"pysweb.ssebop.{func_name} is not wired yet"
        else:
            raise AssertionError("Expected NotImplementedError")

    for func_name in ("preprocess", "calibrate", "run"):
        func = getattr(swb, func_name)
        try:
            func()
        except NotImplementedError as exc:
            assert str(exc) == f"pysweb.swb.{func_name} is not wired yet"
        else:
            raise AssertionError("Expected NotImplementedError")
