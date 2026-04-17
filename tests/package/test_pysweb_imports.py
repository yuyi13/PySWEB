from importlib import import_module


def test_top_level_package_exposes_ssebop_and_swb():
    pysweb = import_module("pysweb")

    assert hasattr(pysweb, "ssebop")
    assert hasattr(pysweb, "swb")


def test_subpackages_import_cleanly():
    assert import_module("pysweb.ssebop").__name__ == "pysweb.ssebop"
    assert import_module("pysweb.swb").__name__ == "pysweb.swb"
    assert import_module("pysweb.met").__name__ == "pysweb.met"
