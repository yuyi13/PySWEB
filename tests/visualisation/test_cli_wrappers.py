#!/usr/bin/env python3
"""
Script: test_cli_wrappers.py
Objective: Verify the package visualisation modules import and expose help cleanly.
Author: Yi Yu (with assistance from Codex)
Created: 2026-04-19
Last updated: 2026-10-26
Inputs: Module imports and subprocess CLI help invocations.
Outputs: Test assertions.
Usage: python -m pytest tests/visualisation/test_cli_wrappers.py
Dependencies: pytest; pysweb
"""
from importlib import import_module
import subprocess
import sys

import pytest

@pytest.mark.parametrize(
    "module_name",
    [
        "pysweb.visualisation.plot_time_series",
        "pysweb.visualisation.plot_heatmap",
    ],
)
def test_package_visualisation_modules_import_cleanly(module_name):
    assert import_module(module_name).__name__ == module_name


def test_plot_font_configuration_registers_user_font_path(tmp_path):
    module = import_module("pysweb.visualisation.plot_time_series")
    font_path = tmp_path / "Arial.ttf"
    font_path.write_bytes(b"placeholder")

    class FakeFontRegistry:
        def __init__(self):
            self.added = []
            self.fontManager = self

        def addfont(self, path):
            self.added.append(path)

    class FakePyplot:
        rcParams = {}

    registry = FakeFontRegistry()

    module._configure_plot_font(
        pyplot=FakePyplot,
        font_manager=registry,
        font_paths=[font_path],
    )

    assert registry.added == [str(font_path)]
    assert FakePyplot.rcParams["font.family"] == "Arial"


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
    assert args.flux_gamma == 0.5


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, "-m", "pysweb.visualisation.plot_time_series", "--help"],
        [sys.executable, "-m", "pysweb.visualisation.plot_heatmap", "--help"],
    ],
)
def test_visualisation_entrypoints_exit_cleanly_for_help(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert result.stderr == ""


def test_component_selection_and_labels_follow_notation():
    import xarray as xr
    module = import_module("pysweb.visualisation.plot_time_series")
    da = xr.DataArray([1.0, 2.0], dims="time", attrs={"units": "mm/day"})
    ds = xr.Dataset({"ET": da, "E_s": da, "E_t": da, "E": da, "T": da})
    assert module._default_variables(ds, "ssebop") == ["ET", "E_s", "E_t"]
    assert module._format_var_label("ET") == r"$ET_a$"
    assert module._format_var_label("E_s") == r"$E_s$"
    assert module._format_var_label("T") == module._format_var_label("E_t")
    assert module._format_var_label("Tc") == module._format_var_label("f_t")
    legacy = ds.drop_vars(["E_s", "E_t"])
    with pytest.warns(FutureWarning):
        assert module._resolve_variables(legacy, ["E_s", "E_t"], "ssebop") == ["E", "T"]
    ds["T"] = ds["T"] * 2
    with pytest.raises(ValueError, match="Conflicting component"):
        module._default_variables(ds, "ssebop")
