#!/usr/bin/env python3
"""
Script: test_paths.py
Objective: Verify package-level meteorology path helpers resolve ERA5-Land and explicit inputs correctly.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: Temporary test paths and direct helper calls.
Outputs: Test assertions.
Usage: pytest tests/met/test_paths.py
Dependencies: pathlib, pytest
"""
from pathlib import Path

import pytest

from pysweb.met.paths import infer_met_var_from_path, resolve_met_input_paths


def test_resolve_met_input_paths_builds_era5land_stack_name(tmp_path: Path):
    met_dir = tmp_path / "met"
    met_dir.mkdir()
    path = met_dir / "tmax_daily_2024-01-01_2024-01-03.nc"
    path.write_text("", encoding="utf-8")

    result = resolve_met_input_paths(
        field="tmax",
        explicit_path=None,
        met_dir=str(met_dir),
        silo_dir=None,
        date_range="2024-01-01 to 2024-01-03",
    )

    assert result == str(path)


def test_infer_met_var_from_custom_file_uses_default_for_non_legacy_names():
    assert infer_met_var_from_path("custom_tmax.nc", default_var="tmax") == "tmax"


def test_resolve_met_input_paths_raises_for_missing_explicit_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_met_input_paths(
            field="ea",
            explicit_path=str(tmp_path / "missing.nc"),
            met_dir=None,
            silo_dir=None,
            date_range=None,
        )
