#!/usr/bin/env python3
"""
Script: test_preprocess.py
Objective: Verify SWB preprocess helpers and package-owned preprocessing orchestration for OpenLandMap and GSSM inputs.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Pytest fixtures, temporary directories, and in-memory xarray DataArrays.
Outputs: Test assertions.
Usage: python -m pytest tests/swb/test_preprocess.py -q
Dependencies: numpy, pandas, pytest, xarray
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysweb.swb.preprocess import (
    GSSM_SCALE_FACTOR,
    OPENLANDMAP_LAYER_SPECS,
    _build_layer_bottoms_mm,
    _matching_gssm_image_ids,
    _parse_gssm_band_date,
    _rename_reference_ssm,
    preprocess_inputs,
)


def _forcing_array(
    values: np.ndarray,
    *,
    name: str,
    dates: list[str],
) -> xr.DataArray:
    return xr.DataArray(
        values.astype(np.float32, copy=False),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.to_datetime(dates),
            "lat": np.array([-35.0], dtype=float),
            "lon": np.array([148.0], dtype=float),
        },
        name=name,
    )


def _soil_array(name: str, values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values.astype(np.float32, copy=False),
        dims=("layer", "lat", "lon"),
        coords={
            "layer": np.array([1, 2, 3, 4, 5], dtype=int),
            "lat": np.array([-35.0], dtype=float),
            "lon": np.array([148.0], dtype=float),
        },
        name=name,
        attrs={"layer_bottoms_mm": _build_layer_bottoms_mm().tolist()},
    )


def test_parse_gssm_band_date_parses_daily_band_names():
    assert _parse_gssm_band_date("band_2000_03_05_classification") == pd.Timestamp("2000-03-05")
    assert _parse_gssm_band_date("band_2020_12_31_classification") == pd.Timestamp("2020-12-31")


def test_openlandmap_depth_mapping_matches_swb_layer_bottoms():
    assert OPENLANDMAP_LAYER_SPECS == [
        ("b0", 50.0),
        ("b10", 150.0),
        ("b30", 300.0),
        ("b60", 600.0),
        ("b100", 1000.0),
    ]
    np.testing.assert_allclose(_build_layer_bottoms_mm(), np.array([50.0, 150.0, 300.0, 600.0, 1000.0]))


def test_reference_ssm_scaling_and_naming_are_neutral():
    raw = xr.DataArray(
        np.array([[[250.0]], [[500.0]]], dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.to_datetime(["2000-03-05", "2000-03-06"]),
            "lat": np.array([-35.0]),
            "lon": np.array([148.0]),
        },
        name="gssm_raw",
    )

    renamed = _rename_reference_ssm(raw / GSSM_SCALE_FACTOR)

    assert renamed.name == "reference_ssm"
    assert renamed.attrs["units"] == "m3 m-3"
    np.testing.assert_allclose(renamed.values[:, 0, 0], np.array([0.25, 0.5]))


def test_matching_gssm_image_ids_filters_collection_indices_by_year():
    indices = [
        "SM2000Africa1km",
        "SM2000Asia1_1km",
        "SM2001Africa1km",
    ]

    assert _matching_gssm_image_ids(indices, year=2000) == [
        "SM2000Africa1km",
        "SM2000Asia1_1km",
    ]


def test_preprocess_inputs_writes_expected_outputs(monkeypatch, tmp_path: Path):
    rain = _forcing_array(
        np.array([[[10.0]], [[11.0]]], dtype=np.float32),
        name="precipitation",
        dates=["2024-01-01", "2024-01-02"],
    )
    effective_precip = _forcing_array(
        np.array([[[8.0]], [[9.0]]], dtype=np.float32),
        name="effective_precipitation",
        dates=["2024-01-01", "2024-01-02"],
    )
    et = _forcing_array(
        np.array([[[4.0]], [[5.0]]], dtype=np.float32),
        name="et",
        dates=["2024-01-01", "2024-01-02"],
    )
    t = _forcing_array(
        np.array([[[2.0]], [[3.0]]], dtype=np.float32),
        name="t",
        dates=["2024-01-01", "2024-01-02"],
    )
    reference = _forcing_array(
        np.array([[[0.2]], [[0.3]]], dtype=np.float32),
        name="reference_ssm",
        dates=["2024-01-01", "2024-01-02"],
    )
    soil_arrays = {
        "porosity": _soil_array("porosity", np.full((5, 1, 1), 0.45, dtype=np.float32)),
        "wilting_point": _soil_array("wilting_point", np.full((5, 1, 1), 0.15, dtype=np.float32)),
        "available_water_capacity": _soil_array(
            "available_water_capacity",
            np.full((5, 1, 1), 0.20, dtype=np.float32),
        ),
        "b_coefficient": _soil_array("b_coefficient", np.full((5, 1, 1), 4.5, dtype=np.float32)),
        "conductivity_sat": _soil_array("conductivity_sat", np.full((5, 1, 1), 8.0, dtype=np.float32)),
    }

    monkeypatch.setattr("pysweb.swb.preprocess.process_precipitation", lambda args, grid, start, end: rain)
    monkeypatch.setattr(
        "pysweb.swb.preprocess.compute_effective_precipitation_smith",
        lambda raw_rain, dtype: effective_precip,
    )
    monkeypatch.setattr(
        "pysweb.swb.preprocess.process_et",
        lambda args, grid, dates: {"et": et, "t": t},
    )
    monkeypatch.setattr(
        "pysweb.swb.preprocess._load_openlandmap_predictors",
        lambda extent, gee_project: {"clay": "clay", "sand": "sand", "soc": "soc"},
    )
    monkeypatch.setattr(
        "pysweb.swb.preprocess.process_soil_properties_from_openlandmap",
        lambda args, grid, predictors: soil_arrays,
    )
    monkeypatch.setattr(
        "pysweb.swb.preprocess._load_reference_ssm",
        lambda **kwargs: reference,
    )

    preprocess_inputs(
        date_range=["2024-01-01", "2024-01-02"],
        extent=[148.0, -35.1, 148.1, -35.0],
        sm_res=0.1,
        output_dir=str(tmp_path),
        reference_source="gssm1km",
        reference_ssm_asset="users/qianrswaterr/GlobalSSM1km0509",
        gee_project="yiyu-research",
        workers=1,
    )

    expected_files = {
        "rain_daily_20240101_20240102.nc": "precipitation",
        "effective_precip_daily_20240101_20240102.nc": "effective_precipitation",
        "et_daily_20240101_20240102.nc": "et",
        "t_daily_20240101_20240102.nc": "t",
        "reference_ssm_daily_20240101_20240102.nc": "reference_ssm",
        "soil_porosity.nc": "porosity",
        "soil_wilting_point.nc": "wilting_point",
        "soil_available_water_capacity.nc": "available_water_capacity",
        "soil_b_coefficient.nc": "b_coefficient",
        "soil_conductivity_sat.nc": "conductivity_sat",
    }

    for filename, variable in expected_files.items():
        path = tmp_path / filename
        assert path.exists(), f"missing output: {path}"
        with xr.open_dataset(path) as ds:
            assert variable in ds
