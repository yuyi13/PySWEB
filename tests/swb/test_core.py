#!/usr/bin/env python3
"""
Script: test_core.py
Objective: Verify reusable SWB core helpers load forcing and soil inputs consistently for package-owned workflows.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-17
Inputs: Temporary NetCDF fixtures and in-memory xarray DataArrays created by pytest.
Outputs: Test assertions.
Usage: pytest tests/swb/test_core.py
Dependencies: numpy, pytest, xarray
"""
from pathlib import Path
import sys

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pysweb.swb.core as swb_core


def _write_single_var_dataset(
    path: Path,
    name: str,
    values: np.ndarray,
    dims: tuple[str, ...],
    coords: dict[str, np.ndarray],
    attrs: dict[str, object] | None = None,
) -> None:
    da = xr.DataArray(
        values,
        dims = dims,
        coords = coords,
        name = name,
        attrs = attrs or {},
    )
    xr.Dataset({name: da}).to_netcdf(path)


def _soil_coords() -> dict[str, np.ndarray]:
    return {
        "layer": np.array([1, 2], dtype=int),
        "lat": np.array([-35.0, -34.5], dtype=float),
        "lon": np.array([148.0], dtype=float),
    }


def _write_required_soil_inputs(
    soil_dir: Path,
    *,
    porosity_path: Path | None = None,
) -> None:
    coords = _soil_coords()
    base_values = np.array(
        [
            [[0.40], [0.42]],
            [[0.38], [0.39]],
        ],
        dtype = np.float32,
    )
    filenames = {
        "porosity": "soil_porosity.nc",
        "wilting_point": "soil_wilting_point.nc",
        "available_water_capacity": "soil_available_water_capacity.nc",
        "b_coefficient": "soil_b_coefficient.nc",
        "conductivity_sat": "soil_conductivity_sat.nc",
    }
    offsets = {
        "porosity": 0.00,
        "wilting_point": -0.20,
        "available_water_capacity": -0.10,
        "b_coefficient": 2.00,
        "conductivity_sat": 5.00,
    }

    soil_dir.mkdir(parents = True, exist_ok = True)
    for key, filename in filenames.items():
        target = porosity_path if key == "porosity" and porosity_path is not None else soil_dir / filename
        _write_single_var_dataset(
            target,
            key,
            base_values + offsets[key],
            ("layer", "lat", "lon"),
            coords,
            attrs = {"layer_bottoms_mm": np.array([50.0, 150.0], dtype=float)},
        )


def test_load_single_variable_and_load_forcing_slice(tmp_path: Path):
    single_var_path = tmp_path / "single.nc"
    _write_single_var_dataset(
        single_var_path,
        "porosity",
        np.array([[0.4, 0.5]], dtype=np.float32),
        ("lat", "lon"),
        {
            "lat": np.array([-35.0], dtype=float),
            "lon": np.array([148.0, 148.5], dtype=float),
        },
    )

    forcing_path = tmp_path / "forcing.nc"
    _write_single_var_dataset(
        forcing_path,
        "effective_precipitation",
        np.array([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float32),
        ("time", "lat", "lon"),
        {
            "time": np.array(
                ["2024-01-01", "2024-01-02", "2024-01-03"],
                dtype = "datetime64[ns]",
            ),
            "lat": np.array([-35.0], dtype=float),
            "lon": np.array([148.0], dtype=float),
        },
    )

    loaded = swb_core.load_single_variable(single_var_path)
    forcing = swb_core.load_forcing(
        forcing_path,
        "effective_precipitation",
        "2024-01-02",
        "2024-01-03",
    )

    assert loaded.name == "porosity"
    np.testing.assert_allclose(loaded.values, np.array([[0.4, 0.5]], dtype=np.float32))
    assert forcing.name == "effective_precipitation"
    assert forcing.sizes["time"] == 2
    np.testing.assert_allclose(forcing.values[:, 0, 0], np.array([2.0, 3.0], dtype=np.float32))


def test_resolve_soil_paths_and_load_soil_arrays_support_overrides(tmp_path: Path):
    soil_dir = tmp_path / "soil"
    override_path = tmp_path / "override_porosity.nc"
    _write_required_soil_inputs(soil_dir)
    _write_required_soil_inputs(tmp_path / "unused", porosity_path = override_path)

    paths = swb_core.resolve_soil_paths(
        soil_dir = soil_dir,
        soil_porosity = override_path,
    )
    arrays, loaded_paths = swb_core.load_soil_arrays(
        soil_dir = soil_dir,
        soil_porosity = override_path,
    )

    assert paths["porosity"] == override_path.resolve()
    assert loaded_paths == paths
    np.testing.assert_allclose(
        arrays["porosity"].values,
        swb_core.load_single_variable(override_path).values,
    )


def test_infer_layer_bottoms_prefers_metadata_then_user_then_defaults():
    metadata_arrays = {
        "porosity": xr.DataArray(
            np.ones((2, 1, 1), dtype=np.float32),
            dims = ("layer", "lat", "lon"),
            coords = {
                "layer": np.array([1, 2], dtype=int),
                "lat": np.array([-35.0], dtype=float),
                "lon": np.array([148.0], dtype=float),
            },
            attrs = {"layer_bottoms_mm": np.array([75.0, 225.0], dtype=float)},
        )
    }
    bare_arrays = {
        "porosity": xr.DataArray(
            np.ones((2, 1, 1), dtype=np.float32),
            dims = ("layer", "lat", "lon"),
            coords = {
                "layer": np.array([1, 2], dtype=int),
                "lat": np.array([-35.0], dtype=float),
                "lon": np.array([148.0], dtype=float),
            },
        )
    }

    np.testing.assert_allclose(
        swb_core.infer_layer_bottoms(metadata_arrays, user_bottoms = [50.0, 150.0]),
        np.array([75.0, 225.0], dtype=float),
    )
    np.testing.assert_allclose(
        swb_core.infer_layer_bottoms(bare_arrays, user_bottoms = [60.0, 180.0]),
        np.array([60.0, 180.0], dtype=float),
    )
    np.testing.assert_allclose(
        swb_core.infer_layer_bottoms(bare_arrays, user_bottoms = None),
        np.asarray(swb_core.DEFAULT_LAYER_BOTTOMS_MM, dtype=float),
    )


def test_ensure_matching_grid_rejects_mismatched_soil_arrays():
    reference = xr.DataArray(
        np.zeros((2, 2), dtype=np.float32),
        dims = ("lat", "lon"),
        coords = {
            "lat": np.array([-35.0, -34.5], dtype=float),
            "lon": np.array([148.0, 148.5], dtype=float),
        },
    )
    soil_arrays = {
        "porosity": xr.DataArray(
            np.zeros((2, 2, 2), dtype=np.float32),
            dims = ("layer", "lat", "lon"),
            coords = {
                "layer": np.array([1, 2], dtype=int),
                "lat": np.array([-35.0, -34.0], dtype=float),
                "lon": np.array([148.0, 148.5], dtype=float),
            },
        )
    }

    with pytest.raises(ValueError, match="same grid"):
        swb_core.ensure_matching_grid(reference, soil_arrays, "lat", "lon")


def test_prepare_soil_property_grids_extracts_cell_values_and_flags_invalid_soil():
    coords = _soil_coords()
    values = {
        "porosity": np.array([[[0.45], [0.42]], [[0.40], [0.38]]], dtype=np.float32),
        "wilting_point": np.array([[[0.15], [0.14]], [[0.16], [0.15]]], dtype=np.float32),
        "available_water_capacity": np.array([[[0.20], [0.18]], [[0.19], [0.17]]], dtype=np.float32),
        "b_coefficient": np.array([[[4.0], [4.2]], [[4.4], [4.6]]], dtype=np.float32),
        "conductivity_sat": np.array([[[8.0], [8.5]], [[9.0], [9.5]]], dtype=np.float32),
    }
    soil_arrays = {
        key: xr.DataArray(
            array,
            dims = ("layer", "lat", "lon"),
            coords = coords,
        )
        for key, array in values.items()
    }

    soil_grids = swb_core.prepare_soil_property_grids(
        soil_arrays,
        np.array([50.0, 150.0], dtype=float),
        lat_dim = "lat",
        lon_dim = "lon",
        root_beta = 0.93,
        drainage_slope = 0.25,
        drainage_upper_limit = 14.0,
        drainage_lower_limit = 1.5,
        sm_max_factor = 1.1,
        sm_min_factor = 0.85,
    )
    props = swb_core.extract_soil_properties_for_cell(soil_grids, 1, 0)

    np.testing.assert_allclose(props["layer_depth"], np.array([50.0, 150.0], dtype=float))
    np.testing.assert_allclose(props["layer_thickness"], np.array([50.0, 100.0], dtype=float))
    np.testing.assert_allclose(props["porosity"], np.array([0.42, 0.38], dtype=np.float32))
    assert props["root_beta"] == pytest.approx(0.93)
    assert props["drainage_slope"] == pytest.approx(0.25)
    assert props["sm_max_factor"] == pytest.approx(1.1)
    assert swb_core.has_invalid_soil_values(props) is False

    invalid_props = dict(props)
    invalid_props["porosity"] = np.array([np.nan, 0.38], dtype=float)
    assert swb_core.has_invalid_soil_values(invalid_props) is True
