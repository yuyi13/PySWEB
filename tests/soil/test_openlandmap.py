#!/usr/bin/env python3
"""
Script: test_openlandmap.py
Objective: Verify the OpenLandMap soil backend owns depth mapping, predictor loading, and hydraulic derivation behavior.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Soil backend helpers, fake Earth Engine objects, and in-memory xarray DataArrays.
Outputs: Test assertions.
Usage: python -m pytest tests/soil/test_openlandmap.py -q
Dependencies: numpy, pytest, xarray
"""

from argparse import Namespace
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr


class _FakeInfo:
    def __init__(self, value):
        self._value = value

    def getInfo(self):
        return self._value


class _FakeImage:
    def __init__(self, bands: dict[str, float], image_id: str | None = None):
        self._bands = dict(bands)
        self.image_id = image_id

    def clip(self, region):
        return self

    def bandNames(self):
        return _FakeInfo(list(self._bands))

    def select(self, names):
        return _FakeImage({name: self._bands[name] for name in names}, image_id=self.image_id)


class _FakeGeometry:
    @staticmethod
    def Rectangle(coords, **kwargs):
        return tuple(coords)


class _FakeEE:
    Geometry = _FakeGeometry

    @staticmethod
    def Initialize(project=None):
        return None

    @staticmethod
    def Image(image_id):
        raise NotImplementedError


def _grid() -> SimpleNamespace:
    return SimpleNamespace(
        latitudes=np.array([-35.05, -35.15], dtype=float),
        longitudes=np.array([148.05, 148.15], dtype=float),
        lat_dim="lat",
        lon_dim="lon",
    )


def _passthrough_reproject(data, grid, resampling=None):
    return data


def test_openlandmap_depth_mapping_matches_layer_bottoms():
    openlandmap = import_module("pysweb.soil.openlandmap")

    assert openlandmap.OPENLANDMAP_LAYER_SPECS == [
        ("b0", 50.0),
        ("b10", 150.0),
        ("b30", 300.0),
        ("b60", 600.0),
        ("b100", 1000.0),
    ]
    np.testing.assert_allclose(
        openlandmap._build_layer_bottoms_mm(),
        np.array([50.0, 150.0, 300.0, 600.0, 1000.0]),
    )


def test_load_openlandmap_predictors_uses_openlandmap_export_scale(monkeypatch):
    openlandmap = import_module("pysweb.soil.openlandmap")
    recorded_scales = []
    dataset_bands = {
        band_name: idx + 1.0 for idx, (band_name, _) in enumerate(openlandmap.OPENLANDMAP_LAYER_SPECS)
    }

    class FakeEEForOpenLandMap(_FakeEE):
        @staticmethod
        def Image(image_id):
            return _FakeImage(dataset_bands, image_id=image_id)

    def fake_download(image, extent, name, scale_m):
        recorded_scales.append((name, scale_m, image.bandNames().getInfo()))
        return xr.DataArray(
            np.ones((len(openlandmap.OPENLANDMAP_LAYER_SPECS), 1, 1), dtype=np.float32),
            dims=("band", "lat", "lon"),
            coords={
                "band": np.array(
                    [band_name for band_name, _ in openlandmap.OPENLANDMAP_LAYER_SPECS],
                    dtype=object,
                ),
                "lat": np.array([-35.0]),
                "lon": np.array([148.0]),
            },
            name=name,
        )

    monkeypatch.setattr(openlandmap, "ee", FakeEEForOpenLandMap)
    monkeypatch.setattr(openlandmap, "_download_ee_multiband_image", fake_download)

    predictors = openlandmap._load_openlandmap_predictors((148.0, -35.1, 148.1, -35.0), "yiyu-research")

    assert set(predictors) == {"clay", "sand", "soc"}
    assert [scale for _, scale, _ in recorded_scales] == [250.0, 250.0, 250.0]


def test_process_soil_properties_from_openlandmap_returns_five_layers_with_expected_depths():
    openlandmap = import_module("pysweb.soil.openlandmap")
    args = Namespace(dtype="float32")
    band_names = np.array([band_name for band_name, _ in openlandmap.OPENLANDMAP_LAYER_SPECS], dtype=object)
    coords = {
        "band": band_names,
        "lat": np.array([-35.05, -35.15], dtype=float),
        "lon": np.array([148.05, 148.15], dtype=float),
    }
    soil_predictors = {
        "clay": xr.DataArray(np.full((5, 2, 2), 30.0, dtype=np.float32), dims=("band", "lat", "lon"), coords=coords),
        "sand": xr.DataArray(np.full((5, 2, 2), 40.0, dtype=np.float32), dims=("band", "lat", "lon"), coords=coords),
        "soc": xr.DataArray(np.full((5, 2, 2), 10.0, dtype=np.float32), dims=("band", "lat", "lon"), coords=coords),
    }

    soil_arrays = openlandmap.process_soil_properties_from_openlandmap(
        args,
        _grid(),
        soil_predictors,
        reproject_to_template=_passthrough_reproject,
    )

    assert set(soil_arrays) == {
        "porosity",
        "wilting_point",
        "available_water_capacity",
        "b_coefficient",
        "conductivity_sat",
    }
    for da in soil_arrays.values():
        assert da.shape == (5, 2, 2)
        np.testing.assert_allclose(da.coords["layer_depth"].values, openlandmap._build_layer_bottoms_mm())
        assert da.attrs["layer_bottoms_mm"] == openlandmap._build_layer_bottoms_mm().tolist()


def test_load_soil_properties_returns_soil_outputs(monkeypatch):
    api = import_module("pysweb.soil.api")
    openlandmap = import_module("pysweb.soil.openlandmap")
    grid = _grid()
    args = Namespace(
        extent=[148.0, -35.2, 148.2, -35.0],
        gee_project="yiyu-research",
        dtype="float32",
    )
    expected_arrays = {
        "porosity": xr.DataArray(
            np.full((5, 2, 2), 0.45, dtype=np.float32),
            dims=("layer", "lat", "lon"),
            coords={
                "layer": np.array([1, 2, 3, 4, 5], dtype=int),
                "lat": grid.latitudes,
                "lon": grid.longitudes,
            },
            name="porosity",
        )
    }
    recorded = {}

    def fake_load_predictors(extent, gee_project):
        recorded["extent"] = extent
        recorded["gee_project"] = gee_project
        return {"clay": "clay", "sand": "sand", "soc": "soc"}

    def fake_process(args, grid, soil_predictors, *, reproject_to_template):
        recorded["args"] = args
        recorded["grid"] = grid
        recorded["soil_predictors"] = soil_predictors
        recorded["reproject_to_template"] = reproject_to_template
        return expected_arrays

    monkeypatch.setattr(openlandmap, "_load_openlandmap_predictors", fake_load_predictors)
    monkeypatch.setattr(openlandmap, "process_soil_properties_from_openlandmap", fake_process)

    result = openlandmap.load_soil_properties(
        args=args,
        grid=grid,
        reproject_to_template=_passthrough_reproject,
    )

    assert isinstance(result, api.SoilOutputs)
    assert result.arrays is expected_arrays
    np.testing.assert_allclose(result.layer_bottoms_mm, openlandmap._build_layer_bottoms_mm())
    assert recorded == {
        "extent": tuple(args.extent),
        "gee_project": "yiyu-research",
        "args": args,
        "grid": grid,
        "soil_predictors": {"clay": "clay", "sand": "sand", "soc": "soc"},
        "reproject_to_template": _passthrough_reproject,
    }
