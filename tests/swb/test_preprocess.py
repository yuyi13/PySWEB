#!/usr/bin/env python3
"""
Script: test_preprocess.py
Objective: Verify SWB preprocess helpers and package-owned preprocessing orchestration for forcing, soil dispatch, and reference SSM outputs.
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
from pysweb.soil.api import SoilOutputs
import pysweb.swb.preprocess as preprocess_module

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysweb.swb.preprocess import (
    GSSM_SCALE_FACTOR,
    _build_args,
    _build_target_grid,
    _load_reference_ssm,
    _matching_gssm_image_ids,
    _parse_gssm_band_date,
    _rename_reference_ssm,
    process_et,
    process_precipitation,
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
    layer_bottoms_mm = [50.0, 150.0, 300.0, 600.0, 1000.0]
    return xr.DataArray(
        values.astype(np.float32, copy=False),
        dims=("layer", "lat", "lon"),
        coords={
            "layer": np.array([1, 2, 3, 4, 5], dtype=int),
            "lat": np.array([-35.0], dtype=float),
            "lon": np.array([148.0], dtype=float),
        },
        name=name,
        attrs={"layer_bottoms_mm": layer_bottoms_mm},
    )


def _grid_args(*, extent: list[float], sm_res: float) -> object:
    return _build_args(
        {
            "extent": extent,
            "sm_res": sm_res,
            "lat_dim": "lat",
            "lon_dim": "lon",
            "crs": "EPSG:4326",
        }
    )


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


class _FakeImageCollection:
    registry: dict[str, list[_FakeImage]] = {}

    def __init__(self, items):
        if isinstance(items, str):
            self._images = list(self.registry[items])
        else:
            self._images = list(items)

    def aggregate_array(self, field_name: str):
        assert field_name == "system:index"
        return _FakeInfo([image.image_id for image in self._images])

    def filter(self, predicate):
        _, field_name, expected = predicate
        assert field_name == "system:index"
        return _FakeImageCollection([image for image in self._images if image.image_id == expected])

    def first(self):
        return self._images[0]

    def mosaic(self):
        bands = {}
        for image in self._images:
            bands.update(image._bands)
        return _FakeImage(bands)


class _FakeFilter:
    @staticmethod
    def eq(field_name: str, value: str):
        return ("eq", field_name, value)


class _FakeGeometry:
    @staticmethod
    def Rectangle(coords, **kwargs):
        return tuple(coords)


class _FakeEE:
    ImageCollection = _FakeImageCollection
    Filter = _FakeFilter
    Geometry = _FakeGeometry

    @staticmethod
    def Initialize(project=None):
        return None

    @staticmethod
    def Image(image):
        return image


def test_parse_gssm_band_date_parses_daily_band_names():
    assert _parse_gssm_band_date("band_2000_03_05_classification") == pd.Timestamp("2000-03-05")
    assert _parse_gssm_band_date("band_2020_12_31_classification") == pd.Timestamp("2020-12-31")


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


def test_build_target_grid_treats_extent_as_bbox_edges():
    grid = _build_target_grid(_grid_args(extent=[148.0, -35.1, 148.1, -35.0], sm_res=0.1))

    assert grid.template.shape == (1, 1)
    np.testing.assert_allclose(grid.latitudes, np.array([-35.05]))
    np.testing.assert_allclose(grid.longitudes, np.array([148.05]))

    transform = grid.template.rio.transform()
    assert transform.a == pytest.approx(0.1)
    assert transform.e == pytest.approx(-0.1)
    assert transform.c == pytest.approx(148.0)
    assert transform.f == pytest.approx(-35.0)


def test_load_reference_ssm_filters_bands_scales_values_and_uses_reference_name(monkeypatch):
    _FakeImageCollection.registry = {
        "users/qianrswaterr/GlobalSSM1km0509": [
            _FakeImage(
                {
                    "band_2000_03_04_classification": 100.0,
                    "band_2000_03_05_classification": 250.0,
                },
                image_id="SM2000Africa1km",
            ),
            _FakeImage(
                {
                    "band_2000_03_06_classification": 500.0,
                    "band_2000_03_07_classification": 750.0,
                },
                image_id="SM2000Asia1_1km",
            ),
        ]
    }
    recorded_scales = []

    def fake_download(image, extent, name, scale_m):
        recorded_scales.append((name, scale_m, image.bandNames().getInfo()))
        values = np.array([image._bands[band_name] for band_name in image.bandNames().getInfo()], dtype=np.float32)
        return xr.DataArray(
            values[:, None, None],
            dims=("band", "lat", "lon"),
            coords={
                "band": np.array(image.bandNames().getInfo(), dtype=object),
                "lat": np.array([-35.0]),
                "lon": np.array([148.0]),
            },
            name=name,
        )

    monkeypatch.setattr(preprocess_module, "ee", _FakeEE)
    monkeypatch.setattr(preprocess_module, "_download_ee_multiband_image", fake_download)

    result = _load_reference_ssm(
        extent=(148.0, -35.1, 148.1, -35.0),
        dates=pd.date_range("2000-03-05", "2000-03-06", freq="D"),
        reference_ssm_asset="users/qianrswaterr/GlobalSSM1km0509",
        gee_project="yiyu-research",
    )

    assert result.name == "reference_ssm"
    assert result.attrs["units"] == "m3 m-3"
    np.testing.assert_array_equal(result.coords["time"].values, pd.to_datetime(["2000-03-05", "2000-03-06"]).values)
    np.testing.assert_allclose(result.values[:, 0, 0], np.array([0.25, 0.5], dtype=np.float32))
    assert recorded_scales == [("gssm_raw_2000", 1000.0, ["band_2000_03_05_classification", "band_2000_03_06_classification"])]


def test_load_reference_ssm_raises_when_requested_days_are_missing(monkeypatch):
    _FakeImageCollection.registry = {
        "users/qianrswaterr/GlobalSSM1km0509": [
            _FakeImage(
                {
                    "band_2000_03_05_classification": 250.0,
                    "band_2000_03_07_classification": 750.0,
                },
                image_id="SM2000Africa1km",
            ),
        ]
    }

    def fake_download(image, extent, name, scale_m):
        values = np.array([image._bands[band_name] for band_name in image.bandNames().getInfo()], dtype=np.float32)
        return xr.DataArray(
            values[:, None, None],
            dims=("band", "lat", "lon"),
            coords={
                "band": np.array(image.bandNames().getInfo(), dtype=object),
                "lat": np.array([-35.0]),
                "lon": np.array([148.0]),
            },
            name=name,
        )

    monkeypatch.setattr(preprocess_module, "ee", _FakeEE)
    monkeypatch.setattr(preprocess_module, "_download_ee_multiband_image", fake_download)

    with pytest.raises(ValueError, match="2000-03-06"):
        _load_reference_ssm(
            extent=(148.0, -35.1, 148.1, -35.0),
            dates=pd.date_range("2000-03-05", "2000-03-07", freq="D"),
            reference_ssm_asset="users/qianrswaterr/GlobalSSM1km0509",
            gee_project="yiyu-research",
        )


def test_process_precipitation_rejects_missing_internal_days_from_netcdf(tmp_path: Path):
    rain_path = tmp_path / "rain.nc"
    xr.Dataset(
        {
            "precipitation": _forcing_array(
                np.array([[[10.0]], [[12.0]]], dtype=np.float32),
                name="precipitation",
                dates=["2024-01-01", "2024-01-03"],
            )
        }
    ).to_netcdf(rain_path)

    args = _build_args(
        {
            "rain_file": str(rain_path),
            "rain_var": "precipitation",
            "extent": None,
            "workers": 1,
            "dtype": "float32",
            "lat_dim": "lat",
            "lon_dim": "lon",
        }
    )
    grid = _build_target_grid(_grid_args(extent=[148.0, -35.1, 148.1, -35.0], sm_res=0.1))

    with pytest.raises(ValueError, match="2024-01-02"):
        process_precipitation(args, grid, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03"))


def test_process_et_rejects_missing_internal_days_from_netcdf(tmp_path: Path):
    et_path = tmp_path / "et.nc"
    dates = pd.to_datetime(["2024-01-01", "2024-01-03"])
    xr.Dataset(
        {
            "et": xr.DataArray(
                np.full((2, 1, 1), 4.0, dtype=np.float32),
                dims=("time", "lat", "lon"),
                coords={"time": dates, "lat": np.array([-35.0]), "lon": np.array([148.0])},
            ),
            "t": xr.DataArray(
                np.full((2, 1, 1), 2.0, dtype=np.float32),
                dims=("time", "lat", "lon"),
                coords={"time": dates, "lat": np.array([-35.0]), "lon": np.array([148.0])},
            ),
        }
    ).to_netcdf(et_path)

    args = _build_args(
        {
            "et_file": str(et_path),
            "et_var": "et",
            "t_var": "t",
            "e_var": None,
            "ndvi_var": "ndvi_interp",
            "extent": None,
            "workers": 1,
            "dtype": "float32",
            "lat_dim": "lat",
            "lon_dim": "lon",
        }
    )
    grid = _build_target_grid(_grid_args(extent=[148.0, -35.1, 148.1, -35.0], sm_res=0.1))

    with pytest.raises(ValueError, match="2024-01-02"):
        process_et(args, grid, pd.date_range("2024-01-01", "2024-01-03", freq="D"))


def test_process_precipitation_accepts_noon_stamped_daily_netcdf(tmp_path: Path):
    rain_path = tmp_path / "rain_noon.nc"
    xr.Dataset(
        {
            "precipitation": _forcing_array(
                np.array([[[10.0]], [[11.0]]], dtype=np.float32),
                name="precipitation",
                dates=["2024-01-01 12:00:00", "2024-01-02 12:00:00"],
            )
        }
    ).to_netcdf(rain_path)

    args = _build_args(
        {
            "rain_file": str(rain_path),
            "rain_var": "precipitation",
            "extent": None,
            "workers": 1,
            "dtype": "float32",
            "lat_dim": "lat",
            "lon_dim": "lon",
        }
    )
    grid = _build_target_grid(_grid_args(extent=[148.0, -35.1, 148.1, -35.0], sm_res=0.1))

    result = process_precipitation(args, grid, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))

    np.testing.assert_array_equal(result.coords["time"].values, pd.to_datetime(["2024-01-01", "2024-01-02"]).values)


def test_process_et_rejects_netcdf_without_time_coordinate(tmp_path: Path):
    et_path = tmp_path / "et_no_time.nc"
    xr.Dataset(
        {
            "et": xr.DataArray(
                np.full((1, 1), 4.0, dtype=np.float32),
                dims=("lat", "lon"),
                coords={"lat": np.array([-35.0]), "lon": np.array([148.0])},
            ),
            "t": xr.DataArray(
                np.full((1, 1), 2.0, dtype=np.float32),
                dims=("lat", "lon"),
                coords={"lat": np.array([-35.0]), "lon": np.array([148.0])},
            ),
        }
    ).to_netcdf(et_path)

    args = _build_args(
        {
            "et_file": str(et_path),
            "et_var": "et",
            "t_var": "t",
            "e_var": None,
            "ndvi_var": "ndvi_interp",
            "extent": None,
            "workers": 1,
            "dtype": "float32",
            "lat_dim": "lat",
            "lon_dim": "lon",
        }
    )
    grid = _build_target_grid(_grid_args(extent=[148.0, -35.1, 148.1, -35.0], sm_res=0.1))

    with pytest.raises(ValueError, match="time coordinate"):
        process_et(args, grid, pd.date_range("2024-01-01", "2024-01-02", freq="D"))


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

    monkeypatch.setattr(preprocess_module, "process_precipitation", lambda args, grid, start, end: rain)
    monkeypatch.setattr(
        preprocess_module,
        "compute_effective_precipitation_smith",
        lambda raw_rain, dtype: effective_precip,
    )
    monkeypatch.setattr(
        preprocess_module,
        "process_et",
        lambda args, grid, dates: {"et": et, "t": t},
    )
    monkeypatch.setattr(
        preprocess_module.soil_api,
        "load_soil_properties",
        lambda *, soil_source, args, grid, reproject_to_template: SoilOutputs(
            arrays=soil_arrays,
            layer_bottoms_mm=np.array([50.0, 150.0, 300.0, 600.0, 1000.0], dtype=float),
        ),
    )
    monkeypatch.setattr(
        preprocess_module,
        "_load_reference_ssm",
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


def test_preprocess_inputs_delegates_soil_loading_to_soil_api(monkeypatch, tmp_path: Path):
    rain = _forcing_array(
        np.array([[[10.0]], [[11.0]]], dtype=np.float32),
        name="precipitation",
        dates=["2024-01-01", "2024-01-02"],
    )
    et = _forcing_array(
        np.array([[[4.0]], [[5.0]]], dtype=np.float32),
        name="et",
        dates=["2024-01-01", "2024-01-02"],
    )
    soil_arrays = {"porosity": _soil_array("porosity", np.full((5, 1, 1), 0.45, dtype=np.float32))}
    recorded = {}

    monkeypatch.setattr(preprocess_module, "process_precipitation", lambda args, grid, start, end: rain)
    monkeypatch.setattr(
        preprocess_module,
        "compute_effective_precipitation_smith",
        lambda raw_rain, dtype: rain.rename("effective_precipitation"),
    )
    monkeypatch.setattr(preprocess_module, "process_et", lambda args, grid, dates: {"et": et})
    monkeypatch.setattr(
        preprocess_module.soil_api,
        "load_soil_properties",
        lambda *, soil_source, args, grid, reproject_to_template: recorded.update(
            {
                "soil_source": soil_source,
                "args": args,
                "grid": grid,
                "reproject_to_template": reproject_to_template,
            }
        )
        or SoilOutputs(
            arrays=soil_arrays,
            layer_bottoms_mm=np.array([50.0, 150.0, 300.0, 600.0, 1000.0], dtype=float),
        ),
    )

    preprocess_inputs(
        date_range=["2024-01-01", "2024-01-02"],
        extent=[148.0, -35.1, 148.1, -35.0],
        sm_res=0.1,
        output_dir=str(tmp_path),
        skip_reference_ssm=True,
        workers=1,
    )

    assert recorded["soil_source"] == "openlandmap"
    assert recorded["args"].soil_source == "openlandmap"
    assert recorded["args"].extent == [148.0, -35.1, 148.1, -35.0]
    assert recorded["grid"].lat_dim == "lat"
    assert recorded["grid"].lon_dim == "lon"
    assert recorded["reproject_to_template"] is preprocess_module._reproject_to_template
