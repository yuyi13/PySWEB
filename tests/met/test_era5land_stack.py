#!/usr/bin/env python3
"""
Script: test_era5land_stack.py
Objective: Verify the package ERA5-Land daily stacker writes the required NetCDF products from daily GeoTIFF inputs.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-05-11
Inputs: Synthetic daily GeoTIFFs, DEM rasters, and temporary output directories.
Outputs: Test assertions.
Usage: pytest tests/met/test_era5land_stack.py
Dependencies: numpy, pytest, rasterio, xarray
"""
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import xarray as xr

from pysweb.met.era5land import stack as era5land_stack
from pysweb.met.era5land.stack import stack_era5land_daily_inputs


BAND_VALUES = {
    "temperature_2m_min": 289.15,
    "temperature_2m_max": 304.15,
    "dewpoint_temperature_2m": 289.75,
    "u_component_of_wind_10m": 3.2,
    "v_component_of_wind_10m": 0.0,
    "surface_solar_radiation_downwards_sum": 24000000.0,
    "total_precipitation_sum": 0.012,
}


def _write_daily_geotiff(path: Path):
    transform = from_origin(147.0, -34.5, 1.0, 1.0)
    profile = {
        "driver": "GTiff",
        "height": 1,
        "width": 1,
        "count": len(BAND_VALUES),
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": None,
    }
    with rasterio.open(path, "w", **profile) as dst:
        for index, (band_name, value) in enumerate(BAND_VALUES.items(), start=1):
            dst.write(np.array([[value]], dtype=np.float32), index)
        dst.descriptions = tuple(BAND_VALUES.keys())


def _write_dem(path: Path):
    transform = from_origin(147.0, -34.5, 1.0, 1.0)
    profile = {
        "driver": "GTiff",
        "height": 1,
        "width": 1,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": None,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.array([[180.0]], dtype=np.float32), 1)


def _write_int16_dem(path: Path):
    transform = from_origin(147.0, -34.5, 1.0, 1.0)
    profile = {
        "driver": "GTiff",
        "height": 1,
        "width": 2,
        "count": 1,
        "dtype": "int16",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -32768,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.array([[180, -32768]], dtype=np.int16), 1)


def test_stack_reader_handles_int16_dem_nodata_as_nan(tmp_path):
    from pysweb.met.era5land.stack import _read_grid

    dem_path = tmp_path / "dem_int16.tif"
    _write_int16_dem(dem_path)

    dem_array, *_ = _read_grid(dem_path)

    assert dem_array.dtype == float
    assert dem_array[0, 0] == pytest.approx(180.0)
    assert np.isnan(dem_array[0, 1])


def test_stacker_writes_expected_daily_netcdfs(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    out_dir.mkdir()

    for date_str in ["2024-01-01", "2024-01-02", "2024-01-03"]:
        _write_daily_geotiff(raw_dir / f"ERA5LandDaily_{date_str}.tif")

    dem_path = tmp_path / "dem.tif"
    _write_dem(dem_path)

    stack_era5land_daily_inputs(
        raw_dir=raw_dir,
        dem=dem_path,
        start_date="2024-01-02",
        end_date="2024-01-03",
        output_dir=out_dir,
    )

    expected_files = {
        "precipitation": out_dir / "precipitation_daily_2024-01-02_2024-01-03.nc",
        "tmax": out_dir / "tmax_daily_2024-01-02_2024-01-03.nc",
        "tmin": out_dir / "tmin_daily_2024-01-02_2024-01-03.nc",
        "rs": out_dir / "rs_daily_2024-01-02_2024-01-03.nc",
        "ea": out_dir / "ea_daily_2024-01-02_2024-01-03.nc",
        "et_short_crop": out_dir / "et_short_crop_daily_2024-01-02_2024-01-03.nc",
    }

    for path in expected_files.values():
        assert path.exists()

    with xr.open_dataset(expected_files["precipitation"]) as ds_precip:
        assert list(ds_precip.data_vars) == ["precipitation"]
        assert ds_precip["precipitation"].sizes["time"] == 2
        np.testing.assert_allclose(ds_precip["precipitation"].values[:, 0, 0], [12.0, 12.0], atol=1e-6)

    with xr.open_dataset(expected_files["tmax"]) as ds_tmax:
        assert list(ds_tmax.data_vars) == ["tmax"]
        np.testing.assert_allclose(ds_tmax["tmax"].values[:, 0, 0], [31.0, 31.0], atol=1e-4)

    with xr.open_dataset(expected_files["tmin"]) as ds_tmin:
        assert list(ds_tmin.data_vars) == ["tmin"]
        np.testing.assert_allclose(ds_tmin["tmin"].values[:, 0, 0], [16.0, 16.0], atol=1e-4)

    with xr.open_dataset(expected_files["rs"]) as ds_rs:
        assert list(ds_rs.data_vars) == ["rs"]
        np.testing.assert_allclose(ds_rs["rs"].values[:, 0, 0], [24.0, 24.0], atol=1e-6)

    with xr.open_dataset(expected_files["ea"]) as ds_ea:
        assert list(ds_ea.data_vars) == ["ea"]
        np.testing.assert_allclose(ds_ea["ea"].values[:, 0, 0], [1.9, 1.9], atol=0.02)

    with xr.open_dataset(expected_files["et_short_crop"]) as ds_eto:
        assert list(ds_eto.data_vars) == ["et_short_crop"]
        eto_values = ds_eto["et_short_crop"].values[:, 0, 0]
        assert eto_values[0] == pytest.approx(6.0, abs=0.5)
        assert eto_values[1] == pytest.approx(6.0, abs=0.5)


def test_stacker_requires_complete_requested_date_range(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    out_dir.mkdir()

    for date_str in ["2024-01-01", "2024-01-03"]:
        _write_daily_geotiff(raw_dir / f"ERA5LandDaily_{date_str}.tif")

    dem_path = tmp_path / "dem.tif"
    _write_dem(dem_path)

    with pytest.raises(ValueError, match="do not cover every date in the range"):
        stack_era5land_daily_inputs(
            raw_dir=raw_dir,
            dem=dem_path,
            start_date="2024-01-01",
            end_date="2024-01-03",
            output_dir=out_dir,
        )


def test_write_netcdf_uses_scipy_netcdf3_backend(monkeypatch, tmp_path):
    calls = {}

    def fake_to_netcdf(self, path, **kwargs):
        calls["path"] = Path(path)
        calls["kwargs"] = kwargs
        calls["path"].write_bytes(b"CDF\001")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fake_to_netcdf)
    monkeypatch.setattr(era5land_stack, "_validate_netcdf", lambda path, var_name: None, raising=False)

    era5land_stack._write_netcdf(
        tmp_path / "tmax.nc",
        "tmax",
        np.ones((1, 1, 1), dtype=np.float32),
        np.array(["2024-01-01"], dtype="datetime64[D]"),
        np.array([-35.0]),
        np.array([149.0]),
        "degree_Celsius",
        "Daily maximum air temperature",
    )

    assert calls["path"].name.startswith(".tmax.nc.")
    assert calls["kwargs"] == {"engine": "scipy", "format": "NETCDF3_64BIT"}


def test_write_netcdf_does_not_leave_zero_byte_output_after_failed_write(monkeypatch, tmp_path):
    output_path = tmp_path / "tmax.nc"

    def fail_to_netcdf(self, path, **kwargs):
        Path(path).write_bytes(b"")
        raise RuntimeError("simulated backend failure")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fail_to_netcdf)

    with pytest.raises(RuntimeError, match="simulated backend failure"):
        era5land_stack._write_netcdf(
            output_path,
            "tmax",
            np.ones((1, 1, 1), dtype=np.float32),
            np.array(["2024-01-01"], dtype="datetime64[D]"),
            np.array([-35.0]),
            np.array([149.0]),
            "degree_Celsius",
            "Daily maximum air temperature",
        )

    assert not output_path.exists()
    assert not list(tmp_path.glob(".tmax.nc.*.tmp"))
