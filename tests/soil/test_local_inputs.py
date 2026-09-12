"""
Script: test_local_inputs.py
Objective: Exercise local soil adapters using real NetCDF and GeoTIFF files.
Author: Yi Yu (with assistance from Codex)
Created: 2026-09-12
Last updated: 2026-09-12
Inputs: Synthetic hydraulic NetCDFs, MLConstraints GeoTIFFs and target-grid fixtures.
Outputs: Regression assertions.
Usage: python -m pytest tests/soil/test_local_inputs.py
Dependencies: numpy, pytest, rasterio, xarray; pysweb
"""

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import rasterio
import xarray as xr
from pysweb.demo import prepare_demo
from pysweb.swb.preprocess import (
    _build_args,
    _build_target_grid,
    _reproject_to_template,
)
from pysweb.soil.api import load_soil_properties
from pysweb.soil import mlcons


def context(tmp_path):
    extent = prepare_demo(tmp_path)
    args = _build_args(
        {
            "extent": extent,
            "sm_res": 0.01,
            "soil_file": str(tmp_path / "custom_soil.nc"),
            "soil_source": "custom",
        }
    )
    return args, _build_target_grid(args)


@pytest.mark.parametrize("defect", ["units", "depths", "crs", "range"])
def test_custom_soil_rejects_invalid_contract(tmp_path, defect):
    args, grid = context(tmp_path)
    with xr.open_dataset(args.soil_file, decode_coords="all") as source:
        ds = source.load()
    if defect == "units":
        ds.porosity.attrs["units"] = "percent"
    elif defect == "depths":
        ds.porosity.attrs["layer_bottoms_mm"] = [100.0, 50.0]
    elif defect == "crs":
        ds = ds.drop_vars("spatial_ref")
        for da in ds.data_vars.values():
            da.encoding.pop("grid_mapping", None)
    else:
        ds["wilting_point"][:] = 0.8
    ds.to_netcdf(args.soil_file)
    with pytest.raises(ValueError):
        load_soil_properties(
            soil_source="custom",
            args=args,
            grid=grid,
            reproject_to_template=_reproject_to_template,
        )


def test_mlcons_rasters_use_canonical_soil_contract(tmp_path):
    args, grid = context(tmp_path)
    source = tmp_path / "mlcons"
    source.mkdir()
    for name, filename in mlcons._MLCONS_SOIL_FILES.items():
        value = {"clay": 25.0, "sand": 40.0, "organic_carbon": 2.0}[name]
        with rasterio.open(
            source / filename,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=4,
            dtype="float32",
            crs="EPSG:4326",
            transform=grid.template.rio.transform(),
        ) as dst:
            dst.write(np.full((4, 2, 2), value, dtype="float32"))
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    args.soil_mlcons_dir = str(source)
    args.output_dir = str(tmp_path / "prepared")
    result = load_soil_properties(
        soil_source="mlcons",
        args=args,
        grid=grid,
        reproject_to_template=_reproject_to_template,
    )
    np.testing.assert_equal(result.layer_bottoms_mm, [150, 300, 600, 1000])
    assert set(result.arrays) == {
        "porosity",
        "wilting_point",
        "available_water_capacity",
        "b_coefficient",
        "conductivity_sat",
    }
    for da in result.arrays.values():
        assert da.shape == (4, 2, 2)
        assert np.isfinite(da).all()
        assert da.attrs["soil_backend"] == "mlcons"
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
