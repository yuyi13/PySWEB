"""
Script: custom.py
Objective: Load user-supplied hydraulic properties onto the canonical SWB grid.
Author: Yi Yu (with assistance from Codex)
Created: 2026-04-19
Last updated: 2026-09-12
Inputs: User hydraulic NetCDF files, CRS, units, layer depths and target grid.
Outputs: Aligned and validated SoilOutputs for the shared SWB solver.
Usage: Imported by pysweb workflows.
Dependencies: numpy, xarray, rioxarray, rasterio
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
from pysweb.soil.api import SoilOutputs
from pysweb.swb.core import SOIL_FILE_STEMS, infer_layer_bottoms, ensure_matching_grid
from pysweb.contracts import layer_bottoms

UNITS = {
    "porosity": "m3 m-3",
    "wilting_point": "m3 m-3",
    "available_water_capacity": "m3 m-3",
    "b_coefficient": "dimensionless",
    "conductivity_sat": "mm day-1",
}


def validate_hydraulics(arrays):
    values = {key: np.asarray(da.values) for key, da in arrays.items()}
    valid = np.logical_and.reduce([np.isfinite(a) for a in values.values()])
    if not np.any(valid):
        raise ValueError("No valid custom soil values remain on the target grid.")
    por, wp, awc, b, k = [values[n] for n in UNITS]
    bad = valid & (
        (por <= 0)
        | (por > 1)
        | (wp < 0)
        | (wp >= por)
        | (awc < 0)
        | (awc > por - wp)
        | (b <= 0)
        | (k < 0)
    )
    if np.any(bad):
        raise ValueError(
            "Invalid hydraulic values: require 0 <= wilting_point < porosity <= 1, "
            "0 <= available_water_capacity <= porosity-wilting_point, b > 0 and Ksat >= 0."
        )
    return {key: da.where(valid) for key, da in arrays.items()}


def load_soil_properties(*, args, grid, reproject_to_template, **kwargs):
    source_file = getattr(args, "soil_file", None)
    source_dir = getattr(args, "soil_input_dir", None)
    if bool(source_file) == bool(source_dir):
        raise ValueError(
            "Custom soil requires exactly one of soil_file or soil_input_dir."
        )
    arrays = {}
    for name, filename in SOIL_FILE_STEMS.items():
        path = Path(source_file) if source_file else Path(source_dir) / filename
        with xr.open_dataset(path, decode_coords="all") as ds:
            if name not in ds:
                raise ValueError(
                    f"Custom soil input {path} must contain variable '{name}'."
                )
            da = ds[name].load()
        unit = str(da.attrs.get("units", "")).strip()
        accepted = {UNITS[name]}
        if name == "b_coefficient":
            accepted.add("1")
        if unit not in accepted:
            raise ValueError(
                f"Custom {name} requires units {UNITS[name]!r}; got {unit!r}."
            )
        if "layer" not in da.dims:
            raise ValueError(f"Custom {name} requires an explicit layer dimension.")
        arrays[name] = da
    bottoms = layer_bottoms(
        infer_layer_bottoms(arrays, getattr(args, "soil_layer_bottoms_mm", None))
    )
    if not getattr(args, "soil_layer_bottoms_mm", None) and not any(
        any(
            k in da.attrs
            for k in ["layer_bottoms_mm", "layer_depth_mm", "layer_depths_mm"]
        )
        or "layer_depth" in da.coords
        for da in arrays.values()
    ):
        raise ValueError(
            "Custom soil requires layer_bottoms_mm metadata or soil_layer_bottoms_mm."
        )
    aligned = {}
    for name, da in arrays.items():
        if da.sizes["layer"] != len(bottoms):
            raise ValueError(f"Custom {name} layer count disagrees with layer depths.")
        own_depth = infer_layer_bottoms(
            {name: da}, getattr(args, "soil_layer_bottoms_mm", None)
        )
        if not np.array_equal(own_depth, bottoms):
            raise ValueError(
                "Custom hydraulic variables must share the same layer depths."
            )
        source_crs = da.rio.crs
        if source_crs is None:
            raise ValueError(
                f"Custom {name} requires CRS metadata (write with rio.write_crs)."
            )
        if (
            str(source_crs) == str(grid.crs)
            and grid.lat_dim in da.dims
            and grid.lon_dim in da.dims
        ):
            try:
                ensure_matching_grid(
                    grid.template, {name: da}, grid.lat_dim, grid.lon_dim
                )
                target = da
            except ValueError:
                target = reproject_to_template(da, grid, resampling=Resampling.bilinear)
        else:
            target = reproject_to_template(da, grid, resampling=Resampling.bilinear)
        target = target.transpose("layer", grid.lat_dim, grid.lon_dim)
        target = target.assign_coords(
            layer=np.arange(1, len(bottoms) + 1), layer_depth=("layer", bottoms)
        )
        target.attrs.update(
            units=UNITS[name],
            layer_bottoms_mm=bottoms.tolist(),
            soil_source=str(Path(source_file or source_dir).resolve()),
            soil_backend="custom",
        )
        aligned[name] = target.astype(args.dtype)
    return SoilOutputs(arrays=validate_hydraulics(aligned), layer_bottoms_mm=bottoms)
