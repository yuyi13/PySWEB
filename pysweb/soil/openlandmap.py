#!/usr/bin/env python3
"""
Script: openlandmap.py
Objective: Implement the OpenLandMap soil backend behind the package-level soil dispatcher.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Soil backend arguments, target grid metadata, and a reprojection callback.
Outputs: SoilOutputs dataclass instances with OpenLandMap-derived hydraulic property arrays.
Usage: Imported via `pysweb.soil.openlandmap`
Dependencies: earthengine-api, numpy, requests, rioxarray, rasterio, xarray
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, Dict, Tuple

import ee
import numpy as np
import requests
import rioxarray
import xarray as xr
from rasterio.enums import Resampling

from pysweb.soil.api import SoilOutputs

OPENLANDMAP_LAYER_SPECS = [
    ("b0", 50.0),
    ("b10", 150.0),
    ("b30", 300.0),
    ("b60", 600.0),
    ("b100", 1000.0),
]
OPENLANDMAP_EXPORT_SCALE_M = 250.0
OPENLANDMAP_DATASETS = {
    "clay": "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02",
    "sand": "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02",
    "soc": "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02",
}
OPENLANDMAP_SOC_SCALE = 5.0


def _build_layer_bottoms_mm() -> np.ndarray:
    return np.array([bottom_mm for _, bottom_mm in OPENLANDMAP_LAYER_SPECS], dtype=float)


def _initialize_ee(gee_project: str) -> None:
    try:
        ee.Initialize(project=gee_project)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Earth Engine with project '{gee_project}'.") from exc


def _region_json_from_extent(extent: Tuple[float, float, float, float]) -> Dict[str, object]:
    min_lon, min_lat, max_lon, max_lat = [float(value) for value in extent]
    return {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]],
        "geodesic": False,
        "evenOdd": True,
    }


def _open_downloaded_raster(raw_bytes: bytes, name: str) -> xr.DataArray:
    with tempfile.TemporaryDirectory(prefix=f"{name}_") as temp_dir:
        temp_path = Path(temp_dir) / f"{name}.tif"
        temp_path.write_bytes(raw_bytes)
        try:
            return rioxarray.open_rasterio(temp_path, masked=True).load()
        except Exception:
            if not zipfile.is_zipfile(temp_path):
                raise
            with zipfile.ZipFile(temp_path) as archive:
                tif_members = [member for member in archive.namelist() if member.lower().endswith(".tif")]
                if not tif_members:
                    raise RuntimeError(f"Earth Engine download for '{name}' did not contain a GeoTIFF.")
                extracted_path = Path(temp_dir) / Path(tif_members[0]).name
                extracted_path.write_bytes(archive.read(tif_members[0]))
            return rioxarray.open_rasterio(extracted_path, masked=True).load()


def _download_ee_multiband_image(
    image: ee.Image,
    extent: Tuple[float, float, float, float],
    name: str,
    scale_m: float,
) -> xr.DataArray:
    band_names = image.bandNames().getInfo()
    url = image.getDownloadURL(
        {
            "name": name,
            "region": _region_json_from_extent(extent),
            "crs": "EPSG:4326",
            "scale": scale_m,
            "filePerBand": False,
            "format": "GEO_TIFF",
        }
    )
    with requests.get(url, timeout=300) as response:
        response.raise_for_status()
        da = _open_downloaded_raster(response.content, name)
    if "band" in da.dims:
        da = da.assign_coords(band=np.array(band_names, dtype=object))
    da.name = name
    return da


def _select_openlandmap_bands(image_id: str) -> ee.Image:
    return ee.Image(image_id).select([band_name for band_name, _ in OPENLANDMAP_LAYER_SPECS])


def _load_openlandmap_predictors(
    extent: Tuple[float, float, float, float],
    gee_project: str,
) -> Dict[str, xr.DataArray]:
    _initialize_ee(gee_project)
    region = ee.Geometry.Rectangle(list(extent), proj="EPSG:4326", geodesic=False)
    predictors: Dict[str, xr.DataArray] = {}
    for key, image_id in OPENLANDMAP_DATASETS.items():
        image = _select_openlandmap_bands(image_id).clip(region)
        predictors[key] = _download_ee_multiband_image(
            image,
            extent,
            f"openlandmap_{key}",
            OPENLANDMAP_EXPORT_SCALE_M,
        )
    return predictors


def process_soil_properties_from_openlandmap(
    args: argparse.Namespace,
    grid,
    soil_predictors: Dict[str, xr.DataArray],
    *,
    reproject_to_template: Callable[[xr.DataArray, object, Resampling], xr.DataArray],
) -> Dict[str, xr.DataArray]:
    required = {"clay", "sand", "soc"}
    missing = required.difference(soil_predictors)
    if missing:
        raise KeyError(f"Missing OpenLandMap predictors: {', '.join(sorted(missing))}")

    clay_da = reproject_to_template(soil_predictors["clay"], grid, resampling=Resampling.bilinear)
    sand_da = reproject_to_template(soil_predictors["sand"], grid, resampling=Resampling.bilinear)
    soc_da = reproject_to_template(soil_predictors["soc"], grid, resampling=Resampling.bilinear)

    porosity_layers = []
    wilting_layers = []
    awc_layers = []
    b_layers = []
    ksat_layers = []
    layer_depth_mm = []

    for idx, (_, bottom_mm) in enumerate(OPENLANDMAP_LAYER_SPECS):
        clay = np.asarray(clay_da.isel(band=idx).values, dtype=float) * 0.01
        sand = np.asarray(sand_da.isel(band=idx).values, dtype=float) * 0.01
        soc_g_per_kg = np.asarray(soc_da.isel(band=idx).values, dtype=float) * OPENLANDMAP_SOC_SCALE
        om = 1.72 * soc_g_per_kg / 1000.0

        theta_33t = (
            -0.251 * sand
            + 0.195 * clay
            + 0.011 * om
            + 0.006 * sand * om
            - 0.027 * clay * om
            + 0.452 * sand * clay
            + 0.299
        )
        theta_33 = theta_33t + (1.283 * theta_33t ** 2 - 0.374 * theta_33t - 0.015)

        theta_s_33t = (
            0.278 * sand
            + 0.034 * clay
            + 0.022 * om
            - 0.018 * sand * om
            - 0.027 * clay * om
            - 0.584 * sand * clay
            + 0.078
        )
        theta_s_33 = theta_s_33t + (0.636 * theta_s_33t - 0.107)

        theta_1500t = (
            -0.024 * sand
            + 0.487 * clay
            + 0.006 * om
            + 0.005 * sand * om
            - 0.013 * clay * om
            + 0.068 * sand * clay
            + 0.031
        )
        theta_1500 = theta_1500t + (0.14 * theta_1500t - 0.02)

        theta_s = theta_33 + theta_s_33 - 0.097 * sand + 0.043
        with np.errstate(divide="ignore", invalid="ignore"):
            b_coeff = (np.log(1500.0) - np.log(33.0)) / (np.log(theta_33) - np.log(theta_1500))
            lambda_coeff = 1.0 / b_coeff
            ksat = 1930.0 * np.power(theta_s - theta_33, 3.0 - lambda_coeff)

        porosity_layers.append(theta_s)
        wilting_layers.append(theta_1500)
        awc_layers.append(np.clip(theta_33 - theta_1500, a_min=0.0, a_max=None))
        b_layers.append(b_coeff)
        ksat_layers.append(ksat * 24.0)
        layer_depth_mm.append(bottom_mm)

    layer_depth_mm_arr = np.asarray(layer_depth_mm, dtype=float)
    layer_ids = np.arange(1, layer_depth_mm_arr.size + 1, dtype=int)

    def _to_da(values: np.ndarray, name: str, attrs: Dict[str, str]) -> xr.DataArray:
        return xr.DataArray(
            np.stack(values, axis=0).astype(args.dtype, copy=False),
            dims=("layer", grid.lat_dim, grid.lon_dim),
            coords={
                "layer": layer_ids,
                "layer_depth": ("layer", layer_depth_mm_arr),
                grid.lat_dim: grid.latitudes,
                grid.lon_dim: grid.longitudes,
            },
            name=name,
            attrs={
                **attrs,
                "layer_bottoms_mm": layer_depth_mm_arr.tolist(),
                "soil_source": "OpenLandMap via Google Earth Engine",
            },
        )

    return {
        "porosity": _to_da(porosity_layers, "porosity", {"long_name": "Soil porosity", "units": "m3 m-3"}),
        "wilting_point": _to_da(
            wilting_layers,
            "wilting_point",
            {"long_name": "Wilting point volumetric water content", "units": "m3 m-3"},
        ),
        "available_water_capacity": _to_da(
            awc_layers,
            "available_water_capacity",
            {"long_name": "Available water capacity", "units": "m3 m-3"},
        ),
        "b_coefficient": _to_da(
            b_layers,
            "b_coefficient",
            {"long_name": "Campbell b coefficient", "units": "dimensionless"},
        ),
        "conductivity_sat": _to_da(
            ksat_layers,
            "conductivity_sat",
            {"long_name": "Saturated hydraulic conductivity", "units": "mm day-1"},
        ),
    }


def load_soil_properties(*, args, grid, reproject_to_template, **kwargs) -> SoilOutputs:
    soil_predictors = _load_openlandmap_predictors(tuple(args.extent), args.gee_project)
    soil_arrays = process_soil_properties_from_openlandmap(
        args,
        grid,
        soil_predictors,
        reproject_to_template=reproject_to_template,
    )
    return SoilOutputs(arrays=soil_arrays, layer_bottoms_mm=_build_layer_bottoms_mm())
