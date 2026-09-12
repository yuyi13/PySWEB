"""
Script: mlcons.py
Objective: Prepare MLConstraints user soil maps for canonical SWB preprocessing.
Author: Yi Yu (with assistance from Codex)
Created: 2026-04-19
Last updated: 2026-09-12
Inputs: In-memory arrays and explicit workflow configuration.
Outputs: Validated data and numerical diagnostics.
Usage: Imported by pysweb workflows.
Dependencies: numpy, xarray, rasterio, rioxarray; optional Rscript with raster/terra for RDS
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence, List, Dict
import numpy as np
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
from pysweb.soil.api import SoilOutputs
from pysweb.contracts import layer_bottoms

_MLCONS_SOIL_FILES = {
    "clay": "LegendModel_2.0.1_Clay_4layers.tif",
    "sand": "LegendModel_2.0.1_Sand_4layers.tif",
    "organic_carbon": "LegendModel_2.0.1_OC_4layers.tif",
}
_MLCONS_RDS_FILE = "LegendModel_2.0.1_output.RDS"
_DEFAULT_MLCONS_LAYER_BOTTOMS_MM = (150.0, 300.0, 600.0, 1000.0)


def _print_progress(label, index, total):
    print(f"{label}: {index}/{total}", flush=True)


def _load_soil_raster(path, grid, band):
    from pysweb.swb.preprocess import _reproject_to_template

    with rioxarray.open_rasterio(path, masked=True) as source:
        if source.rio.crs is None:
            raise ValueError(f"MLConstraints raster requires CRS metadata: {path}")
        if band > source.sizes["band"]:
            raise ValueError(f"Missing soil layer {band} in {path}")
        da = source.isel(band=band - 1, drop=True).load()
    return _reproject_to_template(da, grid, resampling=Resampling.bilinear).values


def _prepare_mlcons_soil_directory(soil_dir, bottoms):
    # Extraction is explicit in load_soil_properties and never overwrites user inputs.
    for filename in _MLCONS_SOIL_FILES.values():
        if not (soil_dir / filename).exists():
            raise FileNotFoundError(soil_dir / filename)
        import rasterio

        with rasterio.open(soil_dir / filename) as source:
            if source.count != len(bottoms):
                raise ValueError(
                    f"MLConstraints band count must match layer bottoms: {filename}"
                )
    return soil_dir


def _format_depth_cm(depth_mm: float) -> str:
    depth_cm = float(depth_mm) / 10.0
    rounded = round(depth_cm)
    if np.isclose(depth_cm, rounded):
        return str(int(rounded))
    return f"{depth_cm:g}"


def _build_layer_descriptions(layer_bottoms_mm: Sequence[float]) -> List[str]:
    bottoms = np.asarray(layer_bottoms_mm, dtype=float)
    descriptions: List[str] = []
    top_mm = 0.0
    for idx, bottom_mm in enumerate(bottoms, start=1):
        descriptions.append(
            f"Layer {idx} ({_format_depth_cm(top_mm)}-{_format_depth_cm(bottom_mm)} cm)"
        )
        top_mm = float(bottom_mm)
    return descriptions


def _has_expected_band_descriptions(path: Path, descriptions: Sequence[str]) -> bool:
    expected = tuple(descriptions)
    import rasterio

    try:
        with rasterio.open(path) as ds:
            if ds.count != len(expected):
                return False
            current = tuple("" if desc is None else desc for desc in ds.descriptions)
    except Exception:
        return False
    return current == expected


def _extract_mlcons_geotiffs_from_rds(
    rds_path: Path,
    output_dir: Path,
    layer_descriptions: Sequence[str],
) -> None:
    desc_joined = "\t".join(layer_descriptions)
    r_script = r"""
args <- commandArgs(trailingOnly = TRUE)
rds_path <- args[[1]]
output_dir <- args[[2]]
desc <- if (length(args) >= 3) strsplit(args[[3]], "\t", fixed = TRUE)[[1]] else character(0)
suppressPackageStartupMessages(library(raster))
suppressPackageStartupMessages(library(terra))
obj <- readRDS(rds_path)
if (!is.list(obj)) {
  stop("MLConstraints RDS root object must be a list.")
}
targets <- list(
  Clay = "LegendModel_2.0.1_Clay_4layers.tif",
  Sand = "LegendModel_2.0.1_Sand_4layers.tif",
  OC = "LegendModel_2.0.1_OC_4layers.tif"
)
for (key in names(targets)) {
  if (!(key %in% names(obj))) {
    stop(sprintf("MLConstraints RDS missing '%s' entry.", key))
  }
  item <- obj[[key]]
  if (!is.list(item) || is.null(item[["Maps"]])) {
    stop(sprintf("MLConstraints RDS entry '%s' missing 'Maps'.", key))
  }
  maps <- item[["Maps"]]
  if (inherits(maps, "SpatRaster")) {
    spat <- maps
  } else if (inherits(maps, c("RasterLayer", "RasterStack", "RasterBrick"))) {
    spat <- terra::rast(maps)
  } else {
    stop(sprintf("MLConstraints entry '%s$Maps' is not a raster object.", key))
  }
  if (length(desc) > 0) {
    if (terra::nlyr(spat) != length(desc)) {
      stop(sprintf(
        "Layer description count (%d) does not match raster layer count (%d) for '%s'.",
        length(desc), terra::nlyr(spat), key
      ))
    }
    names(spat) <- desc
  }
  out_path <- file.path(output_dir, targets[[key]])
  terra::writeRaster(spat, filename = out_path, filetype = "GTiff", overwrite = TRUE)
  cat(sprintf("Wrote %s\n", out_path))
}
"""
    cmd = ["Rscript", "-e", r_script, str(rds_path), str(output_dir), desc_joined]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Rscript is required to extract MLConstraints GeoTIFFs from RDS, but it was not found."
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            f"Failed to extract MLConstraints GeoTIFFs from {rds_path}: {details}"
        ) from exc

    stdout = completed.stdout.strip()
    if stdout:
        print(stdout, flush=True)


def process_soil_properties(args, grid) -> Dict[str, xr.DataArray]:
    soil_dir = Path(args.soil_mlcons_dir).expanduser().resolve()
    if not soil_dir.exists():
        raise FileNotFoundError(
            f"MLConstraints soil directory does not exist: {soil_dir}"
        )

    layer_bottoms_mm = np.asarray(
        args.soil_layer_bottoms_mm or list(_DEFAULT_MLCONS_LAYER_BOTTOMS_MM),
        dtype=float,
    )
    if layer_bottoms_mm.ndim != 1 or layer_bottoms_mm.size == 0:
        raise ValueError(
            "MLConstraints layer-bottom list must be a non-empty 1-D sequence."
        )
    if not np.all(np.isfinite(layer_bottoms_mm)):
        raise ValueError("MLConstraints layer bottoms contain non-finite values.")
    if np.any(layer_bottoms_mm <= 0.0):
        raise ValueError("MLConstraints layer bottoms must be > 0 mm.")
    if np.any(np.diff(layer_bottoms_mm) <= 0.0):
        raise ValueError("MLConstraints layer bottoms must be strictly increasing.")
    soil_dir = _prepare_mlcons_soil_directory(soil_dir, layer_bottoms_mm)

    clay_path = soil_dir / _MLCONS_SOIL_FILES["clay"]
    sand_path = soil_dir / _MLCONS_SOIL_FILES["sand"]
    oc_path = soil_dir / _MLCONS_SOIL_FILES["organic_carbon"]

    porosity_layers = []
    wilting_layers = []
    awc_layers = []
    b_layers = []
    ksat_layers = []
    layer_depth_mm = []

    total_layers = layer_bottoms_mm.size
    for idx, bottom in enumerate(layer_bottoms_mm, start=1):
        _print_progress("Soil layers", idx, total_layers)

        # MLConstraints clay/sand rasters are stored in percentage (0-100); convert to fractions.
        clay = _load_soil_raster(clay_path, grid, band=idx) * 0.01
        sand = _load_soil_raster(sand_path, grid, band=idx) * 0.01
        organic_carbon = _load_soil_raster(oc_path, grid, band=idx)

        om = 1.72 * organic_carbon * 0.01

        theta_33t = (
            -0.251 * sand
            + 0.195 * clay
            + 0.011 * om
            + 0.006 * sand * om
            - 0.027 * clay * om
            + 0.452 * sand * clay
            + 0.299
        )
        theta_33 = theta_33t + (1.283 * theta_33t**2 - 0.374 * theta_33t - 0.015)

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
            b_coeff = (np.log(1500.0) - np.log(33.0)) / (
                np.log(theta_33) - np.log(theta_1500)
            )
            lambda_coeff = 1.0 / b_coeff
            ksat = 1930.0 * np.power(theta_s - theta_33, 3.0 - lambda_coeff)

        wilting_point = theta_1500
        porosity = theta_s
        available_water = np.clip(theta_33 - wilting_point, a_min=0.0, a_max=None)

        porosity_layers.append(porosity)
        wilting_layers.append(wilting_point)
        awc_layers.append(available_water)
        b_layers.append(b_coeff)
        ksat_layers.append(ksat * 24.0)

        layer_depth_mm.append(float(bottom))

    porosity = np.stack(porosity_layers, axis=0)
    wilting = np.stack(wilting_layers, axis=0)
    awc = np.stack(awc_layers, axis=0)
    b_coeff = np.stack(b_layers, axis=0)
    ksat = np.stack(ksat_layers, axis=0)

    layer_depth_mm = np.asarray(layer_depth_mm, dtype=float)
    layer_ids = np.arange(1, layer_depth_mm.size + 1, dtype=int)

    def _to_da(values: np.ndarray, name: str, attrs: Dict[str, str]) -> xr.DataArray:
        da = xr.DataArray(
            values.astype(args.dtype, copy=False),
            dims=("layer", grid.lat_dim, grid.lon_dim),
            coords={
                "layer": layer_ids,
                "layer_depth": ("layer", layer_depth_mm),
                grid.lat_dim: grid.latitudes,
                grid.lon_dim: grid.longitudes,
            },
            name=name,
            attrs={
                **attrs,
                "layer_bottoms_mm": layer_depth_mm.tolist(),
                "soil_source": str(soil_dir),
                "soil_backend": "mlcons",
                "pedotransfer": "historical PySWEB equations; OM input is a fraction",
            },
        )
        return da

    soil_arrays: Dict[str, xr.DataArray] = {
        "porosity": _to_da(
            porosity,
            "porosity",
            {"long_name": "Soil porosity", "units": "m3 m-3"},
        ),
        "wilting_point": _to_da(
            wilting,
            "wilting_point",
            {"long_name": "Wilting point volumetric water content", "units": "m3 m-3"},
        ),
        "available_water_capacity": _to_da(
            awc,
            "available_water_capacity",
            {"long_name": "Available water capacity", "units": "m3 m-3"},
        ),
        "b_coefficient": _to_da(
            b_coeff,
            "b_coefficient",
            {"long_name": "Campbell b coefficient", "units": "dimensionless"},
        ),
        "conductivity_sat": _to_da(
            ksat,
            "conductivity_sat",
            {"long_name": "Saturated hydraulic conductivity", "units": "mm day-1"},
        ),
    }

    return soil_arrays


def load_soil_properties(*, args, grid, **kwargs):
    from argparse import Namespace

    source = getattr(args, "soil_mlcons_dir", None)
    if not source:
        raise ValueError("soil_mlcons_dir is required for the MLConstraints backend.")
    source = Path(source).expanduser().resolve()
    bottoms = layer_bottoms(
        getattr(args, "soil_layer_bottoms_mm", None) or _DEFAULT_MLCONS_LAYER_BOTTOMS_MM
    )
    prepared = source
    if not all((source / f).is_file() for f in _MLCONS_SOIL_FILES.values()):
        rds = source / _MLCONS_RDS_FILE
        if not rds.is_file():
            raise FileNotFoundError(
                f"Expected MLConstraints GeoTIFFs or {rds}; select a specific site directory."
            )
        prepared = Path(args.output_dir) / "mlcons_rasters"
        prepared.mkdir(parents=True, exist_ok=True)
        _extract_mlcons_geotiffs_from_rds(
            rds, prepared, _build_layer_descriptions(bottoms)
        )
    copied = Namespace(**vars(args))
    copied.soil_mlcons_dir = str(prepared)
    copied.soil_layer_bottoms_mm = bottoms.tolist()
    arrays = process_soil_properties(copied, grid)
    for da in arrays.values():
        da.attrs["soil_source"] = str(source)
    return SoilOutputs(arrays=arrays, layer_bottoms_mm=bottoms)
