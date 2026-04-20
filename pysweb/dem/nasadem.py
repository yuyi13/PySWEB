#!/usr/bin/env python3
"""
Script: nasadem.py
Objective: Prepare NASADEM elevation rasters for SSEBop inputs via Earth Engine download requests.
Author: Yi Yu
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: Earth Engine project name, geographic extent, and a local GeoTIFF output path.
Outputs: A non-empty NASADEM GeoTIFF on disk at the requested output path.
Usage: Imported via `pysweb.dem.nasadem`
Dependencies: pathlib, earthengine-api, requests
"""
from __future__ import annotations

from pathlib import Path

import requests

try:
    import ee
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without EE
    ee = None

NASADEM_IMAGE_ID = "NASA/NASADEM_HGT/001"
NASADEM_BAND = "elevation"
DOWNLOAD_TIMEOUT_SECONDS = 300


def _require_ee():
    if ee is None:
        raise ModuleNotFoundError("earthengine-api is required for NASADEM preparation.")
    return ee


def prepare_dem(*, gee_project: str, extent: list[float], output_path: str) -> str:
    ee_module = _require_ee()
    try:
        ee_module.Initialize(project = gee_project)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize Earth Engine for NASADEM backend with gee_project "
            f"'{gee_project}'."
        ) from exc

    region = ee_module.Geometry.Rectangle(extent, proj = "EPSG:4326", geodesic = False)
    image = ee_module.Image(NASADEM_IMAGE_ID).select(NASADEM_BAND).clip(region)
    url = image.getDownloadURL(
        {
            "name": "nasadem",
            "region": region,
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
            "filePerBand": False,
        }
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents = True, exist_ok = True)

    with requests.get(url, timeout = DOWNLOAD_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        payload = response.content

    if not payload:
        raise RuntimeError("Earth Engine NASADEM download returned an empty output.")

    output_file.write_bytes(payload)
    if output_file.stat().st_size == 0:
        raise RuntimeError("Earth Engine NASADEM download wrote an empty output.")

    return str(output_file)
