#!/usr/bin/env python3
"""
Script: ssebop_au.py
Objective: Provide Australia-focused SSEBop helper functions for dT climatology, ET fraction, and reprojection.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-04-17
Inputs: xarray datasets/raster grids, Landsat/SILO-derived variables, and geospatial metadata.
Outputs: Processed geospatial arrays including masks, climatologies, ET fraction, and daily ET products.
Usage: Imported by workflows/2_ssebop_run_model.py; not intended as a standalone CLI script.
Dependencies: numpy, pandas, xarray, rioxarray, rasterio
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pysweb.ssebop.core import (
    build_doy_climatology,
    compute_dt_daily,
    daily_et_from_etf,
    dt_fao56_xr,
    et_fraction_xr,
    tcold_fano_simple_xr,
)
from pysweb.ssebop.grid import reproject_match, reproject_match_crop_first
from pysweb.ssebop.landcover import load_worldcover_landcover, worldcover_masks


AU_SSEBOP_SOURCE_CANDIDATES: Dict[str, Dict[str, str]] = {
    # Landsat Collection 2 L2 is global and still the primary LST/NDVI source.
    "landsat_c2_l2": {
        "gee": "LANDSAT/LC08/C02/T1_L2, LANDSAT/LC09/C02/T1_L2, LANDSAT/LE07/C02/T1_L2, LANDSAT/LT05/C02/T1_L2",
        "notes": "Global coverage; use C2 L2 SR/ST for LST, NDVI, NDWI, QA_PIXEL.",
    },
    # Reference ET sources: use SILO FAO56 ETo (et_short_crop).
    "et_reference": {
        "local": "BoM SILO FAO56 ETo (et_short_crop) NetCDF",
        "silo_index": "https://s3-ap-southeast-2.amazonaws.com/silo-open-data/Official/annual/index.html",
        "notes": "ETf from Landsat; ETa = ETf * ETo (short crop).",
    },
    # dT climatology: computed from daily SILO met (tmax/tmin/rs/ea) + DEM.
    "dt_climatology": {
        "local": "BoM SILO daily met + DEM to build DOY dT climatology",
        "notes": "Compute dT via SSEBop FAO56-based formula; store as DOY climatology for speed.",
    },
    # Land cover masks for FANO: use ESA WorldCover local GeoTIFF.
    "landcover": {
        "local": "/g/data/yx97/EO_collections/ESA/WorldCover/ESA_WorldCover_100m_v200.tif",
        "notes": "Map to ag/grass/wetland and anomalous (barren/shrub/urban) masks.",
    },
    # Water mask: optional; extract from WorldCover if needed.
    "water_mask": {
        "notes": "Optional; use WorldCover class 80 (permanent water) if needed.",
    },
}


@dataclass
class SsebopAuConfig:
    """Configuration hints for AU SSEBop processing."""

    et_reference_type: str = "alfalfa"  # or "grass"
    et_reference_unit: str = "mm/day"
    dt_coeff: float = 0.125
    high_ndvi_threshold: float = 0.9
    veg_ndvi_threshold: float = 0.4
    etf_clamp_max: float = 1.0
    etf_mask_max: float = 2.0
    worldcover_path: str = AU_SSEBOP_SOURCE_CANDIDATES["landcover"]["local"]
