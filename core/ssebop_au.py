"""Australia-focused SSEBop helpers for SWEB integration.

This module documents AU-friendly data sources and provides geospatial
helpers for key SSEBop steps (dT, ETf, and daily ET) using xarray/rioxarray.
It avoids hard dependencies on Earth Engine so it can be used with locally
prepared gridded datasets and site time series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds


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


def _ensure_spatial_dims(data_array: xr.DataArray) -> xr.DataArray:
    """Ensure spatial dims are named and CRS metadata is present."""
    if {"x", "y"}.issubset(set(data_array.dims)):
        return data_array
    if {"lon", "lat"}.issubset(set(data_array.dims)):
        data_array = data_array.rename({"lon": "x", "lat": "y"})
    return data_array


def open_silo_et_short_crop(file_path: str, variable: str = "et_short_crop") -> xr.DataArray:
    """Open SILO FAO56 ETo NetCDF and return the ETo DataArray."""
    ds = xr.open_dataset(file_path)
    if variable not in ds:
        raise ValueError(f"Variable '{variable}' not found in {file_path}")
    da = _ensure_spatial_dims(ds[variable])
    return da


def open_silo_variable(file_path: str, variable: str) -> xr.DataArray:
    """Open a SILO NetCDF and return the requested variable as a DataArray."""
    ds = xr.open_dataset(file_path)
    if variable not in ds:
        raise ValueError(f"Variable '{variable}' not found in {file_path}")
    return _ensure_spatial_dims(ds[variable])


def load_worldcover_landcover(
    path: Optional[str] = None,
    masked: bool = True,
) -> xr.DataArray:
    """Load ESA WorldCover (v200) GeoTIFF and return landcover classes."""
    lc_path = path or AU_SSEBOP_SOURCE_CANDIDATES["landcover"]["local"]
    lc = rioxarray.open_rasterio(lc_path, masked=masked).squeeze("band", drop=True)
    return _ensure_spatial_dims(lc)


def worldcover_masks(
    landcover: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Build ag, anomalous, and water masks from WorldCover classes.

    ESA WorldCover v200 classes:
    10 Tree, 20 Shrub, 30 Grass, 40 Cropland, 50 Built-up, 60 Bare,
    70 Snow/Ice, 80 Permanent Water, 90 Herbaceous Wetland, 95 Mangroves,
    100 Moss/Lichen.
    """
    ag_mask = landcover.isin([30, 40, 90]).astype("uint8")
    anomalous_mask = landcover.isin([20, 50, 60]).astype("uint8")
    water_mask = landcover.isin([80]).astype("uint8")
    return ag_mask, anomalous_mask, water_mask


def reproject_match(
    source: xr.DataArray,
    match: xr.DataArray,
    resampling: str = "nearest",
) -> xr.DataArray:
    """Reproject a DataArray to match another grid."""
    resampling_enum = Resampling[resampling]
    return source.rio.reproject_match(match, resampling=resampling_enum)


def reproject_match_crop_first(
    source: xr.DataArray,
    match: xr.DataArray,
    resampling: str = "nearest",
    buffer: float = 0.0,
) -> xr.DataArray:
    """Crop to the match bounds (plus buffer), then reproject to match."""
    if source.rio.crs is None:
        raise ValueError("source must have a CRS")
    if match.rio.crs is None:
        raise ValueError("match must have a CRS")

    left, bottom, right, top = match.rio.bounds()
    if buffer:
        left -= buffer
        bottom -= buffer
        right += buffer
        top += buffer

    src_bounds = transform_bounds(
        match.rio.crs,
        source.rio.crs,
        left,
        bottom,
        right,
        top,
        densify_pts=21,
    )
    try:
        clipped = source.rio.clip_box(*src_bounds, allow_one_dimensional_raster=True)
    except Exception:
        clipped = source
    return reproject_match(clipped, match, resampling=resampling)


def dt_fao56_xr(
    tmax_k: xr.DataArray,
    tmin_k: xr.DataArray,
    elev_m: xr.DataArray,
    doy: xr.DataArray | int,
    lat_deg: xr.DataArray,
    rs_mj_m2_day: Optional[xr.DataArray] = None,
    ea_kpa: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Compute dT using the SSEBop FAO56-based formulation (xarray).

    Inputs are in SI units (Kelvin for temperature).
    """
    if doy is None:
        raise ValueError("doy is required for dT calculation")

    phi = np.deg2rad(lat_deg)
    if isinstance(doy, xr.DataArray):
        doy_arr = doy
    else:
        doy_arr = xr.zeros_like(tmax_k) + float(doy)

    delta = np.sin((2 * np.pi / 365) * doy_arr - 1.39) * 0.409
    ws = np.arccos(-np.tan(phi) * np.tan(delta))
    dr = np.cos((2 * np.pi / 365) * doy_arr) * 0.033 + 1.0
    ra = (
        (ws * np.sin(phi) * np.sin(delta) +
         np.cos(phi) * np.cos(delta) * np.sin(ws))
        * dr * ((1367.0 / np.pi) * 0.0820)
    )
    rso = (0.75 + 2e-5 * elev_m) * ra

    if rs_mj_m2_day is None:
        rs = rso
        fcd = 1.0
    else:
        rs = rs_mj_m2_day
        fcd = np.clip(rs / rso, 0.3, 1.0) * 1.35 - 0.35

    rns = rs * (1 - 0.23)

    if ea_kpa is None:
        ea = np.exp((17.27 * (tmin_k - 273.15)) / ((tmin_k - 273.15) + 237.3)) * 0.6108
    else:
        ea = ea_kpa

    rnl = (
        (tmax_k ** 4 + tmin_k ** 4)
        * (np.sqrt(ea) * -0.14 + 0.34)
        * (4.901e-9 * 0.5) * fcd
    )
    rn = rns - rnl

    pair = (101.3 * ((293.0 - 0.0065 * elev_m) / 293.0) ** 5.26)
    den = (pair * (3.486 / 1.01)) / ((tmax_k + tmin_k) * 0.5)

    return rn / den * (110.0 / ((1.013 / 1000) * 86400))


def et_fraction_xr(
    lst_k: xr.DataArray,
    tcold_k: xr.DataArray,
    dt_k: xr.DataArray,
    clamp_max: float = 1.0,
    mask_max: float = 2.0,
) -> xr.DataArray:
    """Compute SSEBop ET fraction from LST, Tcold, and dT."""
    etf = (tcold_k + dt_k - lst_k) / dt_k
    etf = etf.where(etf <= mask_max)
    return etf.clip(0.0, clamp_max)


def tcold_fano_simple_xr(
    lst_k: xr.DataArray,
    ndvi: xr.DataArray,
    dt_k: xr.DataArray,
    config: Optional[SsebopAuConfig] = None,
) -> xr.DataArray:
    """Simplified FANO-style Tcold estimate for gridded data."""
    cfg = config or SsebopAuConfig()
    tc = lst_k - (cfg.dt_coeff * dt_k * (cfg.high_ndvi_threshold - ndvi) * 10.0)
    return tc


def build_doy_climatology(data: xr.DataArray) -> xr.DataArray:
    """Build a day-of-year climatology from a daily DataArray."""
    if "time" not in data.dims:
        raise ValueError("Input data must have a time dimension")
    return data.groupby("time.dayofyear").mean("time", skipna=True)


def compute_dt_daily(
    tmax_k: xr.DataArray,
    tmin_k: xr.DataArray,
    elev_m: xr.DataArray,
    lat_deg: xr.DataArray,
    rs_mj_m2_day: Optional[xr.DataArray] = None,
    ea_kpa: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Compute daily dT for a time series using the time coordinate DOY."""
    if "time" not in tmax_k.dims:
        raise ValueError("tmax_k must have a time dimension")
    doy = tmax_k["time"].dt.dayofyear
    doy = xr.DataArray(doy, coords={"time": tmax_k["time"]}, dims=("time",))
    return dt_fao56_xr(
        tmax_k=tmax_k,
        tmin_k=tmin_k,
        elev_m=elev_m,
        doy=doy,
        lat_deg=lat_deg,
        rs_mj_m2_day=rs_mj_m2_day,
        ea_kpa=ea_kpa,
    )


def daily_et_from_etf(
    etf_series: pd.Series,
    et_reference_series: pd.Series,
    max_gap_days: int = 32,
) -> pd.Series:
    """Interpolate ETf to daily and compute ETa (ETf * reference ET).

    The ETf series should be on Landsat overpass dates. The reference ET
    series should be daily (ETr or ETo) over the target period.
    """
    if etf_series.empty or et_reference_series.empty:
        raise ValueError("etf_series and et_reference_series must be non-empty")

    etf = etf_series.sort_index()
    target_index = et_reference_series.sort_index().index
    etf = etf.reindex(target_index.union(etf.index))
    etf_interp = etf.interpolate(method="time")

    prev_obs = etf_interp.index.to_series().where(etf.notna()).ffill()
    next_obs = etf_interp.index.to_series().where(etf.notna()).bfill()
    gap_days = (next_obs - prev_obs).dt.days
    etf_interp = etf_interp.where(gap_days <= max_gap_days)

    etf_daily = etf_interp.reindex(target_index)
    eta = etf_daily * et_reference_series
    return eta.rename("et_ssebop")
