#!/usr/bin/env python3
"""
Script: gee_downloader.py
Objective: Download and post-process Google Earth Engine composites used by SSEBop preprocessing workflows, including tiled fallback for oversized requests.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-04-16
Inputs: YAML configuration file, Earth Engine authentication, date/extent/collection settings.
Outputs: Downloaded GeoTIFF composites with standardized band metadata and post-processing updates.
Usage: python core/gee_downloader.py <config.yaml>
Dependencies: earthengine-api, requests, rasterio, numpy, pyyaml, python-dateutil
"""
import math
import os
import shutil
import sys
import time
from datetime import datetime

import ee
import numpy as np
import rasterio
import requests
import yaml
from dateutil.relativedelta import relativedelta
from rasterio.merge import merge as rio_merge

# ----------------------------
# Small utils
# ----------------------------


def _safe_mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)



def _retry_download(url, out_path, max_retries=5, backoff=2.0, chunk=1024 * 1024):
    attempt = 0
    while True:
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for part in r.iter_content(chunk_size=chunk):
                        if part:
                            f.write(part)
            return
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = backoff ** attempt
            print(f"[retry] {attempt}/{max_retries} after error: {e}. Sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)



def _date_str(y, m, d):
    return datetime(y, m, d).strftime("%Y-%m-%d")


EE_DOWNLOAD_MAX_BYTES = 32 * 1024 * 1024
EE_DOWNLOAD_MAX_DIM = 10000
EE_DOWNLOAD_SAFETY_FRACTION = 0.8
LARGE_DOWNLOAD_ERROR_HINTS = (
    "32 mb",
    "maximum request size",
    "grid dimension",
    "too large",
    "request too large",
    "request exceeds",
)



def _bbox_span_m(coords):
    minlon, minlat, maxlon, maxlat = [float(v) for v in coords]
    mid_lat = max(min((minlat + maxlat) / 2.0, 89.9999), -89.9999)
    width_m = abs(maxlon - minlon) * 111320.0 * max(math.cos(math.radians(mid_lat)), 1e-6)
    height_m = abs(maxlat - minlat) * 110574.0
    return max(width_m, 1.0), max(height_m, 1.0)



def _format_mb(size_bytes):
    return f"{size_bytes / (1024 * 1024):.1f} MB"



def _is_large_download_error(exc):
    msg = str(exc).lower()
    return any(hint in msg for hint in LARGE_DOWNLOAD_ERROR_HINTS)



def _coerce_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default



def _coerce_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default



def _pixel_type_nbytes(band_type):
    if not isinstance(band_type, dict):
        return 4

    precision = str(band_type.get("precision", "")).lower()
    if precision in {"double", "float64"}:
        return 8
    if precision in {"float", "float32"}:
        return 4
    if precision in {"byte", "uint8", "int8"}:
        return 1
    if precision in {"short", "int16", "uint16"}:
        return 2
    if precision in {"long", "int64", "uint64"}:
        return 8
    if precision in {"int", "int32", "uint32"}:
        vmin = band_type.get("min")
        vmax = band_type.get("max")
        if isinstance(vmin, (int, float)) and isinstance(vmax, (int, float)):
            if float(vmin).is_integer() and float(vmax).is_integer():
                vmin = int(vmin)
                vmax = int(vmax)
                if 0 <= vmin and vmax <= 255:
                    return 1
                if -128 <= vmin and vmax <= 127:
                    return 1
                if 0 <= vmin and vmax <= 65535:
                    return 2
                if -32768 <= vmin and vmax <= 32767:
                    return 2
        return 4
    return 4



def _mosaic_geotiffs(tile_paths, out_path, expected_band_names=None):
    if not tile_paths:
        raise ValueError("No tile paths were provided for mosaicking.")

    srcs = [rasterio.open(path) for path in tile_paths]
    try:
        mosaic, transform = rio_merge(srcs)
        profile = srcs[0].profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            count=mosaic.shape[0],
            dtype=str(mosaic.dtype),
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)
            if expected_band_names and len(expected_band_names) == mosaic.shape[0]:
                dst.descriptions = list(expected_band_names)
    finally:
        for src in srcs:
            src.close()


# ----------------------------
# Generic helpers
# ----------------------------


def build_mask_condition(img: ee.Image, cfg: dict) -> ee.Image:
    cm = cfg.get("cloud_mask", {}) or {}
    if not cm.get("enabled", False):
        return ee.Image(1)

    band = cm.get("band")
    if not band:
        raise ValueError("cloud_mask.enabled = true but cloud_mask.band is missing.")

    ctype = (cm.get("type") or "").lower()
    keep = bool(cm.get("keep", False))

    # If band does not exist, return a pass-through mask of ones.
    # We use a server-side conditional to avoid client-side queries.
    band_exists = img.bandNames().contains(ee.String(band))

    def _mask_from_band():
        bimg = img.select([band])
        if ctype in ("equals", "not_equals"):
            vals = cm.get("values")
            if not isinstance(vals, list) or len(vals) == 0:
                raise ValueError("cloud_mask.values must be a non-empty list for equals/not_equals.")
            cond = None
            for v in vals:
                test = bimg.eq(ee.Number(v))
                cond = test if cond is None else cond.Or(test)
            cond = cond if ctype == "equals" else cond.Not()
        elif ctype in ("bits_any", "bits_all"):
            bits = cm.get("bits")
            if not isinstance(bits, list) or len(bits) == 0:
                raise ValueError("cloud_mask.bits must be a non-empty list for bits_any/bits_all.")
            tests = [bimg.bitwiseAnd(ee.Number(1).leftShift(ee.Number(b))).neq(0) for b in bits]
            cond = tests[0]
            for t in tests[1:]:
                cond = cond.Or(t) if ctype == "bits_any" else cond.And(t)
            cond = cond if keep else cond.Not()
        else:
            raise ValueError("cloud_mask.type must be one of: equals, not_equals, bits_any, bits_all.")

        return ee.Image(1).updateMask(cond).unmask(0).gt(0)

    return ee.Image(ee.Algorithms.If(band_exists, _mask_from_band(), ee.Image(1)))



def _postprocess_geotiff(
    tif_path: str,
    expected_band_names: list,
    maskval_to_na: bool = True,
    enforce_float32: bool = False,
):
    """
    Post-download fixes:
      - Rename GeoTIFF band descriptions to EE band names.
      - If maskval_to_na=True: treat exact 0s as masked and set to NA.
    """
    if not os.path.exists(tif_path):
        print(f"[post] file not found: {tif_path}")
        return

    with rasterio.Env():
        with rasterio.open(tif_path, "r") as src:
            profile = src.profile.copy()
            count = src.count
            dtype = src.dtypes[0]
            current_descs = list(src.descriptions)

        if expected_band_names and len(expected_band_names) == count:
            need_rename = (not any(current_descs)) or (current_descs != expected_band_names)
            if need_rename:
                with rasterio.open(tif_path, "r+") as dst:
                    dst.descriptions = list(expected_band_names)
                print(f"[post] Band descriptions set to: {expected_band_names}")
            else:
                print("[post] Band descriptions already match EE names.")
        else:
            if not expected_band_names:
                print("[post] No expected band names provided; skipping rename.")
            else:
                print(f"[post] Band count mismatch (EE={len(expected_band_names)}, TIFF={count}); skipping rename.")

        if not maskval_to_na:
            print("[post] maskval_to_na=False -> skipping maskval->NA conversion.")
            return

        is_float = dtype.startswith("float")
        if is_float:
            with rasterio.open(tif_path, "r+") as dst:
                for i in range(1, count + 1):
                    arr = dst.read(i)
                    z = arr == 0
                    if z.any():
                        arr = arr.astype(np.float32, copy=False)
                        arr[z] = np.nan
                        dst.write(arr, indexes=i)
                try:
                    dst.nodata = np.nan
                except Exception:
                    pass
            print("[post] 0 values converted to NaN (float dataset).")
            return

        if enforce_float32:
            tmp_out = tif_path + ".__post.tmp.tif"
            with rasterio.open(tif_path, "r") as src2:
                profile2 = profile.copy()
                profile2.update(dtype="float32", nodata=np.nan)
                with rasterio.open(tmp_out, "w", **profile2) as dst2:
                    for i in range(1, count + 1):
                        arr = src2.read(i).astype(np.float32)
                        z = arr == 0
                        if z.any():
                            arr[z] = np.nan
                        dst2.write(arr, indexes=i)
                    if expected_band_names and len(expected_band_names) == count:
                        dst2.descriptions = list(expected_band_names)
                    else:
                        dst2.descriptions = current_descs
            os.replace(tmp_out, tif_path)
            print("[post] Integer dataset upcast to float32; 0 values set to NaN.")
        else:
            with rasterio.open(tif_path, "r+") as dst:
                try:
                    dst.nodata = 0
                    print("[post] Integer dataset: set nodata=0 (0 treated as NA).")
                except Exception as e:
                    print(f"[post] Could not set nodata=0 (int dataset): {e}")


# ----------------------------
# Main
# ----------------------------


class GEEDownloader:
    def __init__(self, yaml_path):
        self.cfg = self._load_config(yaml_path)
        self._validate_config()
        self._region_info = None

    def _load_config(self, p):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, "r") as f:
            return yaml.safe_load(f) or {}

    def _validate_config(self):
        req = [
            "coords", "download_dir", "start_year", "start_month", "start_day",
            "end_year", "end_month", "end_day", "bands", "scale", "out_format",
            "auth_mode", "filename_prefix",
        ]
        missing = [k for k in req if k not in self.cfg]
        if missing:
            raise ValueError(f"Missing required config keys: {', '.join(missing)}")
        if "collections" not in self.cfg and "collection" not in self.cfg:
            raise ValueError("Missing required config key: provide either 'collection' or 'collections'.")
        if len(self.cfg["coords"]) != 4:
            raise ValueError("coords must be [min_lon, min_lat, max_lon, max_lat].")
        try:
            self.cfg["scale"] = float(self.cfg["scale"])
        except (TypeError, ValueError) as e:
            raise ValueError("scale must be numeric.") from e
        if self.cfg["scale"] <= 0:
            raise ValueError("scale must be > 0.")
        parent = os.path.dirname(self.cfg["download_dir"]) or "."
        if not os.path.isdir(parent):
            raise ValueError(f"Parent directory of download_dir does not exist: {parent}")
        self.cfg["collections"] = self._normalize_collections(
            self.cfg.get("collections", self.cfg.get("collection"))
        )
        self.cfg["collection"] = self.cfg["collections"][0]
        daily_strategy = str(self.cfg.get("daily_strategy", "median")).lower()
        if daily_strategy not in {"median", "first"}:
            raise ValueError("daily_strategy must be one of: median, first.")
        self.cfg["daily_strategy"] = daily_strategy
        self.cfg.setdefault("crs", "EPSG:4326")
        self.cfg.setdefault("max_images", None)
        self.cfg.setdefault("tiling", {})

    @staticmethod
    def _normalize_collections(raw):
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError("collection/collections must be a non-empty string or list of strings.")

        seen = set()
        out = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("collection/collections entries must be non-empty strings.")
            col = item.strip()
            if col not in seen:
                seen.add(col)
                out.append(col)
        return out

    def initialize(self):
        mode = str(self.cfg["auth_mode"]).lower()
        if mode == "browser":
            try:
                ee.Initialize(project="yiyu-research")
            except Exception:
                ee.Authenticate()
                ee.Initialize(project="yiyu-research")
        elif mode == "service":
            email = self.cfg.get("service_account_email")
            key = self.cfg.get("service_account_key")
            if not email or not key:
                raise ValueError("For auth_mode=service, provide service_account_email and service_account_key.")
            creds = ee.ServiceAccountCredentials(email, key)
            ee.Initialize(creds)
        else:
            raise ValueError("auth_mode must be 'browser' or 'service'.")

    # ---- Region helpers (client-side GeoJSON for getDownloadURL) ----
    def _region(self):
        return ee.Geometry.Rectangle(self.cfg["coords"], proj="EPSG:4326", geodesic=False)

    def _region_json(self):
        if self._region_info is None:
            self._region_info = self._region().getInfo()
        return self._region_info

    @staticmethod
    def _region_json_from_coords(coords):
        minlon, minlat, maxlon, maxlat = [float(v) for v in coords]
        return {
            "type": "Polygon",
            "coordinates": [[
                [minlon, minlat],
                [maxlon, minlat],
                [maxlon, maxlat],
                [minlon, maxlat],
                [minlon, minlat],
            ]],
            "geodesic": False,
            "evenOdd": True,
        }

    # ---- Date helpers ----
    def _date_range(self):
        start = _date_str(self.cfg["start_year"], self.cfg["start_month"], self.cfg["start_day"])
        end = _date_str(self.cfg["end_year"], self.cfg["end_month"], self.cfg["end_day"])
        return start, end

    def _is_globalish_extent(self) -> bool:
        minlon, minlat, maxlon, maxlat = self.cfg["coords"]
        return (
            minlon <= -179.99 and maxlon >= 179.99 and
            minlat <= -89.99 and maxlat >= 89.99
        )

    def _unique_dates(self, collection: str):
        start, end = self._date_range()
        col = ee.ImageCollection(collection).filterDate(start, end)
        if not self._is_globalish_extent():
            col = col.filterBounds(self._region())
        else:
            print("[info] Global/near-global extent detected -> skipping filterBounds()")

        size = col.size().getInfo()
        if size == 0:
            print(f"No images found in {collection} for {start}..{end} and region.")
            return []
        ts = col.aggregate_array("system:time_start").getInfo()
        dates = sorted({datetime.utcfromtimestamp(t / 1000).strftime("%Y-%m-%d") for t in ts})
        return dates

    # ---- Build a per-day daily image (mask + median or first-image selection) ----
    def _composite_for_day(self, day_str: str, collection: str) -> ee.Image:
        next_day = (datetime.strptime(day_str, "%Y-%m-%d") + relativedelta(days=1)).strftime("%Y-%m-%d")
        col = ee.ImageCollection(collection).filterDate(day_str, next_day).sort("system:time_start")
        if not self._is_globalish_extent():
            col = col.filterBounds(self._region())

        if (self.cfg.get("cloud_mask") or {}).get("enabled", False):
            def _apply_mask(im):
                m = build_mask_condition(ee.Image(im), self.cfg)
                return ee.Image(im).updateMask(m)
            col = col.map(_apply_mask)

        if self.cfg.get("daily_strategy", "median") == "first":
            return ee.Image(col.first())

        composite = col.reduce(ee.Reducer.median())
        bnames = composite.bandNames()
        new_names = bnames.map(lambda n: ee.String(n).replace("_median$", ""))
        composite = composite.rename(new_names)
        return composite

    def _select_bands(self, img: ee.Image) -> ee.Image:
        bands = self.cfg.get("bands") or []
        if not bands:
            return img
        avail = img.bandNames().getInfo()
        keep = [b for b in bands if b in avail]
        if not keep:
            raise RuntimeError(f"None of requested bands exist. Requested={bands}, Available={avail}")
        return img.select(keep)

    def _get_download_url(self, img: ee.Image, region_json=None) -> str:
        params = {
            "scale": self.cfg["scale"],
            "crs": self.cfg["crs"],
            "filePerBand": False,
            "region": region_json or self._region_json(),
            "format": "GEO_TIFF",
        }
        return img.getDownloadURL(params)

    def _tiling_cfg(self):
        raw = self.cfg.get("tiling", {}) or {}
        max_request_bytes = max(1, _coerce_int(raw.get("max_request_bytes"), EE_DOWNLOAD_MAX_BYTES))
        safety_fraction = _coerce_float(raw.get("safety_fraction"), EE_DOWNLOAD_SAFETY_FRACTION)
        safety_fraction = min(max(safety_fraction, 0.1), 1.0)
        max_tile_dim = max(1, _coerce_int(raw.get("max_tile_dim"), EE_DOWNLOAD_MAX_DIM))
        max_depth = max(1, _coerce_int(raw.get("max_depth"), 8))
        return {
            "enabled": bool(raw.get("enabled", True)),
            "max_request_bytes": max_request_bytes,
            "target_bytes": max(1, int(max_request_bytes * safety_fraction)),
            "max_tile_dim": max_tile_dim,
            "max_depth": max_depth,
        }

    def _bytes_per_pixel(self, img: ee.Image, band_names: list) -> int:
        band_types = img.bandTypes().getInfo() or {}
        total = 0
        for band_name in band_names:
            total += _pixel_type_nbytes(band_types.get(band_name, {}))
        return max(total, 4 * max(len(band_names), 1))

    def _estimate_request(self, region_coords, bytes_per_pixel: int, band_count: int):
        width_m, height_m = _bbox_span_m(region_coords)
        width_px = max(1, int(math.ceil(width_m / self.cfg["scale"])))
        height_px = max(1, int(math.ceil(height_m / self.cfg["scale"])))
        return {
            "width_px": width_px,
            "height_px": height_px,
            "band_count": band_count,
            "bytes_per_pixel": max(1, int(bytes_per_pixel)),
            "estimated_bytes": width_px * height_px * max(1, int(bytes_per_pixel)),
        }

    @staticmethod
    def _needs_tiling(estimate: dict, tiling_cfg: dict) -> bool:
        return (
            estimate["estimated_bytes"] > tiling_cfg["target_bytes"] or
            estimate["width_px"] > tiling_cfg["max_tile_dim"] or
            estimate["height_px"] > tiling_cfg["max_tile_dim"]
        )

    @staticmethod
    def _tile_grid(estimate: dict, tiling_cfg: dict, force_split: bool = False):
        max_pixels_by_bytes = max(1, tiling_cfg["target_bytes"] // estimate["bytes_per_pixel"])
        max_edge = max(1, min(tiling_cfg["max_tile_dim"], int(math.floor(math.sqrt(max_pixels_by_bytes)))))
        nx = max(1, int(math.ceil(estimate["width_px"] / max_edge)))
        ny = max(1, int(math.ceil(estimate["height_px"] / max_edge)))
        if force_split and nx == 1 and ny == 1:
            if estimate["width_px"] >= estimate["height_px"]:
                nx = 2
            else:
                ny = 2
        return nx, ny

    @staticmethod
    def _split_region(region_coords, nx: int, ny: int):
        minlon, minlat, maxlon, maxlat = [float(v) for v in region_coords]
        lon_edges = np.linspace(minlon, maxlon, nx + 1)
        lat_edges = np.linspace(minlat, maxlat, ny + 1)
        tiles = []
        for row in range(ny - 1, -1, -1):
            bottom = lat_edges[row]
            top = lat_edges[row + 1]
            for col in range(nx):
                left = lon_edges[col]
                right = lon_edges[col + 1]
                tiles.append([float(left), float(bottom), float(right), float(top)])
        return tiles

    def _download_direct(self, img: ee.Image, region_coords, out_path: str):
        parent = os.path.dirname(out_path) or "."
        _safe_mkdir(parent)
        url = self._get_download_url(img, region_json=self._region_json_from_coords(region_coords))
        _retry_download(url, out_path)

    def _download_tiled_region(
        self,
        img: ee.Image,
        region_coords,
        tile_root: str,
        bytes_per_pixel: int,
        band_count: int,
        depth: int = 0,
        tile_tag: str = "tile",
    ):
        tiling_cfg = self._tiling_cfg()
        if depth > tiling_cfg["max_depth"]:
            raise RuntimeError(f"Exceeded max tiling depth ({tiling_cfg['max_depth']}) while downloading {tile_tag}.")

        estimate = self._estimate_request(region_coords, bytes_per_pixel, band_count)
        leaf_path = os.path.join(tile_root, f"{tile_tag}.tif")

        if not self._needs_tiling(estimate, tiling_cfg):
            try:
                self._download_direct(img, region_coords, leaf_path)
                return [leaf_path]
            except Exception as e:
                if not _is_large_download_error(e):
                    raise
                print(f"[tile] {tile_tag}: Earth Engine reported an oversized request; subdividing.")

        nx, ny = self._tile_grid(estimate, tiling_cfg, force_split=True)
        print(
            f"[tile] {tile_tag}: estimate {_format_mb(estimate['estimated_bytes'])} "
            f"for ~{estimate['width_px']}x{estimate['height_px']} px "
            f"({estimate['band_count']} band(s), {estimate['bytes_per_pixel']} B/pixel); "
            f"splitting into {nx}x{ny} tile(s)."
        )

        tile_paths = []
        subtile_regions = self._split_region(region_coords, nx, ny)
        for idx, sub_coords in enumerate(subtile_regions, 1):
            sub_tag = f"{tile_tag}_{idx:03d}"
            tile_paths.extend(
                self._download_tiled_region(
                    img,
                    sub_coords,
                    tile_root,
                    bytes_per_pixel,
                    band_count,
                    depth=depth + 1,
                    tile_tag=sub_tag,
                )
            )
        return tile_paths

    def _download_image(self, img: ee.Image, out_path: str, ee_band_names: list):
        tiling_cfg = self._tiling_cfg()
        band_count = max(len(ee_band_names), 1)
        bytes_per_pixel = self._bytes_per_pixel(img, ee_band_names)
        full_estimate = self._estimate_request(self.cfg["coords"], bytes_per_pixel, band_count)
        needs_tiling = tiling_cfg["enabled"] and self._needs_tiling(full_estimate, tiling_cfg)

        if not needs_tiling:
            try:
                self._download_direct(img, self.cfg["coords"], out_path)
                return
            except Exception as e:
                if not tiling_cfg["enabled"] or not _is_large_download_error(e):
                    raise
                print("[tile] Full-region request was rejected as too large; retrying with tiles.")

        tile_root = out_path + ".__tiles"
        shutil.rmtree(tile_root, ignore_errors=True)
        _safe_mkdir(tile_root)
        try:
            tile_paths = self._download_tiled_region(
                img,
                self.cfg["coords"],
                tile_root,
                bytes_per_pixel,
                band_count,
            )
            _mosaic_geotiffs(tile_paths, out_path, expected_band_names=ee_band_names)
        finally:
            shutil.rmtree(tile_root, ignore_errors=True)

    def run(self):
        print("Initializing Earth Engine...")
        self.initialize()
        _safe_mkdir(self.cfg["download_dir"])
        collections = self.cfg["collections"]
        print(f"Using collections: {', '.join(collections)}")

        ok, fail, skipped_dup = 0, 0, 0
        cap = self.cfg.get("max_images")

        for c_idx, collection in enumerate(collections, 1):
            print(f"[collection {c_idx}/{len(collections)}] {collection}")
            dates = self._unique_dates(collection)
            if isinstance(cap, int) and cap > 0:
                dates = dates[:cap]
                print(f"Limiting to first {cap} day(s) for {collection}.")

            total = len(dates)
            print(f"Found {total} date(s) in {collection}. Starting per-day composites + downloads...")

            for i, day in enumerate(dates, 1):
                out_name = f"{self.cfg['filename_prefix']}_{day}.{self.cfg['out_format']}"
                out_final = os.path.join(self.cfg["download_dir"], out_name)
                collection_tag = collection.replace("/", "_")
                out_tmp = os.path.join(self.cfg["download_dir"], f"__tmp_{collection_tag}_{out_name}")

                if os.path.exists(out_final):
                    print(f"[{i}/{total}] {day}: skip duplicate (already exists {out_name})")
                    skipped_dup += 1
                    continue

                print(f"[{i}/{total}] {day}: building composite ({collection})")
                try:
                    img = self._composite_for_day(day, collection)
                    img = self._select_bands(img)

                    ee_band_names = img.bandNames().getInfo()
                    self._download_image(img, out_tmp, ee_band_names)

                    pp = self.cfg.get("postprocess", {}) or {}
                    maskval_to_na = pp.get("maskval_to_na", True)
                    enforce_f32 = pp.get("enforce_float32", False)

                    try:
                        _postprocess_geotiff(
                            out_tmp,
                            expected_band_names=ee_band_names,
                            maskval_to_na=maskval_to_na,
                            enforce_float32=enforce_f32,
                        )
                    except Exception as e:
                        print(f"[post] warning: postprocessing failed: {e}")

                    os.replace(out_tmp, out_final)
                    print(f"  saved {out_name}")
                    ok += 1

                except Exception as e:
                    print(f"  failed {day} ({collection}): {e}")
                    try:
                        if os.path.exists(out_tmp):
                            os.remove(out_tmp)
                        shutil.rmtree(out_tmp + ".__tiles", ignore_errors=True)
                    except Exception:
                        pass
                    fail += 1

        print(f"Done. Success={ok}, Failed={fail}, SkippedDuplicates={skipped_dup}")


# ----------------------------
# CLI
# ----------------------------


def main():
    if len(sys.argv) < 2:
        print("Usage: python 0_gee_downloader.py <config.yaml>")
        sys.exit(1)
    try:
        GEEDownloader(sys.argv[1]).run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
