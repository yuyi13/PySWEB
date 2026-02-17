import os
import sys
import yaml
import time
import json
import numpy as np
import rasterio
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests
import ee

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

    # If band doesn't exist, return a pass-through mask of ones.
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
            # cond == True where bits present
            cond = cond if keep else cond.Not()
        else:
            raise ValueError("cloud_mask.type must be one of: equals, not_equals, bits_any, bits_all.")

        return ee.Image(1).updateMask(cond).unmask(0).gt(0)

    # If the band exists, build the mask; otherwise use ones (no-op)
    return ee.Image(ee.Algorithms.If(band_exists, _mask_from_band(), ee.Image(1)))

def _postprocess_geotiff(
    tif_path: str,
    expected_band_names: list,   # <— EE band names after select/rename
    maskval_to_na: bool = True,
    enforce_float32: bool = False
):
    """
    Post-download fixes:
      • Rename GeoTIFF band descriptions to EE's band names (so you get e.g., ['B4','B5']).
      • If maskval_to_na=True: treat exact 0s as masked and set to NA (NaN for float rasters; or nodata=0 for ints unless upcast).
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

        # --- 1) Rename band descriptions to match EE band names ---
        if expected_band_names and len(expected_band_names) == count:
            # Only rewrite if they’re missing or different
            need_rename = (not any(current_descs)) or (current_descs != expected_band_names)
            if need_rename:
                with rasterio.open(tif_path, "r+") as dst:
                    dst.descriptions = list(expected_band_names)
                print(f"[post] ✅ Band descriptions set to: {expected_band_names}")
            else:
                print(f"[post] ✅ Band descriptions already match EE names.")
        else:
            if not expected_band_names:
                print("[post] ⚠️ No expected band names provided; skipping rename.")
            else:
                print(f"[post] ⚠️ Band count mismatch (EE={len(expected_band_names)}, TIFF={count}); skipping rename.")

        # --- 2) maskval → NA behaviour ---
        if not maskval_to_na:
            print("[post] maskval_to_na=False → skipping maskval→NA conversion.")
            return

        is_float = dtype.startswith("float")
        if is_float:
            # In-place conversion to NaN for maskvals
            with rasterio.open(tif_path, "r+") as dst:
                for i in range(1, count + 1):
                    arr = dst.read(i)
                    z = (arr == 0)
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

        # Integer dataset
        if enforce_float32:
            tmp_out = tif_path + ".__post.tmp.tif"
            with rasterio.open(tif_path, "r") as src2:
                profile2 = profile.copy()
                profile2.update(dtype="float32", nodata=np.nan)
                with rasterio.open(tmp_out, "w", **profile2) as dst2:
                    for i in range(1, count + 1):
                        arr = src2.read(i).astype(np.float32)
                        z = (arr == 0)
                        if z.any():
                            arr[z] = np.nan
                        dst2.write(arr, indexes=i)
                    # Keep the band descriptions we just set (or had)
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
        self._region_info = None  # cache client-side GeoJSON region

    def _load_config(self, p):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, "r") as f:
            return yaml.safe_load(f) or {}

    def _validate_config(self):
        req = [
            "coords", "download_dir", "start_year", "start_month", "start_day",
            "end_year", "end_month", "end_day", "bands", "scale", "out_format",
            "auth_mode", "filename_prefix"
        ]
        missing = [k for k in req if k not in self.cfg]
        if missing:
            raise ValueError(f"Missing required config keys: {', '.join(missing)}")
        if "collections" not in self.cfg and "collection" not in self.cfg:
            raise ValueError("Missing required config key: provide either 'collection' or 'collections'.")
        if len(self.cfg["coords"]) != 4:
            raise ValueError("coords must be [min_lon, min_lat, max_lon, max_lat].")
        parent = os.path.dirname(self.cfg["download_dir"]) or "."
        if not os.path.isdir(parent):
            raise ValueError(f"Parent directory of download_dir does not exist: {parent}")
        self.cfg["collections"] = self._normalize_collections(
            self.cfg.get("collections", self.cfg.get("collection"))
        )
        # Keep backward-compatible single-collection key for callers that inspect cfg.
        self.cfg["collection"] = self.cfg["collections"][0]
        self.cfg.setdefault("crs", "EPSG:4326")
        self.cfg.setdefault("max_images", None)

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
                ee.Initialize(project='yiyu-research') # Need a specific Google Cloud project
            except Exception:
                ee.Authenticate()
                ee.Initialize(project='yiyu-research')
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

    # ---- Date helpers ----
    def _date_range(self):
        start = _date_str(self.cfg["start_year"], self.cfg["start_month"], self.cfg["start_day"])
        end   = _date_str(self.cfg["end_year"],   self.cfg["end_month"],   self.cfg["end_day"])
        return start, end

    def _is_globalish_extent(self) -> bool:
        """
        Return True if coords basically cover the globe.
        Uses tolerant thresholds to be robust to tiny rounding differences.
        """
        minlon, minlat, maxlon, maxlat = self.cfg["coords"]
        return (
            minlon <= -179.99 and maxlon >= 179.99 and
            minlat <= -89.99  and maxlat >=  89.99
        )

    def _unique_dates(self, collection: str):
        start, end = self._date_range()
        col = ee.ImageCollection(collection).filterDate(start, end)
        if not self._is_globalish_extent():
            col = col.filterBounds(self._region())
        else:
            print("[info] Global/near-global extent detected → skipping filterBounds()")

        size = col.size().getInfo()
        if size == 0:
            print(f"No images found in {collection} for {start}..{end} and region.")
            return []
        ts = col.aggregate_array("system:time_start").getInfo()
        dates = sorted({datetime.utcfromtimestamp(t/1000).strftime("%Y-%m-%d") for t in ts})
        return dates

    # ---- Build a per-day composite (mask-then-reduce-median) ----
    def _composite_for_day(self, day_str: str, collection: str) -> ee.Image:
        next_day = (datetime.strptime(day_str, "%Y-%m-%d") + relativedelta(days=1)).strftime("%Y-%m-%d")
        col = ee.ImageCollection(collection).filterDate(day_str, next_day)
        if not self._is_globalish_extent():
            col = col.filterBounds(self._region())
        # else: global/near-global → skip filterBounds()

        # Map: apply cloud mask independently of final band selection
        if (self.cfg.get("cloud_mask") or {}).get("enabled", False):
            def _apply_mask(im):
                m = build_mask_condition(ee.Image(im), self.cfg)
                return ee.Image(im).updateMask(m)
            col = col.map(_apply_mask)

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

    def _get_download_url(self, img: ee.Image) -> str:
        params = {
            "scale": self.cfg["scale"],
            "crs": self.cfg["crs"],
            "filePerBand": False,
            "region": self._region_json(),  # client-side GeoJSON (dict)
            "format": "GEO_TIFF",
        }
        return img.getDownloadURL(params)

    def run(self):
        print("Initializing Earth Engine…")
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
            print(f"Found {total} date(s) in {collection}. Starting per-day composites + downloads…")

            for i, day in enumerate(dates, 1):
                out_name = f"{self.cfg['filename_prefix']}_{day}.{self.cfg['out_format']}"
                out_final = os.path.join(self.cfg["download_dir"], out_name)
                collection_tag = collection.replace("/", "_")
                out_tmp = os.path.join(self.cfg["download_dir"], f"__tmp_{collection_tag}_{out_name}")

                # Avoid duplicated date downloads when looping multiple collections.
                if os.path.exists(out_final):
                    print(f"[{i}/{total}] {day}: skip duplicate (already exists {out_name})")
                    skipped_dup += 1
                    continue

                print(f"[{i}/{total}] {day}: building composite ({collection})")
                try:
                    img = self._composite_for_day(day, collection)
                    img = self._select_bands(img)  # selection AFTER masking + composite

                    # Get authoritative EE band names after selection/rename.
                    ee_band_names = img.bandNames().getInfo()

                    url = self._get_download_url(img)
                    _retry_download(url, out_tmp)

                    # --- POSTPROCESS ---
                    pp = self.cfg.get("postprocess", {}) or {}
                    maskval_to_na = pp.get("maskval_to_na", True)
                    enforce_f32 = pp.get("enforce_float32", False)

                    try:
                        _postprocess_geotiff(
                            out_tmp,
                            expected_band_names=ee_band_names,
                            maskval_to_na=maskval_to_na,
                            enforce_float32=enforce_f32
                        )
                    except Exception as e:
                        print(f"[post] warning: postprocessing failed: {e}")

                    os.replace(out_tmp, out_final)

                    print(f"  ✓ saved {out_name}")
                    ok += 1

                except Exception as e:
                    print(f"  ✗ failed {day} ({collection}): {e}")
                    try:
                        if os.path.exists(out_tmp):
                            os.remove(out_tmp)
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
