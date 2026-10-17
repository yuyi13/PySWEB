"""
Script: preprocessing.py
Objective: Apply configured Landsat restrictions and independently gated L7 gap filling.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Immutable scene records, raw QA-bearing rasters, optical masks and acceptance rules.
Outputs: Physical L7 bands, per-band provenance and identity-bound validation receipts.
Usage: from pysweb.ssebop.preprocessing import prepare_l7, select_scenes
Dependencies: numpy, pandas, rasterio, xarray, pysweb
"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import xarray as xr
from pysweb.io.provenance import atomic_json, atomic_netcdf, input_manifest, sha256
from pysweb.ssebop.gapfill import local_linear_gapfill
from pysweb.ssebop.quality import landsat_c2_qa_mask


def select_scenes(records, sensors, start, end, *, sensor_windows=None, wrs=None):
    """Select immutable products, then resolve same-day ties by cloud cover/id."""
    result = []
    ids = set()
    for record in records:
        required = {"scene_id", "sensor", "date", "path", "row", "cloud_cover", "file"}
        if required-set(record):
            raise ValueError(f"Scene record lacks {sorted(required-set(record))}")
        if record["scene_id"] in ids:
            raise ValueError("Duplicate scene identifier")
        ids.add(record["scene_id"])
        date = pd.Timestamp(record["date"])
        if record["sensor"] not in sensors or not pd.Timestamp(start) <= date <= pd.Timestamp(end):
            continue
        if wrs and (int(record["path"]), int(record["row"])) != tuple(wrs):
            continue
        if record["sensor"] in (sensor_windows or {}):
            first, last = sensor_windows[record["sensor"]]
            if not pd.Timestamp(first) <= date <= pd.Timestamp(last):
                continue
        if not np.isfinite(record["cloud_cover"]):
            raise ValueError("Finite cloud cover required for deterministic selection")
        result.append(record)
    selected = {}
    for record in sorted(result, key=lambda r: (r["date"], r["cloud_cover"], r["scene_id"])):
        selected.setdefault(str(pd.Timestamp(record["date"]).date()), record)
    return list(selected.values())


def donors_for(target, records, *, start, end, max_days=64, max_donors=5):
    if target["sensor"] != "LE07" or not pd.Timestamp(start) <= pd.Timestamp(target["date"]) <= pd.Timestamp(end):
        raise ValueError("L7 target outside approved development window")
    if max_days < 1 or max_donors < 1:
        raise ValueError("Positive donor limits required")
    date = pd.Timestamp(target["date"])
    donors = [r for r in records if r["sensor"] == "LE07" and r["scene_id"] != target["scene_id"]
        and (r["path"], r["row"]) == (target["path"], target["row"])
        and pd.Timestamp(start) <= pd.Timestamp(r["date"]) <= pd.Timestamp(end)
        and abs((pd.Timestamp(r["date"])-date).days) <= max_days]
    return sorted(donors, key=lambda r: (abs((pd.Timestamp(r["date"])-date).days), r["cloud_cover"], r["scene_id"]))[:max_donors]


def _read(record, template=None, crop_size=None):
    """Read original bands only; exact integer-pixel windows align native donors."""
    with rasterio.open(record["file"]) as src:
        names = {name: i+1 for i, name in enumerate(src.descriptions)}
        required = ["ST_B6", "SR_B3", "SR_B4", "QA_PIXEL", "QA_RADSAT"]
        if not set(required) <= set(names):
            raise ValueError("L7 preprocessing requires named raw C2 bands and QA")
        window = None
        if template is not None:
            if src.crs != template["crs"] or not np.allclose(tuple(src.transform)[:2]+tuple(src.transform)[3:5],
                    tuple(template["transform"])[:2]+tuple(template["transform"])[3:5], atol=1e-9, rtol=0):
                raise ValueError("L7 donors must share source CRS and pixel basis")
            col, row = ~src.transform * (template["transform"].c, template["transform"].f)
            if not np.allclose([col, row], np.rint([col, row]), atol=1e-6, rtol=0):
                raise ValueError("L7 donor native grids are not congruent")
            window = Window(round(col), round(row), template["width"], template["height"])
        elif crop_size:
            width, height = min(crop_size, src.width), min(crop_size, src.height)
            window = Window((src.width-width)//2, (src.height-height)//2, width, height)
        raw = src.read([names[n] for n in required], window=window, boundless=window is not None, fill_value=0)
        profile = dict(crs=src.crs, transform=src.window_transform(window) if window else src.transform,
                       width=raw.shape[2], height=raw.shape[1])
        st, red, nir, qa, saturation = raw
    valid = landsat_c2_qa_mask(qa, saturation, sensor="LE07") & (st > 0) & (red > 0) & (nir > 0)
    values = np.array([st*.00341802+149, red*.0000275-.2, nir*.0000275-.2])
    values[:, ~valid] = np.nan
    # Masked QA zeros cannot distinguish scan gaps from source footprint edges.
    with rasterio.open(record["gap_mask_file"]) as src:
        if src.crs != profile["crs"]:
            raise ValueError("Optical support mask CRS differs")
        col, row = ~src.transform * (profile["transform"].c, profile["transform"].f)
        if not np.allclose([col, row], np.rint([col, row]), atol=1e-6, rtol=0) or not np.allclose(
                [src.transform.a, src.transform.b, src.transform.d, src.transform.e],
                [profile["transform"].a, profile["transform"].b, profile["transform"].d, profile["transform"].e], atol=1e-9, rtol=0):
            raise ValueError("Optical support mask grid differs")
        masks = src.read([1, 2], window=Window(round(col), round(row), profile["width"], profile["height"]), boundless=True, fill_value=0)
    gaps = (masks[0] == 1) & (masks[1] == 1) & (red == 0) & (nir == 0)
    return values, gaps, profile


def _ndvi(values):
    denominator = values[1]+values[2]
    result = np.divide(values[2]-values[1], denominator, out=np.full(denominator.shape, np.nan), where=np.abs(denominator) > 1e-8)
    return np.where((result >= -1) & (result <= 1), result, np.nan)


def fill_bands(primary, donors, gaps, *, min_common=144, max_window=31):
    results = [local_linear_gapfill(primary[b], (d[b] for d in donors), gaps,
        value_limits=limits, min_common=min_common, max_window=max_window)
        for b, limits in enumerate(((200, 370), (-.2, 1.6), (-.2, 1.6)))]
    for b, result in enumerate(results):
        good = np.isfinite(primary[b])
        if not np.array_equal(result["values"][good], primary[b][good]) or np.any((result["provenance"] > 1) & ~gaps):
            raise ValueError("Gap-fill preservation/eligibility contract failed")
    return results


def withheld_check(original, donors, gaps, rules, **options):
    eligible = np.isfinite(original).all(axis=0)
    withheld = np.roll(gaps, int(rules.get("stripe_shift_rows", 7)), axis=0) & eligible
    border = int(rules.get("border_pixels", 16))
    if border:
        withheld[:border] = withheld[-border:] = False
        withheld[:, :border] = withheld[:, -border:] = False
    primary = original.copy()
    primary[:, withheld] = np.nan
    results = fill_bands(primary, donors, withheld, **options)
    values = np.array([r["values"] for r in results])
    supported = withheld & np.isfinite(values).all(axis=0) & np.isfinite(_ndvi(values))
    dt, dv = values[0][supported]-original[0][supported], _ndvi(values)[supported]-_ndvi(original)[supported]
    n = len(dt)
    fraction = float(n/withheld.sum()) if withheld.any() else 0.
    rmse = float(np.sqrt(np.mean(dt**2))) if n else None
    bias = float(dt.mean()) if n else None
    ndvi_rmse = float(np.sqrt(np.mean(dv**2))) if n else None
    passed = n >= rules["min_scored_pixels"] and fraction >= rules["minimum_fill_fraction"] and rmse is not None and rmse <= rules["LST_RMSE_max_K"] and abs(bias) <= rules["LST_absolute_bias_max_K"] and ndvi_rmse <= rules["NDVI_RMSE_max"]
    return dict(passed=bool(passed), scored_pixels=n, fill_fraction=fraction,
        LST_RMSE_K=rmse, LST_bias_K=bias, NDVI_RMSE=ndvi_rmse)


def prepare_l7(records, output_dir, config, *, development_end):
    """Validate first, then process original L7 inputs with full identity receipts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if pd.Timestamp(config["end"]) > pd.Timestamp(development_end):
        raise ValueError("Gap-fill donors cannot include holdout dates")
    selected = [r for r in records if r["sensor"] == "LE07" and pd.Timestamp(config["start"]) <= pd.Timestamp(r["date"]) <= pd.Timestamp(config["end"])]
    if not selected:
        raise ValueError("No eligible L7 scenes")
    paths = [r[k] for r in selected for k in ("file", "gap_mask_file")]
    manifest = input_manifest(config, paths)
    manifest["scene_records"] = selected
    # Include catalogue identity: cloud ranking and dates can change donors even
    # when the files are unchanged.
    import hashlib
    import json
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    gate_path = output_dir/"validation.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    options = {k: config[k] for k in ("min_common", "max_window") if k in config}
    donor_options = {k: config[k] for k in ("start", "end", "max_days", "max_donors") if k in config}
    rules = config["validation"]
    if gate.get("identity") != identity or not gate.get("passed"):
        checks = []
        for year in rules["years"]:
            for half in (0, 1):
                candidates = [r for r in selected if pd.Timestamp(r["date"]).year == year and (pd.Timestamp(r["date"]).month-1)//6 == half]
                if not candidates:
                    raise ValueError("Missing prospectively required L7 validation scene")
                target = min(candidates, key=lambda r: (r["cloud_cover"], r["date"], r["scene_id"]))
                original, gaps, profile = _read(target, crop_size=rules.get("crop_pixels", 512))
                donors = donors_for(target, selected, **donor_options)
                arrays = [_read(donor, profile)[0] for donor in donors]
                check = withheld_check(original, arrays, gaps, rules, **options)
                checks.append(dict(scene_id=target["scene_id"], year=year, date=target["date"],
                                   donors=[d["scene_id"] for d in donors], **check))
        passed = sum(c["passed"] for c in checks) >= rules["minimum_passed_scenes"] and all(
            any(c["passed"] and c["year"] == year for c in checks) for year in rules["years"])
        gate = dict(identity=identity, passed=passed, checks=checks, inputs=manifest)
        atomic_json(gate_path, gate)
    if not gate["passed"]:
        raise ValueError("L7 withholding acceptance failed; retrieval withheld")
    outputs = {}
    for target in selected:
        path = output_dir/(target["scene_id"].split("/")[-1]+".nc")
        receipt = path.with_suffix(".json")
        cached = json.loads(receipt.read_text()) if receipt.exists() else {}
        if cached.get("identity") == identity and path.exists() and cached.get("sha256") == sha256(path):
            outputs[str(Path(target["file"]).resolve())] = str(path)
            continue
        original, gaps, profile = _read(target)
        donors = donors_for(target, selected, **donor_options)
        results = fill_bands(original, [_read(donor, profile)[0] for donor in donors], gaps, **options)
        transform = profile["transform"]
        coords = dict(y=transform.f+(np.arange(profile["height"])+.5)*transform.e,
                      x=transform.c+(np.arange(profile["width"])+.5)*transform.a)
        ds = xr.Dataset(coords=coords)
        for b, name in enumerate(("lst_K", "red_reflectance", "nir_reflectance")):
            ds[name] = xr.DataArray(results[b]["values"].astype("float32"), dims=("y", "x"))
            for key in ("provenance", "method", "window", "common_count"):
                ds[name+"_"+key] = xr.DataArray(results[b][key], dims=("y", "x"))
        ds["scan_gap_candidate"] = xr.DataArray(gaps.astype("uint8"), dims=("y", "x"))
        ds["original_support"] = xr.DataArray(np.isfinite(original).all(axis=0).astype("uint8"), dims=("y", "x"))
        ds.attrs.update(gapfill_identity=identity, source_scene=target["scene_id"],
            donors=json.dumps([d["scene_id"] for d in donors]), validation_receipt=str(gate_path),
            scientific_status="spatiotemporal estimates; residual holes remain unavailable")
        ds = ds.rio.write_crs(profile["crs"]).rio.write_transform(transform)
        atomic_netcdf(ds, path)
        # Verify written float32 originals and provenance, not just in-memory data.
        with xr.open_dataset(path) as written:
            for b, name in enumerate(("lst_K", "red_reflectance", "nir_reflectance")):
                native = np.isfinite(original[b])
                if not np.array_equal(written[name].values[native], original[b].astype("float32")[native]):
                    raise ValueError("Written original L7 pixels changed")
        atomic_json(receipt, dict(identity=identity, sha256=sha256(path), target=target,
                                 donors=donors, validation=str(gate_path)))
        outputs[str(Path(target["file"]).resolve())] = str(path)
    atomic_json(output_dir/"products.json", dict(identity=identity, passed=True, products=outputs))
    return outputs


def read_verified_product(path, raw_path):
    """Reject stale or substituted preprocessing before a retrieval consumes it."""
    import json
    path = Path(path)
    receipt = json.loads(path.with_suffix(".json").read_text())
    gate = json.loads(Path(receipt["validation"]).read_text())
    if not gate["passed"] or gate["identity"] != receipt["identity"] or sha256(path) != receipt["sha256"]:
        raise ValueError("L7 preprocessing receipt is stale or failed")
    if Path(receipt["target"]["file"]).resolve() != Path(raw_path).resolve():
        raise ValueError("L7 preprocessing belongs to a different raw scene")
    expected = {item["path"]: item["sha256"] for item in gate["inputs"]["inputs"]}
    for record in [receipt["target"], *receipt["donors"]]:
        for key in ("file", "gap_mask_file"):
            source = str(Path(record[key]).resolve())
            if sha256(source) != expected[source]:
                raise ValueError("L7 raw inputs changed after validation")
    with xr.open_dataset(path, decode_coords="all") as ds:
        if ds.attrs["gapfill_identity"] != receipt["identity"]:
            raise ValueError("L7 product identity mismatch")
        return ds.load()
