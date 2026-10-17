#!/usr/bin/env python3
"""
Script: landsat.py
Objective: Provide the canonical Landsat preparation helpers used by package-backed SSEBop workflows.
Author: Yi Yu (with assistance from Codex)
Created: 2026-04-20
Last updated: 2026-10-17
Inputs: Date ranges, extents, Earth Engine project names, and optional Landsat download config templates.
Outputs: Materialized Landsat download configs and prepared Landsat GeoTIFF inputs.
Usage: import pysweb.ssebop.landsat
Dependencies: PyYAML for YAML templates; pysweb; authenticated Earth Engine project for downloads
"""
from __future__ import annotations

import json
import copy
import re
from datetime import datetime
from pathlib import Path

from pysweb.io.gee import GEEDownloader, _safe_mkdir

__all__ = [
    "parse_date_range",
    "parse_extent",
    "update_gee_config",
    "write_gee_config_from_cfg",
    "prepare_landsat_inputs",
]

_DEFAULT_LANDSAT_GEE_CFG = {
    "collection": "LANDSAT/LC08/C02/T1_L2",
    "bands": ["ST_B10", "SR_B4", "SR_B5", "QA_PIXEL", "QA_RADSAT"],
    "daily_strategy": "first",
    "cloud_mask": {
        "enabled": True,
        "band": "QA_PIXEL",
        "type": "bits_any",
        "bits": [0, 1, 2, 3, 4, 5, 7],
        "keep": False,
    },
    # Zero denotes an unsaturated pixel in QA_RADSAT. Preserve raw flags;
    # the scene processor handles fill and positive optical/thermal DNs.
    "postprocess": {"maskval_to_na": False},
    "preserve_native_grid": True,
    "out_format": "tif",
    "auth_mode": "browser",
    "filename_prefix": "Landsat",
}


def _load_yaml_module():
    try:
        import yaml
    except ModuleNotFoundError:
        return None
    return yaml


def parse_date_range(date_range: str) -> tuple[str, str]:
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", date_range)
    if len(dates) != 2:
        raise ValueError("Date range must include two dates in YYYY-MM-DD format.")
    return dates[0], dates[1]


def parse_extent(extent_str: str) -> list[float]:
    parts = [part for part in re.split(r"[,\s]+", extent_str.strip()) if part]
    if len(parts) != 4:
        raise ValueError("Extent must be four numbers: min_lon,min_lat,max_lon,max_lat")
    return [float(part) for part in parts]


def _validate_gee_project(gee_project: str) -> str:
    project = gee_project.strip()
    if not project:
        raise ValueError("gee_project must be a non-empty string.")
    return project


def _read_gee_config(base_config_path: str) -> dict:
    payload = Path(base_config_path).read_text(encoding="utf-8")
    yaml = _load_yaml_module()
    if yaml is not None:
        return yaml.safe_load(payload) or {}
    return json.loads(payload or "{}")


def _write_gee_payload(cfg_path: Path, cfg: dict) -> None:
    yaml = _load_yaml_module()
    if yaml is not None:
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _write_gee_config(
    cfg: dict,
    start_date: str,
    end_date: str,
    extent: list[float],
    out_dir: str,
    gee_project: str,
) -> str:
    cfg = copy.deepcopy(cfg)
    gee_project = _validate_gee_project(gee_project)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if "coords" not in cfg or cfg["coords"] is None:
        cfg["coords"] = extent

    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)

    cfg["coords"] = extent
    cfg["download_dir"] = out_dir
    cfg["start_year"] = start_dt.year
    cfg["start_month"] = start_dt.month
    cfg["start_day"] = start_dt.day
    cfg["end_year"] = end_dt.year
    cfg["end_month"] = end_dt.month
    cfg["end_day"] = end_dt.day
    cfg["gee_project"] = gee_project

    cfg_path = Path(out_dir) / f"gee_config_{start_date}_{end_date}.yaml"
    _write_gee_payload(cfg_path, cfg)
    return str(cfg_path)


def update_gee_config(
    base_config_path: str,
    start_date: str,
    end_date: str,
    extent: list[float],
    out_dir: str,
    gee_project: str,
) -> str:
    cfg = _read_gee_config(base_config_path)
    return _write_gee_config(cfg, start_date, end_date, extent, out_dir, gee_project=gee_project)


def write_gee_config_from_cfg(
    gee_cfg: dict,
    start_date: str,
    end_date: str,
    extent: list[float],
    out_dir: str,
    gee_project: str,
) -> str:
    if not isinstance(gee_cfg, dict) or not gee_cfg:
        raise ValueError("gee config must be a non-empty dict")
    return _write_gee_config(dict(gee_cfg), start_date, end_date, extent, out_dir, gee_project=gee_project)


def prepare_landsat_inputs(
    *,
    date_range: str,
    extent: list[float],
    out_dir: str,
    gee_project: str,
    gee_config_template: str | None = None,
    sensors: tuple[str, ...] | list[str] = ("LC08", "LC09"),
) -> str:
    gee_project = _validate_gee_project(gee_project)
    start_date, end_date = parse_date_range(date_range)
    _safe_mkdir(out_dir)
    if gee_config_template is not None:
        cfg_path = update_gee_config(
            gee_config_template,
            start_date,
            end_date,
            extent,
            out_dir,
            gee_project=gee_project,
        )
        GEEDownloader(cfg_path).run()
    else:
        allowed = {"LT05", "LE07", "LC08", "LC09"}
        if not sensors or len(set(sensors)) != len(sensors) or set(sensors)-allowed:
            raise ValueError("sensors must be a nonempty unique selection of LT05, LE07, LC08, LC09")
        from pysweb.ssebop.quality import landsat_c2_l2_bands
        for sensor in sensors:
            cfg = copy.deepcopy(_DEFAULT_LANDSAT_GEE_CFG)
            cfg["collection"] = f"LANDSAT/{sensor}/C02/T1_L2"
            cfg["bands"] = list(landsat_c2_l2_bands(sensor))+["QA_PIXEL", "QA_RADSAT"]
            cfg["filename_prefix"] = f"Landsat_{sensor}"
            if sensor in {"LT05", "LE07"}:
                cfg["cloud_mask"]["bits"] = [0, 1, 3, 4, 5, 7]
            cfg_path = write_gee_config_from_cfg(cfg, start_date, end_date, extent, out_dir, gee_project)
            sensor_path = Path(cfg_path).with_name(f"gee_config_{sensor}_{start_date}_{end_date}.yaml")
            Path(cfg_path).replace(sensor_path)
            cfg_path = str(sensor_path)
            GEEDownloader(cfg_path).run()
    return cfg_path
