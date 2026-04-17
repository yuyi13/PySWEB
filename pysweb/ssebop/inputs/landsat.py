"""Landsat input preparation helpers for pysweb.ssebop."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from pysweb.io.gee import GEEDownloader, _safe_mkdir


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
) -> str:
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

    cfg_path = Path(out_dir) / f"gee_config_{start_date}_{end_date}.yaml"
    _write_gee_payload(cfg_path, cfg)
    return str(cfg_path)


def update_gee_config(
    base_config_path: str,
    start_date: str,
    end_date: str,
    extent: list[float],
    out_dir: str,
) -> str:
    cfg = _read_gee_config(base_config_path)
    return _write_gee_config(cfg, start_date, end_date, extent, out_dir)


def write_gee_config_from_cfg(
    gee_cfg: dict,
    start_date: str,
    end_date: str,
    extent: list[float],
    out_dir: str,
) -> str:
    if not isinstance(gee_cfg, dict) or not gee_cfg:
        raise ValueError("gee config must be a non-empty dict")
    return _write_gee_config(dict(gee_cfg), start_date, end_date, extent, out_dir)


def prepare_landsat_inputs(
    *,
    date_range: str,
    extent: list[float],
    gee_config: str,
    out_dir: str,
) -> str:
    start_date, end_date = parse_date_range(date_range)
    _safe_mkdir(out_dir)
    cfg_path = update_gee_config(gee_config, start_date, end_date, extent, out_dir)
    GEEDownloader(cfg_path).run()
    return cfg_path
