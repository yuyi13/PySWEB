#!/usr/bin/env python3
"""
Script: 1_ssebop_prepare_inputs.py
Objective: Prepare and validate GEE/SSEBop input configuration from date range and extent arguments.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-02-25
Inputs: CLI arguments (--config, --date-range, --extent, --gee-config, --out-dir) and optional base YAML config.
Outputs: Updated GEE config YAML and prepared input directory for SSEBop downloads.
Usage: python workflows/1_ssebop_prepare_inputs.py --help
Dependencies: argparse, pyyaml, re, datetime, core/gee_downloader.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from typing import List, Tuple

import yaml

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORE_DIR = os.path.join(PROJECT_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from gee_downloader import GEEDownloader, _safe_mkdir  # noqa: E402

def parse_date_range(date_range: str) -> Tuple[str, str]:
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", date_range)
    if len(dates) != 2:
        raise ValueError("Date range must include two dates in YYYY-MM-DD format.")
    start, end = dates
    return start, end


def parse_extent(extent_str: str) -> List[float]:
    parts = re.split(r"[,\s]+", extent_str.strip())
    if len(parts) != 4:
        raise ValueError("Extent must be four numbers: min_lon,min_lat,max_lon,max_lat")
    return [float(p) for p in parts]


def update_gee_config(
    base_config_path: str,
    start_date: str,
    end_date: str,
    extent: List[float],
    out_dir: str,
) -> str:
    with open(base_config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return _write_gee_config(cfg, start_date, end_date, extent, out_dir)


def write_gee_config_from_cfg(
    gee_cfg: dict,
    start_date: str,
    end_date: str,
    extent: List[float],
    out_dir: str,
) -> str:
    if not isinstance(gee_cfg, dict) or not gee_cfg:
        raise ValueError("gee config must be a non-empty dict")
    return _write_gee_config(gee_cfg, start_date, end_date, extent, out_dir)


def _write_gee_config(
    cfg: dict,
    start_date: str,
    end_date: str,
    extent: List[float],
    out_dir: str,
) -> str:
    if "coords" not in cfg or cfg["coords"] is None:
        cfg["coords"] = extent
    if "download_dir" not in cfg or cfg["download_dir"] is None:
        cfg["download_dir"] = out_dir

    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)

    cfg["coords"] = extent
    cfg["start_year"] = start_dt.year
    cfg["start_month"] = start_dt.month
    cfg["start_day"] = start_dt.day
    cfg["end_year"] = end_dt.year
    cfg["end_month"] = end_dt.month
    cfg["end_day"] = end_dt.day

    cfg_name = f"gee_config_{start_date}_{end_date}.yaml"
    cfg_path = os.path.join(out_dir, cfg_name)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GEE inputs for SSEBop")
    parser.add_argument("--config", help="YAML config file with input parameters")
    parser.add_argument("--date-range", help="Date range string with two dates (YYYY-MM-DD)")
    parser.add_argument("--extent", help="min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--gee-config", default=None)
    parser.add_argument("--out-dir", default="/g/data/ym05/sweb_model/0_auxiliary/ssebop_inputs")
    args = parser.parse_args()

    cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    date_range = args.date_range or cfg.get("date_range")
    if not date_range:
        date_range = input("Enter date range (e.g., 2020-01-01 to 2020-12-31): ").strip()
    start_date, end_date = parse_date_range(date_range)

    extent_str = args.extent or cfg.get("extent")
    if not extent_str:
        extent_str = input("Enter extent (min_lon,min_lat,max_lon,max_lat): ").strip()
    extent = parse_extent(extent_str)

    gee_config = cfg.get("gee_config", args.gee_config)
    gee_inline = cfg.get("gee", None)
    out_dir = cfg.get("out_dir", args.out_dir)
    _safe_mkdir(out_dir)

    print("[gee] updating config and downloading Landsat data...")
    if gee_inline:
        cfg_path = write_gee_config_from_cfg(gee_inline, start_date, end_date, extent, out_dir)
    elif gee_config:
        cfg_path = update_gee_config(gee_config, start_date, end_date, extent, out_dir)
    else:
        raise ValueError("Provide either gee config under 'gee' or --gee-config.")
    GEEDownloader(cfg_path).run()

if __name__ == "__main__":
    main()
