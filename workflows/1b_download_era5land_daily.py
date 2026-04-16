#!/usr/bin/env python3
"""
Script: 1b_download_era5land_daily.py
Objective: Write and run an ERA5-Land DAILY_AGGR GEE download config from CLI arguments.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-04-16
Inputs: --date-range, --extent, and --output-dir CLI arguments.
Outputs: ERA5-Land YAML config and downloaded daily GeoTIFFs.
Usage: python workflows/1b_download_era5land_daily.py --help
Dependencies: argparse, os, re, sys, yaml, core.era5land_download_config, core.gee_downloader
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.era5land_download_config import build_era5land_cfg  # noqa: E402
from core.gee_downloader import GEEDownloader  # noqa: E402


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


def write_era5land_config(
    start_date: str,
    end_date: str,
    extent: list[float],
    output_dir: str,
) -> Path:
    cfg = build_era5land_cfg(start_date=start_date, end_date=end_date, extent=extent, out_dir=output_dir)
    cfg_path = Path(output_dir) / f"gee_config_era5land_{start_date}_{end_date}.yaml"
    os.makedirs(output_dir, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return cfg_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ERA5-Land DAILY_AGGR inputs from Google Earth Engine")
    parser.add_argument("--date-range", required=True, help="Date range string like '2024-01-01 to 2024-01-03'")
    parser.add_argument("--extent", required=True, help="min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--output-dir", required=True, help="Directory for config and downloaded GeoTIFFs")
    args = parser.parse_args()

    start_date, end_date = parse_date_range(args.date_range)
    extent = parse_extent(args.extent)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    cfg_path = write_era5land_config(start_date, end_date, extent, output_dir)
    GEEDownloader(str(cfg_path)).run()


if __name__ == "__main__":
    main()
