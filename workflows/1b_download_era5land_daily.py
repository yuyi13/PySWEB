#!/usr/bin/env python3
"""
Script: 1b_download_era5land_daily.py
Objective: Write and run an ERA5-Land DAILY_AGGR GEE download config from CLI arguments.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-04-17
Inputs: --date-range, --extent, and --output-dir CLI arguments.
Outputs: ERA5-Land YAML config and downloaded daily GeoTIFFs.
Usage: python workflows/1b_download_era5land_daily.py --help
Dependencies: argparse, os, re, sys, pysweb.io.gee, pysweb.met.era5land.download
"""
from __future__ import annotations

import argparse
import os
import re
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pysweb.met.era5land.download import download_era5land_daily  # noqa: E402


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ERA5-Land DAILY_AGGR inputs from Google Earth Engine")
    parser.add_argument("--date-range", required=True, help="Date range string like '2024-01-01 to 2024-01-03'")
    parser.add_argument("--extent", required=True, help="min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--output-dir", required=True, help="Directory for config and downloaded GeoTIFFs")
    return parser.parse_args(argv)


def _resolve_downloader_cls():
    from pysweb.io.gee import GEEDownloader

    return GEEDownloader


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    start_date, end_date = parse_date_range(args.date_range)
    extent = parse_extent(args.extent)
    download_era5land_daily(
        start_date = start_date,
        end_date = end_date,
        extent = extent,
        output_dir = args.output_dir,
        downloader_cls = _resolve_downloader_cls(),
    )


if __name__ == "__main__":
    main()
