#!/usr/bin/env python3
"""
Script: era5land_stack.py
Objective: Discover ERA5-Land daily GeoTIFF files and sort them by embedded date.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-04-16
Inputs: A directory containing daily ERA5-Land GeoTIFF downloads.
Outputs: Sorted lists of ERA5-Land daily file paths.
Usage: import and call discover_daily_files(raw_dir).
Dependencies: pathlib, re, datetime
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
DAILY_RASTER_SUFFIXES = {".tif", ".tiff"}


def extract_date_from_path(path: Path) -> date:
    match = DATE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not find an embedded date in {path.name!r}.")
    return date.fromisoformat(match.group(1))


def discover_daily_files(raw_dir: Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    daily_files: list[Path] = []
    for path in raw_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in DAILY_RASTER_SUFFIXES:
            continue
        try:
            extract_date_from_path(path)
        except ValueError:
            continue
        daily_files.append(path)

    return sorted(daily_files, key=lambda path: (extract_date_from_path(path), path.name))
