#!/usr/bin/env python3
"""
Script: met_input_paths.py
Objective: Resolve meteorology input paths and infer NetCDF variable names for SILO and ERA5-Land stacks.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-04-16
Inputs: Meteorology source options, date ranges, and file paths.
Outputs: Resolved file paths and inferred variable names.
Usage: import and call the helper functions directly.
Dependencies: pathlib, re
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Union

DATE_RANGE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
ERA5LAND_DAILY_STACK_PATTERN = re.compile(
    r"^(?P<field>[A-Za-z0-9_]+)_daily_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}$"
)
SILO_YEAR_FILE_PATTERN = re.compile(r"^\d{4}\.(?P<field>[^.]+)$")

MET_FIELDS = ("et_short_crop", "tmax", "tmin", "rs", "ea")
SILO_FILENAME_SUFFIX = {
    "et_short_crop": "et_short_crop.nc",
    "tmax": "max_temp.nc",
    "tmin": "min_temp.nc",
    "rs": "radiation.nc",
    "ea": "vp.nc",
}


def parse_date_range(date_range: str) -> tuple[str, str]:
    dates = DATE_RANGE_PATTERN.findall(date_range)
    if len(dates) != 2:
        raise ValueError("date_range must include two dates in YYYY-MM-DD format.")
    return dates[0], dates[1]


def years_in_date_range(date_range: str) -> List[int]:
    start_date, end_date = parse_date_range(date_range)
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    return list(range(start_year, end_year + 1))


def infer_met_var_from_path(path: str) -> Optional[str]:
    stem = Path(path).stem
    match = ERA5LAND_DAILY_STACK_PATTERN.match(stem)
    if match is not None:
        return match.group("field")

    match = SILO_YEAR_FILE_PATTERN.match(stem)
    if match is not None:
        return match.group("field")

    return stem or None


def resolve_met_input_paths(
    field: str,
    explicit_path: Optional[str],
    met_dir: Optional[str],
    silo_dir: Optional[str],
    date_range: Optional[str],
) -> Optional[Union[str, Sequence[str]]]:
    if field not in MET_FIELDS:
        raise ValueError(f"Unsupported meteorology field: {field}")

    if explicit_path:
        return explicit_path

    if met_dir:
        if not date_range:
            raise ValueError(
                f"--met-dir requires --date-range to infer {field} daily stack filenames."
            )
        start_date, end_date = parse_date_range(date_range)
        return str(Path(met_dir) / f"{field}_daily_{start_date}_{end_date}.nc")

    if silo_dir and date_range:
        return [
            str(Path(silo_dir) / f"{year}.{SILO_FILENAME_SUFFIX[field]}")
            for year in years_in_date_range(date_range)
        ]

    return None
