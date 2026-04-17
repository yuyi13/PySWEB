#!/usr/bin/env python3
"""
Script: 1c_stack_era5land_daily.py
Objective: Stack ERA5-Land daily GeoTIFFs into daily NetCDF forcing products and derive short-reference ET.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-04-17
Inputs: Daily ERA5-Land GeoTIFFs, a DEM GeoTIFF, and a date range.
Outputs: Daily NetCDF products for precipitation, temperature, radiation, vapor pressure, and reference ET.
Usage: python workflows/1c_stack_era5land_daily.py --help
Dependencies: os, sys, pysweb.met.era5land.stack
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pysweb.met.era5land.stack import parse_args as package_parse_args  # noqa: E402
from pysweb.met.era5land.stack import stack_era5land_daily_inputs  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return package_parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    stack_era5land_daily_inputs(
        raw_dir = args.raw_dir,
        dem = args.dem,
        start_date = args.date_range[0],
        end_date = args.date_range[1],
        output_dir = args.output_dir,
    )


if __name__ == "__main__":
    main()
