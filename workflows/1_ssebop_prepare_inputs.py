#!/usr/bin/env python3
"""
Script: 1_ssebop_prepare_inputs.py
Objective: Parse CLI arguments and delegate the unified first SSEBop preparation step to the package API.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-04-17
Inputs: CLI arguments (--date-range, --extent, --met-source, --gee-config, --out-dir, --dem).
Outputs: Package-managed Landsat and meteorology inputs for the requested SSEBop run.
Usage: python workflows/1_ssebop_prepare_inputs.py --help
Dependencies: argparse, os, sys, pysweb.ssebop, pysweb.ssebop.inputs.landsat
"""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pysweb.ssebop import prepare_inputs  # noqa: E402
from pysweb.ssebop.inputs.landsat import parse_extent  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Landsat and meteorology inputs for SSEBop")
    parser.add_argument("--date-range", required=True, help="Date range string like '2024-01-01 to 2024-01-03'")
    parser.add_argument("--extent", required=True, help="min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--met-source", default="era5land", choices=["era5land", "silo"])
    parser.add_argument("--gee-config", required=True, help="Path to the base GEE config template")
    parser.add_argument("--out-dir", required=True, help="Base output directory for prepared inputs")
    parser.add_argument("--dem", required=True, help="DEM raster used for stacked meteorology inputs")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = os.path.abspath(args.out_dir)
    prepare_inputs(
        date_range = args.date_range,
        extent = parse_extent(args.extent),
        met_source = args.met_source,
        landsat_dir = os.path.join(out_dir, "landsat"),
        met_raw_dir = os.path.join(out_dir, "met", args.met_source, "raw"),
        met_stack_dir = os.path.join(out_dir, "met", args.met_source, "stack"),
        dem = args.dem,
        gee_config = args.gee_config,
    )


if __name__ == "__main__":
    main()
