#!/usr/bin/env python3
"""
Script: 1_ssebop_prepare_inputs.py
Objective: Prepare Landsat, ERA5-Land, DEM, and stacked meteorology inputs for the first SSEBop step.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-05-03
Inputs: CLI arguments (--date-range, --extent, --met-source, --gee-project, --out-dir).
Outputs: Package-managed Landsat, NASADEM, and meteorology inputs for the requested SSEBop run.
Usage: python workflows/1_ssebop_prepare_inputs.py --help
Dependencies: argparse, os, sys, pysweb.ssebop, pysweb.ssebop.landsat
"""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pysweb.ssebop import prepare_inputs  # noqa: E402
from pysweb.ssebop.landsat import parse_extent  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Landsat, NASADEM, ERA5-Land daily downloads, and "
            "stacked ERA5-Land meteorology inputs for SSEBop."
        )
    )
    parser.add_argument("--date-range", required=True, help="Date range string like '2024-01-01 to 2024-01-03'")
    parser.add_argument("--extent", required=True, help="min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--met-source", default="era5land", choices=["era5land"])
    parser.add_argument("--gee-project", required=True, help="Google Earth Engine project for Landsat and ERA5-Land downloads")
    parser.add_argument("--out-dir", required=True, help="Base output directory for prepared inputs")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    gee_project = args.gee_project.strip()
    if not gee_project:
        raise ValueError("gee_project must be a non-empty string.")
    out_dir = os.path.abspath(args.out_dir)
    prepare_inputs(
        date_range = args.date_range,
        extent = parse_extent(args.extent),
        met_source = args.met_source,
        gee_project = gee_project,
        landsat_dir = os.path.join(out_dir, "landsat"),
        met_raw_dir = os.path.join(out_dir, "met", args.met_source, "raw"),
        met_stack_dir = os.path.join(out_dir, "met", args.met_source, "stack"),
        dem_dir = os.path.join(out_dir, "dem"),
        dem_source = "nasadem",
        gee_config_template = None,
    )


if __name__ == "__main__":
    main()
