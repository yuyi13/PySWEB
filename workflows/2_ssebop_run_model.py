#!/usr/bin/env python3
"""
Script: 2_ssebop_run_model.py
Objective: Provide a thin CLI wrapper around the package-owned SSEBop run API.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-04-17
Inputs: YAML config/CLI options, Landsat GeoTIFFs, meteorology NetCDF files, DEM, landcover raster.
Outputs: Daily SSEBop ET NetCDF outputs and optional gap-filled ETf diagnostics in output directory.
Usage: python workflows/2_ssebop_run_model.py --help
Dependencies: argparse, pysweb
"""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pysweb.ssebop.api import run_ssebop_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SSEBop with local Landsat scenes and meteorology inputs")
    parser.add_argument("config_pos", nargs="?", help="YAML config file with input parameters")
    parser.add_argument("--config", help="YAML config file with input parameters")
    parser.add_argument("--date-range", default=None)
    parser.add_argument("--silo-dir", default=None, help="Directory containing legacy SILO yearly NetCDF inputs.")
    parser.add_argument("--met-dir", default=None, help="Directory containing ERA5-Land daily stack NetCDF inputs.")
    parser.add_argument("--landsat-dir", default=None)
    parser.add_argument("--landsat-pattern", default="*.tif")
    parser.add_argument("--lst-band", default="lst")
    parser.add_argument("--ndvi-band", default="ndvi")
    parser.add_argument("--red-band", default="red")
    parser.add_argument("--nir-band", default="nir")
    parser.add_argument("--et-short-crop", default=None)
    parser.add_argument("--et-short-crop-var", default=None)
    parser.add_argument("--tmax", default=None)
    parser.add_argument("--tmax-var", default=None)
    parser.add_argument("--tmin", default=None)
    parser.add_argument("--tmin-var", default=None)
    parser.add_argument("--rs", default=None)
    parser.add_argument("--rs-var", default=None)
    parser.add_argument("--ea", default=None)
    parser.add_argument("--ea-var", default=None)
    parser.add_argument("--dem", default=None)
    parser.add_argument("--landcover", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-gap-days", type=int, default=32)
    parser.add_argument("--apply-water-mask", action="store_true")
    parser.add_argument(
        "--met-temp-units",
        choices=["celsius", "kelvin"],
        default="celsius",
        help="Temperature units for meteorology inputs.",
    )
    parser.add_argument(
        "--silo-temp-units",
        dest="met_temp_units",
        choices=["celsius", "kelvin"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--gapfill-etf", action="store_true")
    parser.add_argument("--gapfill-window-days", type=int, default=None)
    parser.add_argument("--gapfill-min-samples", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_ssebop_workflow(**vars(args))


if __name__ == "__main__":
    main()
