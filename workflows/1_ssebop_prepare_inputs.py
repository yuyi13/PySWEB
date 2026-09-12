#!/usr/bin/env python3
"""
Script: 1_ssebop_prepare_inputs.py
Objective: Prepare Landsat, NASADEM and ERA5-Land inputs through the canonical SSEBop CLI.
Author: Yi Yu (with assistance from Codex)
Created: 2026-02-17
Last updated: 2026-09-12
Inputs: Dates, extent, Earth Engine project and input-preparation options.
Outputs: Downloaded rasters, meteorology NetCDFs and download configurations.
Usage: python workflows/1_ssebop_prepare_inputs.py --help
Dependencies: Python standard library; pysweb
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pysweb.ssebop.cli_prepare import build_parser, main


if __name__ == "__main__":
    main()
