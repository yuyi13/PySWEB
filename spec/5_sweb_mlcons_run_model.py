#!/usr/bin/env python3
"""
Script: 5_sweb_mlcons_run_model.py
Objective: Run the canonical SWB solver using prepared local-soil and forcing inputs.
Author: Yi Yu (with assistance from Codex)
Created: 2026-02-22
Last updated: 2026-09-12
Inputs: Prepared forcing and soil NetCDFs, simulation settings and optional calibration parameters.
Outputs: A NetCDF of soil moisture, final states, water budgets and quality flags, plus a provenance manifest.
Usage: python spec/5_sweb_mlcons_run_model.py --help
Dependencies: Python standard library; pysweb
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pysweb.swb.run import build_parser, main


if __name__ == "__main__":
    main()
