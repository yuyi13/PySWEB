#!/usr/bin/env python3
"""
Script: 2_ssebop_run_model.py
Objective: Run SSEBop through the canonical package CLI.
Author: Yi Yu (with assistance from Codex)
Created: 2026-02-17
Last updated: 2026-09-12
Inputs: Landsat rasters, meteorology NetCDFs, DEM, optional landcover and CLI or YAML settings.
Outputs: SSEBop ET products, optional interpolation diagnostics and provenance manifests.
Usage: python workflows/2_ssebop_run_model.py --help
Dependencies: Python standard library; pysweb
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pysweb.ssebop.cli_run import build_parser, main


if __name__ == "__main__":
    main()
