#!/usr/bin/env python3
"""
Script: 5_sweb_run_model.py
Objective: Run serial or parallel gridded SWB simulations using the shared soil-water solver.
Author: Yi Yu (with assistance from Codex)
Created: 2026-02-17
Last updated: 2026-04-17
Inputs: Forcing and soil NetCDFs, simulation dates, model settings and optional calibration or parameter files.
Outputs: A NetCDF of soil moisture, final states, water budgets and quality flags, plus a provenance manifest.
Usage: python workflows/5_sweb_run_model.py --help
Dependencies: Python standard library; pysweb
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pysweb.swb.run import build_parser, run_swb_workflow


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_swb_workflow(**vars(args))


if __name__ == "__main__":
    main()
