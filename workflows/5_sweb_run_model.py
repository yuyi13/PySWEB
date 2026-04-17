#!/usr/bin/env python3
"""
Script: 5_sweb_run_model.py
Objective: Provide a thin CLI wrapper around the package-owned SWB run workflow.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-04-17
Inputs: CLI options for forcing NetCDFs, soil-property NetCDFs, and optional NDVI or parameter grids.
Outputs: Delegated package-owned SWB run outputs in the requested output directory.
Usage: python workflows/5_sweb_run_model.py --help
Dependencies: argparse, pysweb
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
