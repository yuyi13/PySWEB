#!/usr/bin/env python3
"""
Script: 4_sweb_calib_domain.py
Objective: Provide a thin CLI wrapper around the package-owned SWB calibration workflow.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-04-19
Inputs: CLI options for prepared forcing, soil-property, and reference SSM calibration inputs.
Outputs: Delegated package-owned SWB calibration CSV output.
Usage: python workflows/4_sweb_calib_domain.py --help
Dependencies: pysweb
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pysweb.swb.calibrate import build_parser, calibrate_domain


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    calibrate_domain(**vars(args))


if __name__ == "__main__":
    main()
