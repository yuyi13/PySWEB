#!/usr/bin/env python3
"""
Script: 3_sweb_preprocess_inputs.py
Objective: Align forcing, selected soil data and optional reference SSM on the SWB model grid.
Author: Yi Yu (with assistance from Codex)
Created: 2026-02-17
Last updated: 2026-04-19
Inputs: Dates, extent, forcing files, soil-source settings, optional reference SSM and output paths.
Outputs: Aligned forcing, soil-property and optional reference SSM NetCDFs with provenance manifests.
Usage: python workflows/3_sweb_preprocess_inputs.py --help
Dependencies: Python standard library; pysweb
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pysweb.swb.preprocess import build_parser, preprocess_inputs


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    preprocess_inputs(**vars(args))


if __name__ == "__main__":
    main()
