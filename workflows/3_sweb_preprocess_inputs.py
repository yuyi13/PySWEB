#!/usr/bin/env python3
"""
Script: 3_sweb_preprocess_inputs.py
Objective: Provide a thin CLI wrapper around the package-owned SWB preprocess workflow.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-04-19
Inputs: CLI options for forcing preparation, Earth Engine soil inputs, and reference SSM preparation.
Outputs: Delegated package-owned SWB preprocess outputs in the requested output directory.
Usage: python workflows/3_sweb_preprocess_inputs.py --help
Dependencies: pysweb
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
