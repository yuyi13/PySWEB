#!/usr/bin/env python3
"""
Script: 3_sweb_mlcons_preprocess_inputs.py
Objective: Run canonical SWB preprocessing with MLConstraints selected as the default soil source.
Author: Yi Yu (with assistance from Codex)
Created: 2026-02-22
Last updated: 2026-09-12
Inputs: Forcing files, local soil maps, dates, extent and preprocessing options.
Outputs: Aligned forcing, soil-property and optional reference SSM NetCDFs with provenance manifests.
Usage: python spec/3_sweb_mlcons_preprocess_inputs.py --help
Dependencies: Python standard library; pysweb
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pysweb.swb.preprocess import build_parser, main as package_main

def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--soil-source" not in arguments:
        arguments = ["--soil-source", "mlcons", *arguments]
    package_main(arguments)


if __name__ == "__main__":
    main()
