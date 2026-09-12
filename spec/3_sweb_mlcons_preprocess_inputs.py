#!/usr/bin/env python3
"""
Script: 3_sweb_mlcons_preprocess_inputs.py
Objective: Delegate the supported workflow to the canonical pysweb package.
Author: Yi Yu (with assistance from Codex)
Created: 2026-02-22
Last updated: 2026-09-12
Inputs: In-memory arrays and explicit workflow configuration.
Outputs: Validated data and numerical diagnostics.
Usage: Imported by pysweb workflows.
Dependencies: pysweb
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
