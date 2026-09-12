#!/usr/bin/env python3
"""
Script: 2_ssebop_run_model.py
Objective: Delegate the supported workflow to the canonical pysweb package.
Author: Yi Yu (with assistance from Codex)
Created: 2026-02-17
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

from pysweb.ssebop.cli_run import build_parser, main


if __name__ == "__main__":
    main()
