#!/usr/bin/env python3
"""
Script: evaluate_site.py
Objective: Dispatch portable site workflow stages through the package implementation.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-23
Last updated: 2026-10-23
Inputs: Site plan JSON and explicit stage arguments.
Outputs: Requested site stage products and acceptance receipts.
Usage: python workflows/evaluate_site.py --plan site.json
Dependencies: pysweb and its scientific runtime.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pysweb.site_workflow import main

if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:] + ['--stages', 'evaluate']))
