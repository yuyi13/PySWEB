"""
Script: cli.py
Objective: Expose installed command-line entry points for the canonical package.
Author: Yi Yu (with assistance from Codex)
Created: 2026-09-12
Last updated: 2026-09-12
Inputs: In-memory arrays and explicit workflow configuration.
Outputs: Validated data and numerical diagnostics.
Usage: Imported by pysweb workflows.
Dependencies: argparse, importlib, pysweb
"""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "ssebop-prepare": ("pysweb.ssebop.cli_prepare", "main"),
    "ssebop-run": ("pysweb.ssebop.cli_run", "main"),
    "swb-preprocess": ("pysweb.swb.preprocess", "main"),
    "swb-calibrate": ("pysweb.swb.calibrate", "main"),
    "swb-run": ("pysweb.swb.run", "main"),
    "time-series": ("pysweb.visualisation.plot_time_series", "main"),
    "heatmap": ("pysweb.visualisation.plot_heatmap", "main"),
    "workflow": ("pysweb.workflow", "main"),
    "demo": ("pysweb.demo", "main"),
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="PySWEB soil water-energy balance workflows"
    )
    parser.add_argument("command", choices=COMMANDS)
    if not args or args[0] in {"-h", "--help"}:
        parser.print_help()
        return
    command = parser.parse_args(args[:1]).command
    module_name, function = COMMANDS[command]
    getattr(importlib.import_module(module_name), function)(args[1:])


if __name__ == "__main__":
    main()
