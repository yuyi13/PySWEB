#!/usr/bin/env python3
"""
Script: 6_plot_results.py
Objective: Dispatch heatmap and time-series plotting commands to PySWEB visualisation modules.
Author: Yi Yu (with assistance from Codex)
Created: 2026-05-03
Last updated: 2026-05-03
Inputs: Plot subcommand, model NetCDF paths, extraction settings and output paths.
Outputs: Saved figures and optional CSV files of extracted plotting data.
Usage: python workflows/6_plot_results.py --help
Dependencies: Python standard library; pysweb
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pysweb.visualisation import plot_heatmap, plot_time_series


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run optional PySWEB post-processing plots."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("heatmap", "time-series"),
        help="Plot type to run. Remaining arguments are forwarded to that plotter.",
    )

    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] in {"-h", "--help"}:
        args = parser.parse_args(tokens)
        if args.command is None:
            parser.error("Provide a plotting command: heatmap or time-series.")
        args.plot_args = []
        return args

    args = parser.parse_args(tokens[:1])
    args.plot_args = tokens[1:]
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    forwarded = list(args.plot_args)

    if args.command == "heatmap":
        plot_heatmap.main(forwarded)
        return 0
    if args.command == "time-series":
        plot_time_series.main(forwarded)
        return 0

    raise ValueError(f"Unsupported plotting command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
