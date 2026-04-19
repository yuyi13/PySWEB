#!/usr/bin/env python3
"""
Script: plot_time_series.py
Objective: Provide a legacy CLI wrapper for the package time-series plotting entrypoint.
Author: Yi Yu
Created: 2026-02-20
Last updated: 2026-04-19
Inputs: CLI arguments forwarded to `pysweb.visualisation.plot_time_series`.
Outputs: Delegated plotting side effects from the package entrypoint.
Usage: python visualisation/plot_time_series.py --help
Dependencies: pysweb.visualisation.plot_time_series
"""
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pysweb.visualisation.plot_time_series import main


if __name__ == "__main__":
    main()
