#!/usr/bin/env python3
"""
Script: plot_heatmap.py
Objective: Provide a legacy CLI wrapper for the package heatmap plotting entrypoint.
Author: Yi Yu
Created: 2026-02-20
Last updated: 2026-04-19
Inputs: CLI arguments forwarded to `pysweb.visualisation.plot_heatmap`.
Outputs: Delegated plotting side effects from the package entrypoint.
Usage: python visualisation/plot_heatmap.py --help
Dependencies: pysweb.visualisation.plot_heatmap
"""
from pysweb.visualisation.plot_heatmap import main


if __name__ == "__main__":
    main()
