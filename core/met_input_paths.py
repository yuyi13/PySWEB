#!/usr/bin/env python3
"""
Script: met_input_paths.py
Objective: Provide a deprecated compatibility wrapper for package-owned meteorology path helpers.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-05-03
Inputs: Arguments forwarded to `pysweb.met.paths`.
Outputs: Delegated meteorology path resolution and variable-name inference.
Usage: import helpers from pysweb.met.paths instead.
Dependencies: pysweb.met.paths
"""
from __future__ import annotations

from pysweb.met.paths import *  # noqa: F401,F403
