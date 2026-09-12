"""
Script: paths.py
Objective: Expose SILO filename constants from the shared meteorology path helpers.
Author: Yi Yu (with assistance from Codex)
Created: 2026-04-17
Last updated: Unknown
Inputs: Imports of SILO path constants.
Outputs: Re-exported meteorology fields and SILO filename suffixes.
Usage: import pysweb.met.silo.paths
Dependencies: Python standard library; pysweb
"""

from pysweb.met.paths import SILO_FILENAME_SUFFIX

__all__ = ["SILO_FILENAME_SUFFIX"]
