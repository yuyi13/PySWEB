#!/usr/bin/env python3
"""
Script: era5land_download_config.py
Objective: Provide a deprecated compatibility wrapper for package-owned ERA5-Land download helpers.
Author: Yi Yu
Created: 2026-04-16
Last updated: 2026-05-03
Inputs: Arguments forwarded to `pysweb.met.era5land.download`.
Outputs: Delegated ERA5-Land download config dictionaries and config files.
Usage: import helpers from pysweb.met.era5land.download instead.
Dependencies: pysweb.met.era5land.download
"""
from __future__ import annotations

from pysweb.met.era5land.download import *  # noqa: F401,F403
