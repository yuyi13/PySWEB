#!/usr/bin/env python3
"""
Script: gee_downloader.py
Objective: Provide a deprecated compatibility wrapper for the package-owned Earth Engine downloader.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-05-03
Inputs: YAML configuration file and CLI arguments forwarded to `pysweb.io.gee_downloader`.
Outputs: Delegated GeoTIFF downloads from the package entrypoint.
Usage: python core/gee_downloader.py <config.yaml>
Dependencies: pysweb.io.gee_downloader
"""
from __future__ import annotations

from pysweb.io.gee_downloader import *  # noqa: F401,F403
from pysweb.io.gee_downloader import _safe_mkdir
from pysweb.io.gee_downloader import main


if __name__ == "__main__":
    main()
