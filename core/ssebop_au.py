#!/usr/bin/env python3
"""
Script: ssebop_au.py
Objective: Provide a deprecated compatibility wrapper for package-owned SSEBop helper functions.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-05-03
Inputs: Arguments forwarded to `pysweb.ssebop`, `pysweb.met.silo`, and package landcover helpers.
Outputs: Delegated SSEBop arrays, SILO variables, landcover masks, and config objects.
Usage: import helpers from pysweb.ssebop and pysweb.met.silo instead.
Dependencies: pysweb.ssebop, pysweb.met.silo
"""
from __future__ import annotations

from pysweb.met.silo.readers import open_silo_et_short_crop, open_silo_variable
from pysweb.ssebop.core import *  # noqa: F401,F403
from pysweb.ssebop.core import AU_SSEBOP_SOURCE_CANDIDATES, SsebopAuConfig
from pysweb.ssebop.grid import reproject_match, reproject_match_crop_first
from pysweb.ssebop.landcover import (
    load_worldcover_landcover as _load_worldcover_landcover,
    worldcover_masks,
)


def load_worldcover_landcover(path: str | None = None, masked: bool = True):
    """Load ESA WorldCover using the AU default path for legacy callers."""
    lc_path = path or AU_SSEBOP_SOURCE_CANDIDATES["landcover"]["local"]
    return _load_worldcover_landcover(lc_path, masked=masked)
