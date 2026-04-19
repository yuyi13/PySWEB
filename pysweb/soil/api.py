#!/usr/bin/env python3
"""
Script: api.py
Objective: Dispatch soil property loading requests to the supported backend surface.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Soil source selection, args namespace, grid object, and backend-specific kwargs.
Outputs: SoilOutputs dataclass instances from supported backends.
Usage: Imported and called by package consumers.
Dependencies: dataclasses, numpy, xarray
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Dict

import numpy as np
import xarray as xr

SUPPORTED_SOIL_SOURCES = ("openlandmap", "mlcons", "slga", "custom")


@dataclass(frozen=True)
class SoilOutputs:
    arrays: Dict[str, xr.DataArray]
    layer_bottoms_mm: np.ndarray


def load_soil_properties(*, soil_source: str, args, grid, **kwargs) -> SoilOutputs:
    if soil_source in SUPPORTED_SOIL_SOURCES:
        backend = import_module(f"pysweb.soil.{soil_source}")
        return backend.load_soil_properties(args=args, grid=grid, **kwargs)
    supported_values = ", ".join(SUPPORTED_SOIL_SOURCES)
    raise ValueError(
        f"Unsupported soil_source '{soil_source}'. Supported values: {supported_values}."
    )
