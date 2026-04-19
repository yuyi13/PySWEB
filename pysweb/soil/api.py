#!/usr/bin/env python3
"""
Script: api.py
Objective: Validate and dispatch soil property loading requests to the supported backend surface.
Author: Yi Yu
Created: 2026-04-19
Last updated: 2026-04-19
Inputs: Soil source selection, args namespace, grid object, and backend-specific kwargs.
Outputs: SoilOutputs dataclass instances from implemented backends and validation errors for unsupported placeholders.
Usage: Imported and called by package consumers.
Dependencies: dataclasses, numpy, xarray
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Dict

import numpy as np
import xarray as xr

SUPPORTED_SOIL_SOURCES = ("openlandmap", "mlcons", "slga", "custom")
IMPLEMENTED_SOIL_SOURCES = ("openlandmap",)
PLACEHOLDER_SOIL_SOURCES = tuple(
    soil_source for soil_source in SUPPORTED_SOIL_SOURCES if soil_source not in IMPLEMENTED_SOIL_SOURCES
)


@dataclass(frozen=True)
class SoilOutputs:
    arrays: Dict[str, xr.DataArray]
    layer_bottoms_mm: np.ndarray


def validate_soil_source(soil_source: str) -> None:
    if soil_source in IMPLEMENTED_SOIL_SOURCES:
        return
    if soil_source in PLACEHOLDER_SOIL_SOURCES:
        placeholders = ", ".join(PLACEHOLDER_SOIL_SOURCES)
        raise NotImplementedError(
            f"Soil backend '{soil_source}' is a placeholder and has not been implemented yet. "
            f"Implemented backends: {', '.join(IMPLEMENTED_SOIL_SOURCES)}. "
            f"Placeholders: {placeholders}."
        )
    supported_values = ", ".join(SUPPORTED_SOIL_SOURCES)
    raise ValueError(
        f"Unsupported soil_source '{soil_source}'. Supported values: {supported_values}."
    )


def load_soil_properties(*, soil_source: str, args, grid, **kwargs) -> SoilOutputs:
    validate_soil_source(soil_source)
    backend = import_module(f"pysweb.soil.{soil_source}")
    return backend.load_soil_properties(args=args, grid=grid, **kwargs)
