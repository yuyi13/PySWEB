#!/usr/bin/env python3
"""
Script: api.py
Objective: Validate and dispatch DEM preparation requests to supported backends.
Author: Yi Yu
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: DEM source selection, Earth Engine project name, and backend-specific kwargs.
Outputs: Backend dispatch results or explicit validation and placeholder errors.
Usage: Imported as `pysweb.dem.api`
Dependencies: importlib
"""

from importlib import import_module

SUPPORTED_DEM_SOURCES = ("nasadem",)


def validate_dem_source(dem_source: str) -> None:
    if dem_source in SUPPORTED_DEM_SOURCES:
        return
    supported_values = ", ".join(SUPPORTED_DEM_SOURCES)
    raise ValueError(
        f"Unsupported dem_source '{dem_source}'. Supported values: {supported_values}."
    )


def validate_gee_project(gee_project: str) -> None:
    if not isinstance(gee_project, str) or not gee_project.strip():
        raise ValueError("gee_project must be a non-empty string.")


def prepare_dem(*, dem_source: str, gee_project: str, **kwargs):
    validate_dem_source(dem_source)
    validate_gee_project(gee_project)
    backend = import_module(f"pysweb.dem.{dem_source}")
    return backend.prepare_dem(gee_project=gee_project, **kwargs)
