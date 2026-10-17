#!/usr/bin/env python3
"""
Script: test_raster_grid.py
Objective: Verify exact-window selection and reject incompatible raster grids.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Small synthetic raster grids and coordinate metadata.
Outputs: Pytest assertions for safe extraction without interpolation.
Usage: pytest tests/test_raster_grid.py
Dependencies: numpy, pytest, rasterio, pysweb.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio import Affine
from rasterio.io import MemoryFile

from pysweb.io.raster_grid import aligned_grid_window


def grid(transform=None, **kwargs):
    return SimpleNamespace(**dict(dict(crs="EPSG:4326", width=9, height=9,
        transform=transform or Affine(.1, 0, 149.4, 0, -.1, -29.8)), **kwargs))


def test_offset_window_preserves_original_cells():
    values = np.arange(100, dtype="float32").reshape(10, 10)
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", height=10, width=10, count=1,
                         dtype="float32", crs="EPSG:4326",
                         transform=Affine(.1, 0, 149.3, 0, -.1, -29.7)) as src:
            src.write(values, 1)
            window = aligned_grid_window(src, grid(crs=src.crs))
            assert window.col_off == window.row_off == 1
            np.testing.assert_array_equal(src.read(1, window=window), values[1:, 1:])


def test_float_serialization_roundoff_is_accepted():
    source = grid(Affine(.1, 0, 149.3, 0, -.1, -29.700000000000003), width=10, height=10)
    target = grid(Affine(.1, 0, 149.40000000000003, 0, -.1, -29.80000000000001))
    assert aligned_grid_window(source, target).col_off == 1


@pytest.mark.parametrize("target", [
    grid(Affine(.1, 0, 149.35, 0, -.1, -29.8)),
    grid(Affine(.2, 0, 149.4, 0, -.1, -29.8)),
    grid(Affine(.1, .001, 149.4, 0, -.1, -29.8)),
    grid(width=11), grid(crs="EPSG:3857"), grid(crs=None),
])
def test_incompatible_grid_rejected(target):
    with pytest.raises(ValueError):
        aligned_grid_window(grid(), target)
