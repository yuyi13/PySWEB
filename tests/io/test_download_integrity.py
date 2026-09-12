"""
Script: test_download_integrity.py
Objective: Verify failed downloads propagate and only validated raster identities are reused.
Author: Yi Yu (with assistance from Codex)
Created: 2026-09-12
Last updated: 2026-09-12
Inputs: Synthetic fixtures and package interfaces supplied by pytest.
Outputs: Regression assertions.
Usage: python -m pytest tests/io/test_download_integrity.py
Dependencies: pytest, numpy, pandas, xarray, pysweb
"""

from importlib import import_module
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import rasterio
import pytest


def test_download_failures_and_cache_validation(monkeypatch, tmp_path):
    mod = import_module("pysweb.io.gee_downloader")
    # Tests in this directory use isolated import shims; restore the real raster backend here.
    monkeypatch.setattr(mod, "rasterio", rasterio)
    obj = mod.GEEDownloader.__new__(mod.GEEDownloader)
    obj.cfg = {
        "download_dir": str(tmp_path),
        "collections": ["test/source"],
        "filename_prefix": "test",
        "out_format": "tif",
    }
    monkeypatch.setattr(obj, "initialize", lambda: None)
    monkeypatch.setattr(obj, "_unique_dates", lambda collection: ["2024-01-01"])
    image = SimpleNamespace(
        bandNames=lambda: SimpleNamespace(getInfo=lambda: ["value"])
    )
    monkeypatch.setattr(obj, "_composite_for_day", lambda day, collection: image)
    monkeypatch.setattr(obj, "_select_bands", lambda image: image)
    monkeypatch.setattr(mod, "_postprocess_geotiff", lambda *args, **kwargs: None)

    def fail(*args):
        raise OSError("interrupted transfer")

    monkeypatch.setattr(obj, "_download_image", fail)
    with pytest.raises(RuntimeError, match="incomplete"):
        obj.run()
    assert not list(tmp_path.glob("*.tif"))
    calls = []

    def download(image, path, bands):
        calls.append(path)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=1,
            width=1,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=rasterio.transform.from_origin(148.0, -35.0, 0.01, 0.01),
        ) as ds:
            ds.write(np.ones((1, 1, 1), dtype="float32"))

    monkeypatch.setattr(obj, "_download_image", download)
    obj.run()
    obj.run()
    assert len(calls) == 1
    output = tmp_path / "test_2024-01-01.tif"
    with output.open("ab") as stream:
        stream.write(b"tamper")
    obj.run()
    assert len(calls) == 2
