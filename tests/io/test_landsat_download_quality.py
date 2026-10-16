#!/usr/bin/env python3
"""
Script: test_landsat_download_quality.py
Objective: Prevent missing QA bands, bitfield medians and zero-QA loss during downloads.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-16
Last updated: 2026-10-16
Inputs: Synthetic download configurations and missing-band stubs, without remote calls.
Outputs: Regression assertions.
Usage: python -m pytest tests/io/test_landsat_download_quality.py
Dependencies: pytest; pysweb
"""
from importlib import import_module
import pytest


def downloader(tmp_path, **updates):
    mod = import_module("pysweb.io.gee_downloader")
    obj = mod.GEEDownloader.__new__(mod.GEEDownloader)
    obj.cfg = {"coords": [0, 0, 1, 1], "download_dir": str(tmp_path / "data"),
               "start_year": 2017, "start_month": 1, "start_day": 1,
               "end_year": 2017, "end_month": 1, "end_day": 2,
               "bands": ["ST_B10", "QA_PIXEL", "QA_RADSAT"], "scale": 30,
               "out_format": "tif", "auth_mode": "browser", "filename_prefix": "Landsat",
               "collection": "LANDSAT/LC08/C02/T1_L2", "daily_strategy": "first",
               "postprocess": {"maskval_to_na": False}, **updates}
    return obj


def test_qa_download_rejects_bitfield_medians(tmp_path):
    with pytest.raises(ValueError, match="bitfields"):
        downloader(tmp_path, daily_strategy="median")._validate_config()


def test_qa_download_rejects_zero_to_nodata(tmp_path):
    with pytest.raises(ValueError, match="QA zeros"):
        downloader(tmp_path, postprocess={"maskval_to_na": True})._validate_config()
    downloader(tmp_path)._validate_config()


def test_explicit_missing_mask_band_does_not_pass_through():
    mod = import_module("pysweb.io.gee_downloader")
    class MissingBand:
        def select(self, names):
            raise ValueError("QA_PIXEL absent")
    with pytest.raises(ValueError, match="QA_PIXEL absent"):
        mod.build_mask_condition(MissingBand(), {"cloud_mask": {
            "enabled": True, "band": "QA_PIXEL", "type": "bits_any", "bits": [3]}})


def test_select_bands_requires_every_requested_band(tmp_path):
    class MissingBand:
        def bandNames(self):
            return self
        def getInfo(self):
            return ["ST_B10", "QA_PIXEL"]
    with pytest.raises(RuntimeError, match="QA_RADSAT"):
        downloader(tmp_path)._select_bands(MissingBand())
