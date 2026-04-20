#!/usr/bin/env python3
"""
Script: test_api_prepare_inputs.py
Objective: Verify the SSEBop package API orchestrates Landsat and meteorological input preparation steps.
Author: Yi Yu
Created: 2026-04-17
Last updated: 2026-04-20
Inputs: Temporary paths and monkeypatched package functions supplied by pytest.
Outputs: Test assertions.
Usage: pytest tests/ssebop/test_api_prepare_inputs.py
Dependencies: pytest
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pysweb.ssebop.api import prepare_inputs
from pysweb.ssebop.landsat import update_gee_config


def _load_config_payload(path: Path) -> dict:
    payload = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ModuleNotFoundError:
        return json.loads(payload)
    return yaml.safe_load(payload) or {}


def test_prepare_inputs_calls_landsat_era5land_dem_and_stack_steps(monkeypatch, tmp_path: Path):
    recorded = []

    monkeypatch.setattr(
        "pysweb.ssebop.landsat.prepare_landsat_inputs",
        lambda **kwargs: recorded.append(("landsat", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.dem.api.prepare_dem",
        lambda **kwargs: recorded.append(("dem", kwargs)) or kwargs["output_path"],
    )
    monkeypatch.setattr(
        "pysweb.met.era5land.download.download_era5land_daily",
        lambda **kwargs: recorded.append(("era5land_download", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.met.era5land.stack.stack_era5land_daily_inputs",
        lambda **kwargs: recorded.append(("era5land_stack", kwargs)),
    )

    prepare_inputs(
        date_range="2024-01-01 to 2024-01-03",
        extent=[147.2, -35.1, 147.3, -35.0],
        met_source="era5land",
        landsat_dir=str(tmp_path / "landsat"),
        met_raw_dir=str(tmp_path / "raw"),
        met_stack_dir=str(tmp_path / "stack"),
        dem_dir=str(tmp_path / "dem"),
        dem_source="nasadem",
        gee_config_template="/tmp/gee.yaml",
        gee_project="workflow-project",
    )

    assert recorded == [
        (
            "landsat",
            {
                "date_range": "2024-01-01 to 2024-01-03",
                "extent": [147.2, -35.1, 147.3, -35.0],
                "out_dir": str(tmp_path / "landsat"),
                "gee_project": "workflow-project",
                "gee_config_template": "/tmp/gee.yaml",
            },
        ),
        (
            "era5land_download",
            {
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "extent": [147.2, -35.1, 147.3, -35.0],
                "output_dir": str(tmp_path / "raw"),
                "gee_project": "workflow-project",
            },
        ),
        (
            "dem",
            {
                "dem_source": "nasadem",
                "gee_project": "workflow-project",
                "extent": [147.2, -35.1, 147.3, -35.0],
                "output_path": str(tmp_path / "dem" / "nasadem.tif"),
            },
        ),
        (
            "era5land_stack",
            {
                "raw_dir": str(tmp_path / "raw"),
                "dem": str(tmp_path / "dem" / "nasadem.tif"),
                "start_date": "2024-01-01",
                "end_date": "2024-01-03",
                "output_dir": str(tmp_path / "stack"),
            },
        ),
    ]


def test_prepare_inputs_rejects_unsupported_met_source_before_any_side_effects(monkeypatch, tmp_path: Path):
    recorded = []

    monkeypatch.setattr(
        "pysweb.ssebop.landsat.prepare_landsat_inputs",
        lambda **kwargs: recorded.append(("landsat", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.dem.api.prepare_dem",
        lambda **kwargs: recorded.append(("dem", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.met.era5land.download.download_era5land_daily",
        lambda **kwargs: recorded.append(("era5land_download", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.met.era5land.stack.stack_era5land_daily_inputs",
        lambda **kwargs: recorded.append(("era5land_stack", kwargs)),
    )

    try:
        prepare_inputs(
            date_range="2024-01-01 to 2024-01-03",
            extent=[147.2, -35.1, 147.3, -35.0],
            met_source="silo",
            landsat_dir=str(tmp_path / "landsat"),
            met_raw_dir=str(tmp_path / "raw"),
            met_stack_dir=str(tmp_path / "stack"),
            dem_dir=str(tmp_path / "dem"),
            dem_source="nasadem",
            gee_config_template="/tmp/gee.yaml",
            gee_project="workflow-project",
        )
    except NotImplementedError as exc:
        assert "Unsupported met_source: silo" == str(exc)
    else:
        raise AssertionError("Expected prepare_inputs to reject unsupported met_source values")

    assert recorded == []


def test_update_gee_config_forces_download_dir_to_requested_out_dir(tmp_path: Path):
    template_path = tmp_path / "template.json"
    out_dir = tmp_path / "landsat"
    out_dir.mkdir()
    template_path.write_text(
        (
            '{"collection":"LANDSAT/LC08/C02/T1_L2",'
            '"download_dir":"/tmp/old-dir",'
            '"coords":[0,0,1,1]}'
        ),
        encoding="utf-8",
    )

    cfg_path = update_gee_config(
        str(template_path),
        "2024-01-01",
        "2024-01-03",
        [147.2, -35.1, 147.3, -35.0],
        str(out_dir),
        "workflow-project",
    )

    payload = _load_config_payload(Path(cfg_path))
    assert payload["download_dir"] == str(out_dir)
    assert payload["coords"] == [147.2, -35.1, 147.3, -35.0]


def test_prepare_inputs_rejects_blank_gee_project_before_side_effects(monkeypatch, tmp_path: Path):
    recorded = []

    monkeypatch.setattr(
        "pysweb.ssebop.landsat.prepare_landsat_inputs",
        lambda **kwargs: recorded.append(("landsat", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.dem.api.prepare_dem",
        lambda **kwargs: recorded.append(("dem", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.met.era5land.download.download_era5land_daily",
        lambda **kwargs: recorded.append(("era5land_download", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.met.era5land.stack.stack_era5land_daily_inputs",
        lambda **kwargs: recorded.append(("era5land_stack", kwargs)),
    )

    try:
        prepare_inputs(
            date_range="2024-01-01 to 2024-01-03",
            extent=[147.2, -35.1, 147.3, -35.0],
            met_source="era5land",
            landsat_dir=str(tmp_path / "landsat"),
            met_raw_dir=str(tmp_path / "raw"),
            met_stack_dir=str(tmp_path / "stack"),
            dem_dir=str(tmp_path / "dem"),
            dem_source="nasadem",
            gee_config_template="/tmp/gee.yaml",
            gee_project="   ",
        )
    except ValueError as exc:
        assert "gee_project" in str(exc)
    else:
        raise AssertionError("Expected prepare_inputs to reject blank gee_project values")

    assert recorded == []


def test_prepare_inputs_rejects_reversed_date_range_before_any_side_effects(monkeypatch, tmp_path: Path):
    recorded = []

    monkeypatch.setattr(
        "pysweb.ssebop.landsat.prepare_landsat_inputs",
        lambda **kwargs: recorded.append(("landsat", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.dem.api.prepare_dem",
        lambda **kwargs: recorded.append(("dem", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.met.era5land.download.download_era5land_daily",
        lambda **kwargs: recorded.append(("era5land_download", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.met.era5land.stack.stack_era5land_daily_inputs",
        lambda **kwargs: recorded.append(("era5land_stack", kwargs)),
    )

    try:
        prepare_inputs(
            date_range="2024-01-03 to 2024-01-01",
            extent=[147.2, -35.1, 147.3, -35.0],
            met_source="era5land",
            landsat_dir=str(tmp_path / "landsat"),
            met_raw_dir=str(tmp_path / "raw"),
            met_stack_dir=str(tmp_path / "stack"),
            dem_dir=str(tmp_path / "dem"),
            dem_source="nasadem",
            gee_config_template="/tmp/gee.yaml",
            gee_project="workflow-project",
        )
    except ValueError as exc:
        assert "date_range" in str(exc)
    else:
        raise AssertionError("Expected prepare_inputs to reject reversed date ranges")

    assert recorded == []


def test_prepare_inputs_rejects_invalid_extent_before_any_side_effects(monkeypatch, tmp_path: Path):
    recorded = []

    monkeypatch.setattr(
        "pysweb.ssebop.landsat.prepare_landsat_inputs",
        lambda **kwargs: recorded.append(("landsat", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.dem.api.prepare_dem",
        lambda **kwargs: recorded.append(("dem", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.met.era5land.download.download_era5land_daily",
        lambda **kwargs: recorded.append(("era5land_download", kwargs)),
    )
    monkeypatch.setattr(
        "pysweb.met.era5land.stack.stack_era5land_daily_inputs",
        lambda **kwargs: recorded.append(("era5land_stack", kwargs)),
    )

    try:
        prepare_inputs(
            date_range="2024-01-01 to 2024-01-03",
            extent=[147.2, -35.1, 147.3],
            met_source="era5land",
            landsat_dir=str(tmp_path / "landsat"),
            met_raw_dir=str(tmp_path / "raw"),
            met_stack_dir=str(tmp_path / "stack"),
            dem_dir=str(tmp_path / "dem"),
            dem_source="nasadem",
            gee_config_template="/tmp/gee.yaml",
            gee_project="workflow-project",
        )
    except ValueError as exc:
        assert "extent" in str(exc)
    else:
        raise AssertionError("Expected prepare_inputs to reject malformed extents")

    assert recorded == []
