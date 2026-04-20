#!/usr/bin/env python3
"""
Script: test_landsat.py
Objective: Verify the canonical SSEBop Landsat helper module and its compatibility shim preserve config semantics.
Author: Yi Yu
Created: 2026-04-20
Last updated: 2026-04-20
Inputs: Temporary config templates and package imports supplied by pytest.
Outputs: Test assertions.
Usage: pytest tests/ssebop/test_landsat.py
Dependencies: pytest, pyyaml
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pysweb.ssebop.landsat as canonical_landsat
import pysweb.ssebop.inputs.landsat as shim_landsat


def _load_config_payload(path: Path) -> dict:
    payload = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ModuleNotFoundError:
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            parsed = {}
            current_key = None
            for raw_line in payload.splitlines():
                line = raw_line.rstrip()
                if not line:
                    continue
                if line.startswith("  - "):
                    parsed.setdefault(current_key, []).append(_coerce_yaml_scalar(line[4:]))
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value:
                    parsed[key] = _coerce_yaml_scalar(value)
                    current_key = None
                else:
                    parsed[key] = []
                    current_key = key
            return parsed
    return yaml.safe_load(payload)


def _coerce_yaml_scalar(value):
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    if value == "null" or value == "~":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def test_canonical_landsat_module_exports_helper_contract(tmp_path: Path):
    out_dir = tmp_path / "landsat"

    cfg_path = canonical_landsat.write_gee_config_from_cfg(
        gee_cfg={
            "collection": "LANDSAT/LC08/C02/T1_L2",
            "auth_mode": "browser",
        },
        start_date="2024-01-01",
        end_date="2024-01-03",
        extent=[147.2, -35.1, 147.3, -35.0],
        out_dir=str(out_dir),
        gee_project="project-from-canonical-module",
    )

    payload = _load_config_payload(Path(cfg_path))
    assert payload["gee_project"] == "project-from-canonical-module"
    assert payload["download_dir"] == str(out_dir)


def test_legacy_shim_re_exports_canonical_landsat_helpers():
    assert shim_landsat.parse_date_range is canonical_landsat.parse_date_range
    assert shim_landsat.parse_extent is canonical_landsat.parse_extent
    assert shim_landsat.update_gee_config is canonical_landsat.update_gee_config
    assert shim_landsat.write_gee_config_from_cfg is canonical_landsat.write_gee_config_from_cfg
    assert shim_landsat.prepare_landsat_inputs is canonical_landsat.prepare_landsat_inputs


def test_update_gee_config_preserves_yaml_template_support_and_injects_project(tmp_path: Path):
    template_path = tmp_path / "template.yaml"
    out_dir = tmp_path / "landsat"
    template_path.write_text(
        "\n".join(
            [
                "collection: LANDSAT/LC08/C02/T1_L2",
                "bands:",
                "  - ST_B10",
                "  - SR_B4",
                "  - SR_B5",
                "auth_mode: browser",
                "scale: 30",
                "out_format: tif",
                "filename_prefix: Landsat",
                "download_dir: /tmp/original",
                "coords:",
                "  - 0",
                "  - 0",
                "  - 1",
                "  - 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_yaml_safe_load(payload):
        if hasattr(payload, "read"):
            payload = payload.read()

        parsed = {}
        current_key = None
        for raw_line in payload.splitlines():
            line = raw_line.rstrip()
            if not line:
                continue
            if line.startswith("  - "):
                parsed.setdefault(current_key, []).append(_coerce_yaml_scalar(line[4:]))
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                parsed[key] = _coerce_yaml_scalar(value)
                current_key = None
            else:
                parsed[key] = []
                current_key = key
        return parsed

    def fake_yaml_safe_dump(payload, sort_keys=False):
        keys = sorted(payload) if sort_keys else list(payload)
        lines = []
        for key in keys:
            value = payload[key]
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {json.dumps(item) if isinstance(item, str) else item}")
                continue
            if isinstance(value, str):
                lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines) + "\n"

    fake_yaml = type(
        "FakeYaml",
        (),
        {
            "safe_load": staticmethod(fake_yaml_safe_load),
            "safe_dump": staticmethod(fake_yaml_safe_dump),
        },
    )
    original_loader = canonical_landsat._load_yaml_module
    canonical_landsat._load_yaml_module = lambda: fake_yaml
    try:
        cfg_path = canonical_landsat.update_gee_config(
            base_config_path=str(template_path),
            start_date="2024-01-01",
            end_date="2024-01-03",
            extent=[147.2, -35.1, 147.3, -35.0],
            out_dir=str(out_dir),
            gee_project="yaml-template-project",
        )
    finally:
        canonical_landsat._load_yaml_module = original_loader

    payload = _load_config_payload(Path(cfg_path))
    assert payload["gee_project"] == "yaml-template-project"
    assert payload["download_dir"] == str(out_dir)
    assert payload["coords"] == [147.2, -35.1, 147.3, -35.0]


def test_update_gee_config_requires_explicit_gee_project(tmp_path: Path):
    template_path = tmp_path / "template.json"
    template_path.write_text(
        json.dumps(
            {
                "collection": "LANDSAT/LC08/C02/T1_L2",
                "auth_mode": "browser",
                "download_dir": "/tmp/original",
                "coords": [0, 0, 1, 1],
            }
        ),
        encoding="utf-8",
    )

    try:
        canonical_landsat.update_gee_config(
            base_config_path=str(template_path),
            start_date="2024-01-01",
            end_date="2024-01-03",
            extent=[147.2, -35.1, 147.3, -35.0],
            out_dir=str(tmp_path / "landsat"),
        )
    except TypeError as exc:
        assert "gee_project" in str(exc)
    else:
        raise AssertionError("Expected update_gee_config to require gee_project explicitly")


def test_write_gee_config_from_cfg_requires_explicit_gee_project(tmp_path: Path):
    try:
        canonical_landsat.write_gee_config_from_cfg(
            gee_cfg={
                "collection": "LANDSAT/LC08/C02/T1_L2",
                "auth_mode": "browser",
            },
            start_date="2024-01-01",
            end_date="2024-01-03",
            extent=[147.2, -35.1, 147.3, -35.0],
            out_dir=str(tmp_path / "landsat"),
        )
    except TypeError as exc:
        assert "gee_project" in str(exc)
    else:
        raise AssertionError("Expected write_gee_config_from_cfg to require gee_project explicitly")


def test_update_gee_config_rejects_blank_gee_project(tmp_path: Path):
    template_path = tmp_path / "template.json"
    out_dir = tmp_path / "landsat"
    template_path.write_text(
        json.dumps(
            {
                "collection": "LANDSAT/LC08/C02/T1_L2",
                "auth_mode": "browser",
                "download_dir": "/tmp/original",
                "coords": [0, 0, 1, 1],
            }
        ),
        encoding="utf-8",
    )

    try:
        canonical_landsat.update_gee_config(
            base_config_path=str(template_path),
            start_date="2024-01-01",
            end_date="2024-01-03",
            extent=[147.2, -35.1, 147.3, -35.0],
            out_dir=str(out_dir),
            gee_project="   ",
        )
    except ValueError as exc:
        assert "gee_project" in str(exc)
    else:
        raise AssertionError("Expected update_gee_config to reject blank gee_project values")


def test_write_gee_config_from_cfg_rejects_blank_gee_project(tmp_path: Path):
    out_dir = tmp_path / "landsat"

    try:
        canonical_landsat.write_gee_config_from_cfg(
            gee_cfg={
                "collection": "LANDSAT/LC08/C02/T1_L2",
                "auth_mode": "browser",
            },
            start_date="2024-01-01",
            end_date="2024-01-03",
            extent=[147.2, -35.1, 147.3, -35.0],
            out_dir=str(out_dir),
            gee_project="   ",
        )
    except ValueError as exc:
        assert "gee_project" in str(exc)
    else:
        raise AssertionError("Expected write_gee_config_from_cfg to reject blank gee_project values")


def test_prepare_landsat_inputs_requires_explicit_gee_project(tmp_path: Path):
    out_dir = tmp_path / "landsat"
    out_dir.mkdir()

    try:
        canonical_landsat.prepare_landsat_inputs(
            date_range="2024-01-01 to 2024-01-03",
            extent=[147.2, -35.1, 147.3, -35.0],
            out_dir=str(out_dir),
        )
    except TypeError as exc:
        assert "gee_project" in str(exc)
    else:
        raise AssertionError("Expected prepare_landsat_inputs to require gee_project explicitly")


def test_prepare_landsat_inputs_uses_optional_template_override(monkeypatch, tmp_path: Path):
    out_dir = tmp_path / "landsat"
    template_path = tmp_path / "template.json"
    template_path.write_text(
        json.dumps(
            {
                "collection": "LANDSAT/LC08/C02/T1_L2",
                "bands": ["ST_B10", "SR_B4", "SR_B5"],
                "auth_mode": "browser",
                "scale": 30,
                "out_format": "tif",
                "filename_prefix": "Landsat",
                "download_dir": "/tmp/original",
                "coords": [0, 0, 1, 1],
            }
        ),
        encoding="utf-8",
    )

    recorded = {"init_paths": [], "run_calls": 0}

    class FakeDownloader:
        def __init__(self, config_path):
            recorded["init_paths"].append(config_path)

        def run(self):
            recorded["run_calls"] += 1

    monkeypatch.setattr(canonical_landsat, "GEEDownloader", FakeDownloader)
    monkeypatch.setattr(canonical_landsat, "_safe_mkdir", lambda path: Path(path).mkdir(parents=True, exist_ok=True))

    cfg_path = canonical_landsat.prepare_landsat_inputs(
        date_range="2024-01-01 to 2024-01-03",
        extent=[147.2, -35.1, 147.3, -35.0],
        out_dir=str(out_dir),
        gee_project="canonical-project",
        gee_config_template=str(template_path),
    )

    payload = _load_config_payload(Path(cfg_path))
    assert payload["collection"] == "LANDSAT/LC08/C02/T1_L2"
    assert payload["bands"] == ["ST_B10", "SR_B4", "SR_B5"]
    assert payload["scale"] == 30
    assert payload["out_format"] == "tif"
    assert payload["filename_prefix"] == "Landsat"
    assert payload["gee_project"] == "canonical-project"
    assert payload["download_dir"] == str(out_dir)
    assert recorded["init_paths"] == [str(Path(cfg_path))]
    assert recorded["run_calls"] == 1


def test_prepare_landsat_inputs_writes_valid_default_config(monkeypatch, tmp_path: Path):
    out_dir = tmp_path / "landsat"
    recorded = {"init_paths": [], "run_calls": 0}

    class FakeDownloader:
        def __init__(self, config_path):
            recorded["init_paths"].append(config_path)

        def run(self):
            recorded["run_calls"] += 1

    monkeypatch.setattr(canonical_landsat, "GEEDownloader", FakeDownloader)
    monkeypatch.setattr(canonical_landsat, "_safe_mkdir", lambda path: Path(path).mkdir(parents=True, exist_ok=True))

    cfg_path = canonical_landsat.prepare_landsat_inputs(
        date_range="2024-01-01 to 2024-01-03",
        extent=[147.2, -35.1, 147.3, -35.0],
        out_dir=str(out_dir),
        gee_project="canonical-project",
        gee_config_template=None,
    )

    payload = _load_config_payload(Path(cfg_path))
    assert payload["collection"] == "LANDSAT/LC08/C02/T1_L2"
    assert payload["auth_mode"] == "browser"
    assert payload["bands"] == ["ST_B10", "SR_B4", "SR_B5"]
    assert payload["scale"] == 30
    assert payload["out_format"] == "tif"
    assert payload["filename_prefix"] == "Landsat"
    assert payload["gee_project"] == "canonical-project"
    assert payload["download_dir"] == str(out_dir)
    assert recorded["init_paths"] == [str(Path(cfg_path))]
    assert recorded["run_calls"] == 1


def test_prepare_landsat_inputs_rejects_blank_gee_project(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(canonical_landsat, "_safe_mkdir", lambda path: Path(path).mkdir(parents=True, exist_ok=True))

    try:
        canonical_landsat.prepare_landsat_inputs(
            date_range="2024-01-01 to 2024-01-03",
            extent=[147.2, -35.1, 147.3, -35.0],
            out_dir=str(tmp_path / "landsat"),
            gee_project="   ",
        )
    except ValueError as exc:
        assert "gee_project" in str(exc)
    else:
        raise AssertionError("Expected prepare_landsat_inputs to reject blank gee_project values")
