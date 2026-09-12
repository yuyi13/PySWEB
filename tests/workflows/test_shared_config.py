"""
Script: test_shared_config.py
Objective: Verify one shared configuration controls calibration and simulation paths.
Author: Yi Yu (with assistance from Codex)
Created: 2026-09-12
Last updated: 2026-09-12
Inputs: Synthetic fixtures and package interfaces supplied by pytest.
Outputs: Regression assertions.
Usage: python -m pytest tests/workflows/test_shared_config.py
Dependencies: pytest, numpy, pandas, xarray, pysweb
"""

from pathlib import Path
import pytest
import pysweb
from pysweb.workflow import load_config, run_config

ROOT = Path(__file__).resolve().parents[2]


def test_shared_windows_reference_and_calibration_parameters(monkeypatch, tmp_path):
    import pysweb
    config = load_config(ROOT / "examples/workflow.toml")
    config["run"].update(root=str(tmp_path), state_timing="end")
    config["calibration"].update(enabled=True, start="2023-12-01", end="2023-12-31")
    config["model"].update(drainage_slope=0.2, use_ndvi_root_depth=True)
    calls = []
    # Resolve lazy imports before patching: importing run also loads its sibling modules.
    interfaces = [
        getattr(pysweb.swb, name) for name in ["preprocess", "calibrate", "run"]
    ]
    for name in ["preprocess", "calibrate", "run"]:
        monkeypatch.setattr(
            pysweb.swb, name, lambda _name=name, **kw: calls.append((_name, kw))
        )
    run_config(config, stages=["preprocess", "calibrate", "run"])
    preps = [kw for name, kw in calls if name == "preprocess"]
    assert len(preps) == 2
    assert preps[0]["skip_reference_ssm"] is True
    assert preps[1]["skip_reference_ssm"] is False
    calibration = next(kw for name, kw in calls if name == "calibrate")
    model = next(kw for name, kw in calls if name == "run")
    assert calibration["date_range"] == ["2023-12-01", "2023-12-31"]
    assert calibration["state_timing"] == model["state_timing"] == "end"
    assert calibration["drainage_slope"] == model["drainage_slope"] == 0.2
    assert model["calibration_file"] == calibration["output"]
    assert calibration["ndvi"].endswith("ndvi_daily_20231201_20231231.nc")
    assert (tmp_path / config["run"]["name"] / "resolved-config.json").exists()


def test_config_rejects_unknown_options_before_execution(tmp_path):
    config = tmp_path / "bad.toml"
    config.write_text(
        (ROOT / "examples/workflow.toml").read_text() + "\nunknown_typo = 1\n"
    )
    with pytest.raises(ValueError, match="Unknown ssebop settings"):
        load_config(config)
