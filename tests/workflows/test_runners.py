"""
Script: test_runners.py
Objective: Verify shell runners resolve the shared configuration when invoked outside the checkout.
Author: Yi Yu (with assistance from Codex)
Created: 2026-04-17
Last updated: 2026-09-12
Inputs: Shell runner paths, the example TOML configuration and temporary working directories.
Outputs: Assertions on dry-run plans and the absence of generated workflow outputs.
Usage: python -m pytest tests/workflows/test_runners.py
Dependencies: pytest; pysweb; bash
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "script,stages",
    [
        ("workflows/ssebop_runner_landsat.sh", ["prepare", "ssebop"]),
        ("workflows/sweb_domain_runner.sh", ["preprocess", "calibrate", "run"]),
        ("spec/sweb_mlcons_runner.sh", ["preprocess", "calibrate", "run"]),
    ],
)
def test_runner_works_outside_checkout(script, stages, tmp_path):
    env = {**os.environ, "PYTHON": sys.executable}
    result = subprocess.run(
        [
            "bash",
            str(ROOT / script),
            "--config",
            str(ROOT / "examples/workflow.toml"),
            "--dry-run",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["stages"] == stages
    assert plan["config"]["run"]["root"] == str(ROOT / "examples/outputs")
    assert not list(tmp_path.iterdir())
