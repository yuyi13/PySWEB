#!/usr/bin/env bash
# Script: sweb_domain_runner.sh
# Objective: Run configured SWB preprocessing, calibration and simulation through the shared workflow CLI.
# Author: Yi Yu (with assistance from Codex)
# Created: 2026-02-17
# Last updated: 2026-09-12
# Inputs: Workflow TOML, optional --stages/--dry-run flags and PYTHON interpreter override.
# Outputs: Configured model products and provenance manifests, or a JSON dry-run plan.
# Usage: bash workflows/sweb_domain_runner.sh --config examples/workflow.toml --dry-run
# Requirements: bash, dirname and a Python interpreter with pysweb dependencies
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python}" -m pysweb.workflow --stages preprocess,calibrate,run "$@"
