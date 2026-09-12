#!/usr/bin/env bash
# Script: ssebop_runner_landsat.sh
# Objective: Run configured PySWEB stages using the shared package orchestrator.
# Author: Yi Yu (with assistance from Codex)
# Created: 2026-02-17
# Last updated: 2026-09-12
# Inputs: --config study.toml and optional --stages / --dry-run arguments.
# Outputs: Configured model products and provenance records.
# Usage: bash workflows/ssebop_runner_landsat.sh --config examples/workflow.toml --dry-run
# Requirements: bash, installed pysweb or a source checkout with dependencies
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python}" -m pysweb.workflow --stages prepare,ssebop "$@"
