#!/usr/bin/env bash
# Script: run_site.sh
# Objective: Run the same configured site stages locally without scheduler dependencies.
# Author: Yi Yu (with assistance from Codex)
# Created: 2026-10-23
# Last updated: 2026-10-23
# Inputs: SWEB_PYTHON (optional), site plan and stage arguments.
# Outputs: Site products and stage receipts.
# Usage: bash workflows/local/run_site.sh --plan site.json --stages all
# Requirements: Bash, POSIX host and installed PySWEB scientific dependencies.
set -euo pipefail
SWEB_SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
exec "${SWEB_PYTHON:-python3}" "$SWEB_SCRIPT_ROOT/run_site.py" "$@"
