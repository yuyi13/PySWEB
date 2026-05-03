#!/usr/bin/env python3
"""
Script: thomas_solve_tridiagonal_matrix.py
Objective: Provide a deprecated compatibility wrapper for the package-owned Thomas tridiagonal solver.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-05-03
Inputs: Lower/main/upper diagonal arrays and right-hand-side vector for a tridiagonal system.
Outputs: Solution vector from `pysweb.swb.solver.thomas_solve_tridiagonal_matrix`.
Usage: import thomas_solve_tridiagonal_matrix from pysweb.swb.solver instead.
Dependencies: pysweb.swb.solver
"""
from __future__ import annotations

from pysweb.swb.solver import thomas_solve_tridiagonal_matrix
