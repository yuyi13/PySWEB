#!/usr/bin/env python3
"""
Script: quality.py
Objective: Apply explicit, bounded precipitation packing-noise correction while preserving missingness provenance.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Precipitation in millimetres and a caller-justified small-negative tolerance.
Outputs: Corrected floating values and same-shaped categorical quality flags.
Usage: qc_precipitation_mm(values, small_negative_tolerance_mm=0.001)
Dependencies: numpy.
"""
import numpy as np


def qc_precipitation_mm(precipitation, *, small_negative_tolerance_mm):
    """Bound correction to explicitly declared small negative accumulations.

    Flags:0=unchanged finite nonnegative,1=small negative reset to zero,
    2=negative beyond tolerance masked,3=nonfinite input masked. Missing inputs
    never become zero and nonnegative values remain unchanged. The tolerance
    is mandatory: no universal packing-error magnitude is assumed. This does
    not correct positive packing noise or validate large rainfall extremes.
    """
    tolerance=float(small_negative_tolerance_mm)
    if not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('small_negative_tolerance_mm must be finite and nonnegative')
    source=np.asarray(precipitation,dtype=float)
    finite=np.isfinite(source)
    small=finite&(source<0)&(source>=-tolerance)
    rejected=finite&(source<-tolerance)
    flags=np.zeros(source.shape,dtype=np.uint8)
    flags[small]=1
    flags[rejected]=2
    flags[~finite]=3
    corrected=np.where(small,0.,source)
    corrected=np.where(rejected|~finite,np.nan,corrected)
    return corrected,flags
