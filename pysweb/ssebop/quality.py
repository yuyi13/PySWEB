#!/usr/bin/env python3
"""
Script: quality.py
Objective: Decode Landsat Collection 2 pixel and saturation quality consistently for local workflows.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-16
Last updated: 2026-10-16
Inputs: QA_PIXEL and QA_RADSAT arrays from Landsat 8/9 Collection 2 Level 2.
Outputs: Boolean valid-pixel mask without changing or filling source observations.
Usage: from pysweb.ssebop.quality import landsat_c2_qa_mask
Dependencies: numpy
"""
import numpy as np


def landsat_c2_qa_mask(qa_pixel, qa_radsat, *, exclude_water=True):
    """Keep finite integer QA with no fill, cloud, cirrus, shadow, snow or saturation.

    QA_PIXEL bits 0–5 reject fill, dilated cloud, high-confidence cirrus,
    cloud, shadow and snow. Bit 7 optionally rejects water. No additional
    buffer or confidence threshold is applied. QA_RADSAT must equal zero.
    Raw zero is a valid QA value, especially for QA_RADSAT, not nodata.
    Optical/thermal availability and NDVI bounds are checked by the caller.
    """
    qa = np.asarray(qa_pixel)
    sat = np.asarray(qa_radsat)
    if qa.shape != sat.shape:
        raise ValueError("QA_PIXEL and QA_RADSAT must have identical shapes.")
    valid = (np.isfinite(qa) & np.isfinite(sat) & (qa >= 0) & (qa <= 65535)
             & (sat >= 0) & (sat <= 65535))
    # Reject noninteger QA rather than turning a median of flags into a new flag.
    valid &= (qa == np.floor(qa)) & (sat == np.floor(sat))
    values = np.where(valid, qa, 0).astype(np.uint16)
    rejected_bits = 63 | (128 if exclude_water else 0)
    return valid & ((values & rejected_bits) == 0) & (sat == 0)
