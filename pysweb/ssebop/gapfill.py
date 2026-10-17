#!/usr/bin/env python3
"""
Script: gapfill.py
Objective: Fill explicit scan-gap masks with adaptive local radiometric matching and provenance.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Coregistered floating-point primary/donor bands and an explicit eligible-gap mask.
Outputs: Gap estimates, donor indices, support counts, windows and estimator flags.
Usage: from pysweb.ssebop.gapfill import local_linear_gapfill
Dependencies: numpy.

Based on the public-domain USGS Landsat 7 SLC-off Phase Two Methodology,
version 2.0 (2004-10-07). This implementation requires the minimum common
support even at the maximum window, uses original observed target pixels
for every donor fit, and leaves values in their supplied physical units.
It does not apply the original 8-bit DN clipping. These conservative
adaptations require application-specific validation, especially for LST.
"""
import numpy as np


def _integral(values):
    return np.pad(np.cumsum(np.cumsum(values, axis=0), axis=1), ((1, 0), (1, 0)))


def _sum_at(integral, rows, cols, radius, shape):
    top = np.maximum(rows - radius, 0)
    bottom = np.minimum(rows + radius + 1, shape[0])
    left = np.maximum(cols - radius, 0)
    right = np.minimum(cols + radius + 1, shape[1])
    return integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]


def local_linear_gapfill(primary, donors, gap_mask, *, min_common=144,
                         max_window=31, gain_limits=(1 / 3, 3), value_limits=None):
    """Fill only eligible missing pixels using ordered, already-masked donors.

    Inputs are a single 2-D band in consistent units/grid. Donors may be an
    iterable to bound memory. Cloud/shadow/snow/saturation must already be NaN.
    The caller owns donor dates, sensor compatibility and eligibility masks.
    Original finite values are preserved exactly. Residual missing pixels
    remain NaN. Donor index 0 denotes residual missing, 1 original, 2+ donor.
    Method 1 is OLS, 2 standard-deviation matching, and 3 mean-offset matching.
    No spatial registration, spectral harmonisation or time interpolation is
    implied. Per-band provenance should be retained when assembling products.
    """
    original = np.asarray(primary, dtype=np.float64)
    gaps = np.asarray(gap_mask, dtype=bool)
    if original.ndim != 2 or gaps.shape != original.shape:
        raise ValueError('Primary and gap mask must be identically shaped 2-D arrays')
    if not isinstance(max_window, int) or max_window < 3 or max_window % 2 != 1:
        raise ValueError('max_window must be an odd integer >= 3')
    if not isinstance(min_common, int) or not 2 <= min_common <= max_window ** 2:
        raise ValueError('min_common must be an integer from 2 to max_window squared')
    low, high = gain_limits
    if not 0 < low <= high or not np.isfinite([low, high]).all():
        raise ValueError('Gain limits must be finite, positive and ordered')
    if value_limits is not None and not value_limits[0] < value_limits[1]:
        raise ValueError('Value limits must be ordered')
    result = original.copy()
    observed = np.isfinite(original)
    provenance = observed.astype(np.uint16)
    window = np.zeros(original.shape, dtype=np.uint16)
    support = np.zeros(original.shape, dtype=np.uint32)
    method = np.zeros(original.shape, dtype=np.uint8)
    eligible = gaps & ~observed
    for donor_number, donor in enumerate(donors, start=2):
        if donor_number > np.iinfo(np.uint16).max:
            raise ValueError('Too many donors for uint16 provenance')
        donor = np.asarray(donor, dtype=np.float64)
        if donor.shape != original.shape:
            raise ValueError('All donors must match the primary shape')
        rows, cols = np.nonzero(eligible & ~np.isfinite(result) & np.isfinite(donor))
        if rows.size == 0:
            continue
        common = observed & np.isfinite(donor)
        if common.sum() < min_common:
            continue
        # Centre temperature/reflectance before squared integrals to reduce
        # cancellation, retaining exact affine matching in physical units.
        x_origin = np.mean(donor[common])
        y_origin = np.mean(original[common])
        x = np.where(common, donor - x_origin, 0)
        y = np.where(common, original - y_origin, 0)
        totals = [_integral(a) for a in (common.astype(float), x, y, x*x, x*y, y*y)]
        pending = np.ones(rows.size, dtype=bool)
        for radius in range(max_window // 2 + 1):
            indices = np.flatnonzero(pending)
            if indices.size == 0:
                break
            rr, cc = rows[indices], cols[indices]
            count = _sum_at(totals[0], rr, cc, radius, original.shape)
            sufficient = count >= min_common
            indices, rr, cc, count = indices[sufficient], rr[sufficient], cc[sufficient], count[sufficient]
            if indices.size == 0:
                continue
            pending[indices] = False
            sx, sy, sxx, sxy, syy = [_sum_at(t, rr, cc, radius, original.shape) for t in totals[1:]]
            mx, my = sx / count, sy / count
            vx, vy = np.maximum(sxx / count - mx*mx, 0), np.maximum(syy / count - my*my, 0)
            cov = sxy / count - mx*my
            gain = np.divide(cov, vx, out=np.full_like(vx, np.nan), where=vx > 1e-14)
            kind = np.ones(gain.shape, dtype=np.uint8)
            bad = ~np.isfinite(gain) | (gain < low) | (gain > high)
            ratio = np.sqrt(np.divide(vy, vx, out=np.full_like(vx, np.nan), where=vx > 1e-14))
            gain[bad] = ratio[bad]
            kind[bad] = 2
            bad = ~np.isfinite(gain) | (gain < low) | (gain > high)
            gain[bad] = 1
            kind[bad] = 3
            prediction = gain * (donor[rr, cc] - x_origin - mx) + y_origin + my
            valid = np.isfinite(prediction)
            if value_limits is not None:
                valid &= (prediction >= value_limits[0]) & (prediction <= value_limits[1])
            rr, cc = rr[valid], cc[valid]
            result[rr, cc] = prediction[valid]
            provenance[rr, cc] = donor_number
            window[rr, cc] = radius * 2 + 1
            support[rr, cc] = np.rint(count[valid]).astype(np.uint32)
            method[rr, cc] = kind[valid]
    return dict(values=result, provenance=provenance, window=window, common_count=support, method=method)
