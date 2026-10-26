"""
Script: formulation.py
Objective: Define the versioned two-parameter SWB contract and reject incompatible calibrations.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-18
Last updated: 2026-10-23
Inputs: Fixed reference artifacts, model settings and calibration CSV files.
Outputs: Validated fixed settings, artifact identities and fitted coefficients.
Usage: import pysweb.swb.formulation
Dependencies: numpy, pandas; pysweb
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FORMULATION_VERSION = "swb-daily-depth-extraction-2p-v3"
TIME_INTEGRATION = "single-bounded-update"
EXTRACTION_PROFILE = "gssm-wp-depth-blend-0-150-300mm-v1"
# These observation-derived scalars are independent of the integration scheme.
REFERENCE_COMPATIBLE_FORMULATIONS = {
    "swb-conservative-2p-v1", "swb-daily-conservative-2p-v2", FORMULATION_VERSION}
REFERENCE_VERSION = "gssm-native-p01-p99-v1"
AVAILABLE_REFERENCE_VERSION = "gssm-native-p01-p99-support-v2"
PARAMETER_NAMES = ("diff_factor", "root_beta")
DEFAULT_ROOT_BETA = 0.96
TAU_DAYS = 3.0
LEGACY_KEYS = {"sm_max_factor", "sm_min_factor", "sm_max_bounds", "sm_min_bounds",
               "sm_min_bound", "sm_max_bound", "sm_min_relax_tau_days",
               "sm_min_relax_trigger_factor", "sm_min_factor_max_depth_mm",
               "rhs_diffusion_enabled", "rhs_diffusion_limiter_enabled",
               "rhs_diffusion_abs_cap_mm_day", "max_substep_days", "hydraulic_cfl"}


def reject_legacy(settings):
    keys = LEGACY_KEYS.intersection(settings)
    if keys:
        raise ValueError(f"Incompatible legacy SWB settings {sorted(keys)}; remove them, "
                         "freeze a GSSM reference artifact and recalibrate diff_factor/root_beta.")


def canonical_id(record):
    payload = {k: v for k, v in record.items() if k != "artifact_id"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_reference(path):
    """Load an immutable reference; source support and derivation are part of its hash."""
    if path is None:
        return dict(artifact_id="physical-unadjusted", dry_reference_factor=1., wet_reference_factor=1.)
    record = json.loads(Path(path).read_text())
    if (record.get("schema_version") not in {REFERENCE_VERSION, AVAILABLE_REFERENCE_VERSION}
            or record.get("artifact_id") != canonical_id(record)):
        raise ValueError("Incompatible or modified GSSM reference artifact")
    if record.get("formulation_version") not in REFERENCE_COMPATIBLE_FORMULATIONS:
        raise ValueError("Reference formulation version differs; derive a new reference")
    dry, wet = record["dry_reference_factor"], record["wet_reference_factor"]
    if not (np.isfinite(dry) and 0 <= dry <= 1 and np.isfinite(wet) and wet >= 1):
        raise ValueError("Invalid fixed GSSM reference scalars")
    for key in ("climatology", "spatial_support", "hydraulic_support", "sources"):
        if not record.get(key):
            raise ValueError(f"Reference artifact missing {key}")
    if record["schema_version"] == AVAILABLE_REFERENCE_VERSION:
        selection = record.get("support_selection", {})
        if (selection.get("policy_version") != "available-native-cells-v1"
                or not selection.get("candidate_cells")
                or not selection.get("retained_cell_count")):
            raise ValueError("Reference artifact missing available-cell support selection")
    return record


def settings_identity(reference_file=None, *, max_root_depth=1000., drainage_slope=.5,
                      drainage_upper_limit=25., drainage_lower_limit=0.):
    reference = load_reference(reference_file)
    return dict(formulation_version=FORMULATION_VERSION, tau_days=TAU_DAYS,
                extraction_threshold_profile=EXTRACTION_PROFILE,
                physical_upper_bound="hydraulic_porosity",
                time_integration=TIME_INTEGRATION, time_step_days=1.,
                reference_artifact_id=reference["artifact_id"],
                max_root_depth_mm=float(max_root_depth), drainage_slope=float(drainage_slope),
                drainage_upper_limit=float(drainage_upper_limit), drainage_lower_limit=float(drainage_lower_limit))


def reference_properties(reference_file=None):
    record = load_reference(reference_file)
    return {k: float(record[k]) for k in ("dry_reference_factor", "wet_reference_factor")}


def extraction_thresholds(wilting_point, bottoms_mm, dry_factor):
    """Transfer the surface dry ratio by depth, without changing hydraulic WP.

    Full adjustment applies through 150 mm, half adjustment from 150 to 300 mm,
    and none below 300 mm. Layers crossing these boundaries use thickness-
    weighted adjustment, preserving the same depth rule on other layer grids.
    """
    wp = np.asarray(wilting_point, dtype=float)
    bottoms = np.asarray(bottoms_mm, dtype=float)
    if (wp.ndim != 1 or not len(wp) or wp.shape != bottoms.shape
            or not np.isfinite(wp).all() or np.any(wp < 0)
            or not np.isfinite(bottoms).all() or np.any(np.diff(np.r_[0., bottoms]) <= 0)
            or not np.isfinite(dry_factor) or not 0 <= dry_factor <= 1):
        raise ValueError("Invalid depth-dependent extraction threshold inputs")
    tops = np.r_[0., bottoms[:-1]]
    full = np.maximum(0., np.minimum(bottoms, 150.) - tops)
    half = np.maximum(0., np.minimum(bottoms, 300.) - np.maximum(tops, 150.))
    weight = (full + .5 * half) / (bottoms - tops)
    return wp * (1. - (1. - dry_factor) * weight)


def load_calibration(path, reference_file=None, **fixed_settings):
    """Never reinterpret a four-parameter or unversioned fit as the revised model."""
    # The daily trajectory can distinguish neighbouring float64 coefficients.
    # Preserve exactly the values written by calibration's 17-digit CSV export.
    table = pd.read_csv(path, float_precision="round_trip")
    reject_legacy(table.columns)
    if len(table) != 1:
        raise ValueError("Calibration CSV must contain exactly one parameter row")
    row = table.iloc[0]
    for key, expected in settings_identity(reference_file, **fixed_settings).items():
        if key not in row or (str(row[key]) != expected if isinstance(expected, str)
                              else not np.isclose(float(row[key]), expected, rtol=0, atol=1e-15)):
            raise ValueError(f"Incompatible calibration {key}; recalibration is required")
    values = {key: float(row[key]) for key in PARAMETER_NAMES}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Non-finite calibrated parameter")
    if not (10 <= values["diff_factor"] <= 10000 and .9 <= values["root_beta"] <= .995):
        raise ValueError("Calibrated parameters outside the supported bounds")
    return values


BASE_BUDGET_NAMES = ("infiltration", "actual_et", "actual_evaporation", "actual_transpiration",
                    "requested_evaporation", "requested_transpiration", "unmet_evaporation",
                    "unmet_transpiration", "drainage", "overflow", "storage_adjustment",
                    "storage_change", "balance_residual", "substeps")


def budget_names(layers):
    return (BASE_BUDGET_NAMES
            + tuple(f"interface_flux_{i+1}" for i in range(layers-1))
            + tuple(f"transpiration_layer_{i+1}" for i in range(layers))
            + tuple(f"layer_residual_{i+1}" for i in range(layers))
            + tuple(f"below_wp_days_layer_{i+1}" for i in range(layers))
            + tuple(f"below_reference_days_layer_{i+1}" for i in range(layers))
            + tuple(f"above_reference_days_layer_{i+1}" for i in range(min(2, layers))))
