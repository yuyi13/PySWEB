"""
Script: test_references.py
Objective: Verify temporal-before-spatial GSSM quantiles, support integrity and artifact identity.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-18
Last updated: 2026-10-23
Inputs: Synthetic geographic cells with anticorrelated temporal moisture extremes.
Outputs: Exact reference order, coverage and tamper-rejection assertions.
Usage: python -m pytest tests/swb/test_references.py
Dependencies: numpy, pandas, xarray, pytest, pysweb
"""
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import rioxarray
from pysweb.swb.references import derive_reference, frozen_mask
from pysweb.swb.formulation import load_reference


def inputs():
    coords = dict(y=[1.5, .5], x=[.5, 1.5])
    mask = xr.DataArray(np.ones((2, 2), dtype=bool), dims=("y", "x"), coords=coords).rio.write_crs(4326)
    dates = pd.date_range("2011-01-01", "2020-12-31")
    data = np.zeros((len(dates), 2, 2))
    data[::2, :, 0] = .4
    data[1::2, :, 1] = .4
    gssm = xr.DataArray(data, dims=("time", "y", "x"), coords={**coords, "time": dates}, attrs=dict(units="m3 m-3")).rio.write_crs(4326)
    hyd = {key: mask.astype(float)*value for key, value in
           dict(wilting_point=.1, porosity=.5, field_capacity=.3, b_coefficient=4., conductivity_sat=20.).items()}
    return gssm, mask, hyd


def test_quantiles_precede_spatial_averaging_and_hash_binds_support(tmp_path):
    gssm, mask, hyd = inputs()
    result = derive_reference(gssm, mask, hyd, site="fixture", sources=["synthetic"], hydraulic_provenance="native fixture")
    assert result["area_mean_gssm_p01"] == 0
    assert result["area_mean_gssm_p99"] == pytest.approx(.4)
    assert float(gssm.mean(("y", "x")).quantile(.01)) == pytest.approx(.2)
    assert result["hydraulic_support"]["means"]["wilting_point"] == pytest.approx(.1)
    np.testing.assert_array_equal(frozen_mask(mask, result), mask)
    path = tmp_path/"reference.json"
    path.write_text(json.dumps(result))
    assert load_reference(path)["artifact_id"] == result["artifact_id"]
    result["dry_reference_factor"] = .1
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="modified"):
        load_reference(path)


def test_missing_soil_or_incomplete_reference_cannot_shrink_frozen_support():
    gssm, mask, hyd = inputs()
    hyd["wilting_point"].values[0, 0] = np.nan
    with pytest.raises(ValueError, match="Missing native hydraulic"):
        derive_reference(gssm, mask, hyd, site="fixture", sources=["synthetic"], hydraulic_provenance="fixture")

    gssm, mask, hyd = inputs()
    gssm.values[:40, 0, 0] = np.nan
    with pytest.raises(ValueError, match="99%"):
        derive_reference(gssm, mask, hyd, site="fixture", sources=["synthetic"], hydraulic_provenance="fixture")


@pytest.mark.parametrize("previous", ["swb-conservative-2p-v1", "swb-daily-conservative-2p-v2"])
def test_daily_model_reuses_unchanged_observation_reference(tmp_path, previous):
    from pysweb.swb.formulation import canonical_id
    gssm, mask, hyd = inputs()
    record = derive_reference(gssm, mask, hyd, site="fixture", sources=["synthetic"],
                              hydraulic_provenance="fixture")
    record["formulation_version"] = previous
    record["artifact_id"] = canonical_id(record)
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(record))
    assert load_reference(path) == record
    record["formulation_version"] = "unknown-model"
    record["artifact_id"] = canonical_id(record)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="formulation version"):
        load_reference(path)


def test_available_cells_keep_identical_gssm_and_hydraulic_footprints(tmp_path):
    gssm, mask, hyd = inputs()
    gssm.values[:, 0, 0] = np.nan
    # This hydraulic pixel must not contribute after its GSSM cell is excluded.
    hyd["wilting_point"].values[0, 0] = .45
    result = derive_reference(gssm, mask, hyd, site="fixture", sources=["synthetic"],
        hydraulic_provenance="fixture", coverage_policy=dict(
            minimum_valid_fraction=.99, minimum_area_fraction=.5))
    selection = result["support_selection"]
    assert selection["candidate_cell_count"] == 4
    assert selection["retained_cell_count"] == 3
    assert selection["retained_area_fraction"] == pytest.approx(.75, abs=1e-4)
    assert selection["candidate_cells"][0]["valid_days"] == 0
    assert not selection["candidate_cells"][0]["retained"]
    assert result["hydraulic_support"]["means"]["wilting_point"] == pytest.approx(.1)
    assert result["hydraulic_support"]["area_m2"] == pytest.approx(selection["retained_area_m2"])
    expected = mask.copy()
    expected.values[0, 0] = False
    np.testing.assert_array_equal(frozen_mask(mask, result), expected)
    path = tmp_path/"reference.json"
    path.write_text(json.dumps(result))
    assert load_reference(path) == result
    assert result["schema_version"] == "gssm-native-p01-p99-support-v2"


def test_sparse_and_empty_support_still_fail():
    gssm, mask, hyd = inputs()
    gssm.values[:, 0, :] = np.nan
    gssm.values[:, 1, 0] = np.nan
    options = dict(site="fixture", sources=["synthetic"], hydraulic_provenance="fixture",
        coverage_policy=dict(minimum_valid_fraction=.99, minimum_area_fraction=.5))
    with pytest.raises(ValueError, match="Retained GSSM area fraction"):
        derive_reference(gssm, mask, hyd, **options)
    gssm.values[:] = np.nan
    with pytest.raises(ValueError, match="No GSSM cells"):
        derive_reference(gssm, mask, hyd, **options)


def test_retained_cells_still_need_99_percent_of_ten_year_calendar():
    gssm, mask, hyd = inputs()
    gssm.values[:36, 0, 0] = np.nan
    gssm.values[:37, 0, 1] = np.nan
    result = derive_reference(gssm, mask, hyd, site="fixture", sources=["synthetic"],
        hydraulic_provenance="fixture", coverage_policy=dict(
            minimum_valid_fraction=.99, minimum_area_fraction=.5))
    candidates = result["support_selection"]["candidate_cells"]
    assert result["climatology"]["expected_days"] == 3653
    assert candidates[0]["retained"] and candidates[0]["valid_days"] == 3617
    assert not candidates[1]["retained"] and candidates[1]["valid_days"] == 3616


def test_all_valid_support_preserves_numerics_but_binds_new_policy():
    gssm, mask, hyd = inputs()
    options = dict(site="fixture", sources=["synthetic"], hydraulic_provenance="fixture")
    strict = derive_reference(gssm, mask, hyd, **options)
    available = derive_reference(gssm, mask, hyd, **options, coverage_policy=dict(
        minimum_valid_fraction=.99, minimum_area_fraction=.5))
    assert strict["schema_version"] == "gssm-native-p01-p99-v1"
    assert "support_selection" not in strict
    for key in ("dry_reference_factor", "wet_reference_factor", "area_mean_gssm_p01",
                "area_mean_gssm_p99", "spatial_support", "hydraulic_support"):
        assert strict[key] == available[key]
    assert strict["artifact_id"] != available["artifact_id"]


@pytest.mark.parametrize("policy", [
    {}, {"minimum_valid_fraction": .99},
    {"minimum_valid_fraction": 0, "minimum_area_fraction": .5},
    {"minimum_valid_fraction": .99, "minimum_area_fraction": 1.1},
    {"minimum_valid_fraction": True, "minimum_area_fraction": .5},
    {"minimum_valid_fraction": float("nan"), "minimum_area_fraction": .5},
])
def test_invalid_coverage_policy_rejected(policy):
    gssm, mask, hyd = inputs()
    with pytest.raises(ValueError):
        derive_reference(gssm, mask, hyd, site="fixture", sources=["synthetic"],
            hydraulic_provenance="fixture", coverage_policy=policy)
