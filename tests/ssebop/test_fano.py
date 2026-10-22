"""
Script: test_fano.py
Objective: Verify FANO reference selection, physical branches, missing inputs and grid invariance.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-22
Last updated: 2026-10-22
Inputs: Synthetic projected fields with independently specified temperature expectations.
Outputs: Numerical and workflow regression assertions.
Usage: python -m pytest tests/ssebop/test_fano.py
Dependencies: numpy, pytest, xarray, rasterio, pysweb
"""
from dataclasses import replace
import numpy as np
import pytest
import xarray as xr
from rasterio import Affine
from tests.io.test_native_grids import raster
from pysweb.ssebop.core import LocalFanoConfig
from pysweb.ssebop.fano import worldcover_fano, _grid, _mean, validate_method


def scene(ndvi=.8, cover=40, size=96):
    grid = raster(np.ones((size, size)), Affine(30, 0, 15, 0, -30, 15+30*size))
    return grid*300, grid*ndvi, grid*8, grid*cover


@pytest.mark.parametrize('cover', [30, 40, 90])
def test_green_reference_matches_analytic_cold_temperature(cover):
    fields = scene(cover=cover)
    output = worldcover_fano(*fields)
    # 300 - .125 * 8 * (.9-.8) * 10 = 299 K.
    np.testing.assert_allclose(output.tcold_K, 299.)
    np.testing.assert_allclose(output.cold_boundary_fallback_fraction, 0.)
    assert output.fano_reference_eligible.all()


@pytest.mark.parametrize('cover', [10, 20, 50, 60, 70, 95, 100])
def test_nonag_uses_fine_temperature_without_claiming_reference_support(cover):
    output = worldcover_fano(*scene(ndvi=.2, cover=cover))
    np.testing.assert_allclose(output.tcold_K, 293.)
    assert not output.fano_reference_eligible.any()
    np.testing.assert_allclose(output.cold_boundary_fallback_fraction, 0.)


def test_dry_crop_without_green_references_reports_scene_fallback():
    output = worldcover_fano(*scene(ndvi=.2))
    np.testing.assert_allclose(output.tcold_K, 293.)
    np.testing.assert_allclose(output.cold_boundary_fallback_fraction, 1.)


def test_green_forest_cannot_replace_green_crop_reference():
    lst, ndvi, dt, lc = scene(size=160)
    lc.values[:, 128:] = 10
    ndvi.values[:, 128:] = .9
    lst.values[:, 128:] = 270.
    # Dry field surrounded by the 299 K crop reference.
    ndvi.values[64:80, 64:80] = .2
    lst.values[64:80, 64:80] = 315.
    cfg = LocalFanoConfig(coarse_scale_m=480, smooth_radius_pixels=0)
    first = worldcover_fano(lst, ndvi, dt, lc, config=cfg)
    assert first.tcold_K.values[70, 70] == pytest.approx(299.)
    assert first.cold_boundary_fallback_fraction.values[70, 70] == 0
    assert not first.fano_reference_eligible.values[:, 128:].any()
    lst.values[:, 128:] = 320.
    second = worldcover_fano(lst, ndvi, dt, lc, config=cfg)
    assert second.tcold_K.values[70, 70] == pytest.approx(299.)


def test_water_excluded_and_negative_ndvi_anomalous_branch():
    lst, ndvi, dt, lc = scene(ndvi=-.2, cover=60)
    lc.values[:, :32] = 80
    output = worldcover_fano(lst, ndvi, dt, lc)
    assert output.tcold_K.values[:, :32].shape == (96, 32)
    assert np.isnan(output.tcold_K.values[:, :32]).all()
    np.testing.assert_allclose(output.tcold_K.values[:, 32:], 250.)
    assert not output.fano_reference_eligible.any()


@pytest.mark.parametrize('missing', [np.nan, 0, 42])
def test_unknown_landcover_is_an_error(missing):
    lst, ndvi, dt, lc = scene()
    lc.values[20, 20] = missing
    with pytest.raises(ValueError, match='recognised classes'):
        worldcover_fano(lst, ndvi, dt, lc)


def test_missing_observations_remain_missing():
    lst, ndvi, dt, lc = scene()
    lst.values[10:30, 10:30] = np.nan
    result = worldcover_fano(lst, ndvi, dt, lc)
    assert np.isnan(result.tcold_K.values[10:30, 10:30]).all()
    dt.values[:] = 0
    assert worldcover_fano(lst, ndvi, dt, lc).tcold_K.isnull().all()


def test_fixed_grid_survives_shifted_crop():
    lst, ndvi, dt, lc = scene(size=160)
    # Spatial gradient detects accidental anchoring at each array's top-left.
    lst.values += np.arange(160)[None, :]*.05
    fields = (lst, ndvi, dt, lc)
    full = worldcover_fano(*fields)
    cropped = worldcover_fano(*(v.isel(x=slice(7, 151), y=slice(5, 150)) for v in fields))
    target = dict(x=slice(45, 100), y=slice(45, 100))
    expected = full.tcold_K.isel(**target)
    np.testing.assert_allclose(cropped.tcold_K.sel(x=expected.x, y=expected.y), expected)
    grid = _grid(cropped.tcold_K, 4800)
    assert (grid.rio.transform().c-15) % 4800 == 0
    assert (grid.rio.transform().f-15) % 4800 == 0


def test_radius_five_is_eleven_cells_and_ignores_missing():
    data = xr.DataArray(np.zeros((15, 15)), dims=('y', 'x'))
    data.values[7, 7] = 121
    result = _mean(data, 5)
    assert result.values[7, 7] == pytest.approx(1.)
    assert result.values[7, 2] > 0
    assert result.values[7, 1] == 0


def test_default_rejects_missing_cover_and_legacy_is_explicit():
    with pytest.raises(ValueError, match='requires ESA WorldCover'):
        validate_method('worldcover_fano', None)
    validate_method('legacy', None)
    with pytest.raises(ValueError, match='tcold_method'):
        validate_method('typo', None)


@pytest.mark.parametrize('changes', [dict(fine_scale_m=0), dict(coarse_scale_m=-1),
    dict(smooth_radius_pixels=2.5), dict(anchor_ndvi_threshold=2), dict(dt_coeff=np.nan)])
def test_invalid_fano_parameters(changes):
    with pytest.raises(ValueError):
        worldcover_fano(*scene(), config=replace(LocalFanoConfig(), **changes))


def test_worldcover_download_covers_exported_cells(tmp_path):
    from rasterio.warp import transform_bounds
    from pysweb.io.acquisition import worldcover_extent
    grid = scene()[0]
    path = tmp_path/'scene.tif'
    grid.rio.to_raster(path)
    bounds = transform_bounds(grid.rio.crs, 'EPSG:4326', *grid.rio.bounds(), densify_pts=65)
    # Deliberately request less than the exported extent.
    requested = [bounds[0]+.001, bounds[1]+.001, bounds[2]-.001, bounds[3]-.001]
    actual = worldcover_extent([path], requested)
    assert actual[0] < bounds[0] and actual[1] < bounds[1]
    assert actual[2] > bounds[2] and actual[3] > bounds[3]


def test_categorical_landcover_alignment_and_windowed_loading(tmp_path):
    from pysweb.ssebop.landcover import load_worldcover_landcover
    lst, ndvi, dt, _ = scene(size=96)
    cover = raster(np.tile(np.repeat([10., 40.], 144), (288, 1)),
                   Affine(10, 0, 15, 0, -10, 2895))
    path = tmp_path/'worldcover.tif'
    cover.rio.to_raster(path)
    chip = lst.isel(x=slice(20, 80), y=slice(20, 80))
    loaded = load_worldcover_landcover(path, template=chip)
    assert loaded.size < cover.size
    result = worldcover_fano(chip, ndvi.sel(x=chip.x, y=chip.y),
                            dt.sel(x=chip.x, y=chip.y), loaded)
    assert set(np.unique(result.worldcover_class)) == {10., 40.}
    assert not result.fano_reference_eligible.where(result.worldcover_class == 10, 0).any()
