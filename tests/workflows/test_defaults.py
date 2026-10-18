"""
Script: test_defaults.py
Objective: Verify sensor opt-in, 2013 initialisation and exact metric point chips.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-18
Last updated: 2026-10-18
Inputs: Temporary TOML configurations and synthetic native Landsat rasters.
Outputs: Offline workflow policy, pixel identity and footprint assertions.
Usage: python -m pytest tests/workflows/test_defaults.py
Dependencies: numpy, pytest, pyproj, rasterio, xarray, pysweb
"""
import json
import numpy as np
import pytest
import xarray as xr
from pyproj import Transformer
from rasterio import Affine
from pysweb.io.grids import centered_template
from pysweb.workflow import load_config, run_config
from pysweb.workflow_native import _cached_scene_files
from tests.io.test_native_grids import raster


def config_file(tmp_path, *, start='2013-05-17', end='2020-12-31',
                area='center = [146.2908, -34.9878]', sensors='', calibration='enabled = true'):
    path = tmp_path / 'study.toml'
    path.write_text(f'''[run]
name = "study"
root = "outputs"
start = "{start}"
end = "{end}"
{area}
{sensors}
[calibration]
{calibration}
''')
    return path


def test_defaults_use_single_modern_sensor_path_and_2013_initialisation(tmp_path, monkeypatch):
    config = load_config(config_file(tmp_path))
    assert config['run']['sensors'] == ['LC08', 'LC09']
    assert config['run']['chip_size_m'] == 5100
    plan = run_config(config, dry_run=True)
    assert plan['calibration_scoring_window'] == ['2014-01-01', '2020-12-31']
    assert plan['initialization_window'] == ['2013-05-17', '2013-12-31']
    import pysweb
    calls = []
    monkeypatch.setattr(pysweb.ssebop, 'prepare_inputs', lambda **kwargs: calls.append(kwargs))
    config['calibration']['enabled'] = False
    run_config(config, stages=['prepare'])
    assert len(calls) == 1 and calls[0]['sensors'] == ['LC08', 'LC09']


@pytest.mark.parametrize('enabled', ['true', 'false'])
def test_actual_2013_warmup_is_shared_by_calibration_and_normal_runs(tmp_path, enabled):
    config = load_config(config_file(tmp_path, calibration=f'enabled = {enabled}'))
    plan = run_config(config, dry_run=True)
    assert plan['initialization_window'] == ['2013-05-17', '2013-12-31']
    assert plan['analysis_window'] == ['2014-01-01', '2020-12-31']
    assert plan['initialization']['initial_state'] == 'physical_field_capacity'
    assert plan['initialization']['mode'] == 'actual_forcing_prefix'
    assert plan['initialization']['artificial_warmup_days'] == 0
    assert not config.get('spinup')
    historical = load_config(config_file(tmp_path, calibration=f'enabled = {enabled}',
        sensors='sensors = ["LT05", "LE07", "LC08", "LC09"]'))
    assert run_config(historical, dry_run=True)['initialization_window'] is None


def test_2013_only_normal_run_is_explicitly_all_warmup(tmp_path):
    config = load_config(config_file(tmp_path, end='2013-12-31', calibration='enabled = false'))
    plan = run_config(config, dry_run=True)
    assert plan['analysis_window'] is None
    assert plan['initialization_window'] == ['2013-05-17', '2013-12-31']


def test_historical_sensors_require_explicit_selection(tmp_path):
    with pytest.raises(ValueError, match='explicitly select LT05/LE07'):
        load_config(config_file(tmp_path, start='2011-01-01'))
    config = load_config(config_file(tmp_path, start='2011-01-01',
        sensors='sensors = ["LT05", "LE07", "LC08", "LC09"]'))
    plan = run_config(config, dry_run=True)
    assert plan['calibration_scoring_window'][0] == '2011-01-01'
    assert config['run']['sensors'] == ['LT05', 'LE07', 'LC08', 'LC09']


def test_reused_cache_cannot_silently_add_earlier_missions(tmp_path):
    names = ['Landsat_LE07_2020-01-01.tif', 'Landsat_LT05_2020-01-02.tif',
             'Landsat_LC08_2020-01-03.tif', 'Landsat_LC09_2020-01-03.tif',
             'Landsat_LC08_2019-01-01.tif']
    for name in names:
        (tmp_path / name).touch()
    modern = _cached_scene_files(tmp_path, ['LC08', 'LC09'], '2020-01-01', '2020-01-31')
    assert modern == [str(tmp_path / names[3])]
    historical = _cached_scene_files(tmp_path, ['LT05', 'LE07', 'LC08', 'LC09'], '2020-01-01', '2020-01-31')
    assert historical == [str(tmp_path / names[i]) for i in (0, 1, 3)]
    with pytest.raises(ValueError, match='No cached Landsat scenes match'):
        _cached_scene_files(tmp_path, ['LC08', 'LC09'], '2020-01-01', '2020-01-02')
    (tmp_path / 'unidentified_2020-01-04.tif').touch()
    with pytest.raises(ValueError, match='explicit landsat.manifest_file'):
        _cached_scene_files(tmp_path, ['LC08', 'LC09'], '2020-01-01', '2020-01-31')


def test_initialisation_policy_allows_explicit_dates_and_later_studies(tmp_path):
    config = load_config(config_file(tmp_path,
        calibration='enabled = true\nstart = "2013-09-01"'))
    assert run_config(config, dry_run=True)['calibration_scoring_window'][0] == '2013-09-01'
    config = load_config(config_file(tmp_path, start='2020-01-01'))
    assert run_config(config, dry_run=True)['initialization_window'] is None
    with pytest.raises(ValueError, match='2013 is initialization'):
        load_config(config_file(tmp_path, end='2013-12-31'))


@pytest.mark.parametrize('area', [
    'center = [146.3, -35.0]\nextent = [146.2, -35.1, 146.4, -34.9]',
    'center = [146.3, -35.0]\nchip_size_m = 5000',
    'extent = [146.2, -35.1, 146.4, -34.9]\nchip_size_m = 5100',
    'center = [146.3, -35.0]\ngrid_mode = "legacy"',
])
def test_ambiguous_or_nonpixel_chip_settings_fail_before_execution(tmp_path, area):
    with pytest.raises(ValueError):
        load_config(config_file(tmp_path, area=area))


def test_explicit_extent_is_preserved(tmp_path):
    config = load_config(config_file(tmp_path, area='extent = [146.2, -35.1, 146.4, -34.9]'))
    assert config['run']['extent'] == [146.2, -35.1, 146.4, -34.9]
    assert 'center' not in config['run'] and 'chip_size_m' not in config['run']


@pytest.mark.parametrize('crs,center', [
    ('EPSG:32632', [9.0, 55.0]),
    ('EPSG:32755', [146.2908, -34.9878]),
    ('EPSG:32655', [146.2908, -34.9878]),
])
def test_default_chip_preserves_native_pixels_and_exact_5100_m_size(crs, center):
    x, y = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(*center)
    source = raster(np.arange(200*200).reshape(200, 200),
        Affine(30, 0, x-3005, 0, -30, y+3007), crs)
    chip = centered_template(source, center)
    assert chip.shape == (170, 170)
    np.testing.assert_array_equal(chip.values, source.values[15:185, 15:185])
    left, bottom, right, top = chip.rio.bounds()
    np.testing.assert_allclose([right-left, top-bottom], [5100, 5100], rtol=0, atol=1e-7)
    assert abs((left+right)/2-x) <= 15 and abs((bottom+top)/2-y) <= 15
    np.testing.assert_array_equal(chip.x, source.x[15:185])
    np.testing.assert_array_equal(chip.y, source.y[15:185])
    assert centered_template(source, center, 300).shape == (10, 10)
    with pytest.raises(ValueError, match='complete point chip'):
        centered_template(source.isel(x=slice(50, 150), y=slice(50, 150)), center)


def test_point_chip_crop_and_both_preparation_targets_use_identical_bounds(tmp_path, monkeypatch):
    config = load_config(config_file(tmp_path, start='2020-01-01', end='2020-01-01'))
    base = tmp_path / 'outputs/study'
    (base / 'ssebop').mkdir(parents=True)
    (base / 'inputs').mkdir()
    center = config['run']['center']
    x, y = Transformer.from_crs('EPSG:4326', 'EPSG:32655', always_xy=True).transform(*center)
    source = raster(np.arange(200*200).reshape(200, 200),
        Affine(30, 0, x-3005, 0, -30, y+3007), 'EPSG:32655')
    daily = source.expand_dims(time=np.array(['2020-01-01'], dtype='datetime64[D]')).rename('ET')
    daily.to_netcdf(base / 'ssebop/et_daily_ssebop_2020-01-01_2020-01-01.nc')
    extent = config['run']['extent']
    reference = raster(np.ones((12, 12)),
        Affine(.01, 0, extent[0]-.03, 0, -.01, extent[3]+.03), 'EPSG:4326')
    reference.rename('reference_ssm').to_netcdf(base / 'inputs/reference_ssm_native.nc')
    import pysweb
    calls = []
    monkeypatch.setattr(pysweb.swb, 'preprocess', lambda **kwargs: calls.append(kwargs))
    run_config(config, stages=['preprocess'])
    expected = centered_template(source, center)
    with xr.open_dataset(base / 'products/SSEBop_ET.nc', decode_coords='all') as result:
        assert result.ET.shape == (1, 170, 170)
        np.testing.assert_array_equal(result.ET[0], expected)
    assert len(calls) == 2
    assert {call['grid_mode'] for call in calls} == {'reference', 'landsat'}
    for call in calls:
        assert call['template_extent_crs'] == 'EPSG:32655'
        np.testing.assert_allclose(call['template_extent'], expected.rio.bounds(), rtol=0, atol=1e-7)
    receipt = json.loads((base / 'chip-window.json').read_text())
    assert receipt['shape'] == [170, 170] and receipt['size_m'] == 5100
    # A mixed cache must honour the same sensor selection during retrieval.
    landsat = base / 'inputs/landsat'
    landsat.mkdir()
    for sensor in ('LC08', 'LE07'):
        (landsat / f'Landsat_{sensor}_2020-01-01.tif').touch()
    retrievals = []
    monkeypatch.setattr(pysweb.ssebop, 'run', lambda **kwargs: retrievals.append(kwargs))
    run_config(config, stages=['ssebop'])
    assert len(retrievals) == 1
    assert retrievals[0]['scene_files'] == [str(landsat / 'Landsat_LC08_2020-01-01.tif')]
