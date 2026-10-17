"""
Script: test_protocols.py
Objective: Verify holdout isolation, donor restrictions, spin-up and evaluation contracts.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Small deterministic synthetic scenes, forcing and observation arrays.
Outputs: Scientific protocol regression assertions.
Usage: python -m pytest tests/test_protocols.py
Dependencies: numpy, pandas, pytest, xarray, pysweb
"""
import importlib.util
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pysweb.forcing import daily_support, reconstruct, paired_support, complete_support, precipitation_qc
from pysweb.swb.spinup import cycle_indices, verify_convergence, prepend_cycles
from pysweb.ssebop.preprocessing import donors_for, select_scenes, fill_bands
from pysweb.evaluation import depth_weights, metrics, evaluate_series, cell_day_objective, aggregate_probe_storage

SOLVER_BACKEND = "numba" if importlib.util.find_spec("numba") is not None else "reference"


def test_cycle_phase_and_holdout_rejection():
    dates = pd.date_range("2010-01-01", "2025-12-31")
    indices, metadata = cycle_indices(dates, "2013-05-17", 2, 2014, "2020-12-31")
    assert len(indices) == 730
    assert dates[indices[0]] == pd.Timestamp("2014-05-17")
    assert dates[indices[-1]] == pd.Timestamp("2014-05-16")
    assert (dates[indices].year == 2014).all()
    leap, _ = cycle_indices(dates, "2012-02-29", 1, 2014, "2020-12-31")
    assert dates[leap[-1]] == pd.Timestamp("2014-02-28")
    with pytest.raises(ValueError, match="development year"):
        cycle_indices(dates, "2013-05-17", 2, 2021, "2020-12-31")
    with pytest.raises(ValueError, match="Complete source"):
        cycle_indices(dates[dates != pd.Timestamp("2014-05-01")], "2013-05-17", 2, 2014, "2020-12-31")


def scene(name, date, sensor="LE07", path=92, row=84, cloud=1):
    return dict(scene_id=name, sensor=sensor, date=date, path=path, row=row, cloud_cover=cloud, file=name+".tif")


def test_donor_restrictions_and_same_day_choice():
    target = scene("target", "2020-12-20")
    records = [target, scene("holdout", "2021-01-01"), scene("other_sensor", "2020-12-19", "LC08"),
        scene("other_wrs", "2020-12-19", path=91), scene("distant", "2020-01-01"),
        scene("z", "2020-12-04", cloud=2), scene("a", "2020-12-04", cloud=1)]
    donors = donors_for(target, records, start="2010-01-01", end="2020-12-31")
    assert [r["scene_id"] for r in donors] == ["a", "z"]
    selected = select_scenes(records, ["LE07"], "2010-01-01", "2025-12-31",
        sensor_windows={"LE07": ["2010-01-01", "2020-12-31"]}, wrs=[92, 84])
    assert [r["scene_id"] for r in selected] == ["distant", "a", "target"]


def test_gapfill_preserves_original_and_unrequested_holes():
    y, x = np.indices((21, 21))
    raw = np.array([280+x+y, .2+x*.002+y*.003, .4+x*.003+y*.002])
    donor = raw.copy()
    target = raw.copy()
    target[:, 10, 10] = np.nan
    target[:, 2, 2] = np.nan
    gap = np.zeros((21, 21), bool)
    gap[10, 10] = True
    result = fill_bands(target, [donor], gap, min_common=16, max_window=9)
    for band, item in enumerate(result):
        np.testing.assert_array_equal(item["values"][np.isfinite(target[band])], target[band][np.isfinite(target[band])])
        assert item["values"][10, 10] == pytest.approx(raw[band, 10, 10])
        assert np.isnan(item["values"][2, 2])
        assert item["provenance"][10, 10] == 2


def test_daily_support_uses_actual_pixel_observations():
    dates = pd.to_datetime(["2020-01-01", "2020-01-17", "2020-02-02", "2020-02-18"])
    valid = np.array([[True, True], [True, False], [True, False], [True, True]])
    result = daily_support(valid, dates, pd.date_range(dates[0], dates[-1]), 32)
    assert result[:, 0].all()
    assert result[[0, -1], 1].all() and not result[1:-1, 1].any()


def test_reconstruction_never_crosses_holdout_and_retains_flags():
    dates = pd.to_datetime(["2020-12-01", "2020-12-17", "2021-01-02", "2021-01-18"])
    data = xr.DataArray(np.array([.2, .2, .9, .9])[:, None, None], dims=("time", "y", "x"),
                        coords={"time": dates, "y": [0], "x": [0]})
    scenes = xr.Dataset(dict(etf=data, ndvi=data, original_support=xr.ones_like(data)))
    result = reconstruct(scenes, pd.date_range("2020-12-01", "2021-01-31"), development_end="2020-12-31",
                         acquisition_max_gap_days=64, boundary_carry=True)
    assert result.etf.sel(time="2020-12-31").item() == pytest.approx(.2)
    assert result.etf.sel(time="2021-01-01").item() == pytest.approx(.9)
    assert result.boundary_extrapolated.sel(time="2020-12-31").item() == 1
    assert result.original_supported.sel(time="2020-12-31").item() == 0
    assert result.reconstructed.sel(time="2020-12-31").item() == 1


def test_support_excludes_missing_development_but_ignores_test():
    dates = pd.date_range("2020-12-29", "2021-01-02")
    a = xr.DataArray(np.ones((5, 1, 2)), dims=("time", "y", "x"), coords={"time": dates, "y": [1], "x": [1, 2]})
    a[1, 0, 0] = np.nan
    a[-1, 0, 1] = np.nan
    forcing = xr.Dataset({k: a for k in ("effective_precip", "et", "E_t", "ndvi")})
    soil = xr.ones_like(a[0], dtype=bool).drop_vars("time")
    mask = complete_support(forcing, soil, "2020-12-29", "2020-12-31")
    np.testing.assert_array_equal(mask, [[False, True]])
    np.testing.assert_array_equal(paired_support([mask, soil]), mask)
    with pytest.raises(ValueError):
        paired_support([mask, soil.assign_coords(x=[2, 3])])


def test_precipitation_correction_is_bounded_and_flagged():
    raw = xr.DataArray([-.0005, -.1, 2., np.nan], dims="time", coords={"time": pd.date_range("2020-01-01", periods=4)}, name="rain")
    output, gate = precipitation_qc(raw, small_negative_tolerance_mm=.001, maximum_annual_correction_mm=.1)
    np.testing.assert_allclose(output.rain, [0, np.nan, 2, np.nan], equal_nan=True)
    np.testing.assert_array_equal(output.precipitation_qc, [1, 2, 0, 2])
    assert gate["passed"] and gate["small_negative_values"] == 1


def test_reference_initialization_is_missing(tmp_path):
    dates = pd.date_range("2014-01-01", "2015-01-02")
    source, output = tmp_path/"reference.nc", tmp_path/"initialized.nc"
    xr.DataArray(np.ones(len(dates))*.2, dims="time", coords={"time": dates}, name="reference_ssm").to_netcdf(source)
    prepend_cycles(source, output, "2014-01-01", "2015-01-02", source_year=2014,
        repeats=1, development_end="2014-12-31", reference=True, scoring_start="2014-02-01")
    with xr.open_dataset(output) as ds:
        assert ds.reference_ssm.sel(time=slice(None, "2014-01-31")).isnull().all()
        assert ds.reference_ssm.sel(time="2015-01-01").isnull()
        assert float(ds.reference_ssm.sel(time="2014-02-01")) == pytest.approx(.2)


def test_depth_and_weighted_cell_day_objective():
    np.testing.assert_array_equal(depth_weights([50, 150, 300, 600, 1000], [30, 750]), [20, 100, 150, 300, 150])
    model = xr.DataArray([[[0., 1.]], [[1., 0.]]], dims=("time", "y", "x"))
    reference = xr.full_like(model, .5)
    mask = xr.ones_like(model[0], dtype=bool)
    weights = mask.astype(float)*xr.DataArray([1., 3.], dims="x")
    result = cell_day_objective(model, reference, mask, weights)
    assert result["rmse"] == .5 and result["n_cell_days"] == 4
    model[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="frozen"):
        cell_day_objective(model, reference, mask, weights)


def test_metrics_own_and_common_support_are_explicit():
    dates = pd.date_range("2020-12-30", "2021-01-02")
    a = pd.DataFrame(dict(sim=[1, 2, 3, 4], continuous_trajectory_valid=[True]*4), index=dates)
    b = a.copy()
    b.loc[dates[1], "sim"] = np.nan
    obs = pd.DataFrame(dict(obs=[1, 1, 1, 1]), index=dates)
    rows = evaluate_series(dict(a=a, b=b), obs, pairs={"surface": dict(model="sim", observed="obs", units="m3 m-3")},
        phases={"development": ["2020-01-01", "2020-12-31"], "test": ["2021-01-01", "2021-12-31"]},
        scoring_starts=dict(a="2020-12-30", b="2020-12-30"))
    selected = rows[(rows.model == "a") & (rows.phase == "development") & (rows.period == "pooled")]
    assert selected.set_index("support").n.to_dict() == dict(own=2, common=1)
    assert metrics([1, 2], [1, 1])["pearson_r"] is None


def test_fixed_joint_probe_support():
    observations = pd.DataFrame(dict(a=np.ones(48)*.2, b=np.ones(48)*.3, qa=np.zeros(48), qb=np.zeros(48)),
        index=pd.date_range("2020-01-01", periods=48, freq="30min"))
    observations.loc[observations.index[:13], "qa"] = 1
    daily = aggregate_probe_storage(observations, columns=["a", "b"], weights_mm=[100, 200],
        minimum_joint_samples=36, expected_samples=48, qc_columns=["qa", "qb"])
    assert daily.joint_samples.iloc[0] == 35 and np.isnan(daily.storage_mm.iloc[0])


def test_configured_native_support_path(tmp_path, monkeypatch):
    from tests.workflows import test_native_workflow as existing
    configuration = existing.configuration
    def configured(path):
        config = configuration(path)
        config["forcing"] = {"peer_runs": []}
        return config
    monkeypatch.setattr(existing, "configuration", configured)
    existing.test_native_full_offline_workflow(tmp_path)
    with xr.open_dataset(tmp_path/"native/run_inputs/initialized/fit_mask.nc") as ds:
        assert ds.fit_mask.any()


def test_wet_dry_convergence_and_missing_history():
    dates = pd.date_range('2013-01-01','2014-12-31')
    data = xr.DataArray(np.zeros((len(dates),1,1)),dims=('time','y','x'),coords=dict(time=dates,y=[0],x=[0]))
    forcing = xr.Dataset(dict(effective_precip=data,et=data+3,E_t=data+2,ndvi=data+.5))
    soil = xr.Dataset({k: xr.DataArray(np.full((2,1,1),v),dims=('layer','y','x'),
        coords=dict(layer=[1,2],y=[0],x=[0])) for k,v in dict(porosity=.45,wilting_point=.12,
        available_water_capacity=.2,b_coefficient=4.5,conductivity_sat=8.).items()})
    soil = soil.assign_coords(layer_depth=('layer',[50.,200.]))
    mask = data[0].drop_vars('time').astype(bool) | True
    parameters = dict(diff_factor=1000.,sm_max_factor=1.,sm_min_factor=1.,root_beta=.96)
    gate,difference = verify_convergence(forcing,soil,mask,parameters,scoring_start='2014-01-01',
        development_end='2014-12-31',tolerance=.005,minimum_fraction=.99,solver_backend=SOLVER_BACKEND)
    assert gate['passed'] and gate['initialization_days'] == 365
    assert np.isfinite(difference).all()
    forcing.et[30] = np.nan
    with pytest.raises(ValueError,match='missing forcing'):
        verify_convergence(forcing,soil,mask,parameters,scoring_start='2014-01-01',
            development_end='2014-12-31',tolerance=.005,minimum_fraction=.99,solver_backend=SOLVER_BACKEND)


def test_l7_file_gate_and_receipt_reject_mutation(tmp_path):
    import rasterio
    from rasterio import Affine
    from pysweb.ssebop.preprocessing import prepare_l7, read_verified_product
    records=[]
    # Two half-years, with a near-date original donor for each target. Identical
    # physical fields make the withholding result known without fitting errors.
    for i,date in enumerate(['2014-02-01','2014-02-17','2014-08-01','2014-08-17']):
        raw=tmp_path/f'LE07_092084_{pd.Timestamp(date):%Y%m%d}.tif'
        mask=tmp_path/f'mask_{i}.tif'
        y,x=np.indices((40,40))
        values=np.array([40000+x+y,15000+2*x+y,20000+2*x+y,np.zeros_like(x),np.zeros_like(x)],dtype='uint16')
        gap=np.zeros((40,40),dtype='uint8')
        if i%2==0:
            gap[15:25,20]=1
            values[:3,gap.astype(bool)]=0
        profile=dict(driver='GTiff',width=40,height=40,crs='EPSG:32655',transform=Affine(30,0,400005,0,-30,-3800005))
        with rasterio.open(raw,'w',count=5,dtype='uint16',**profile) as dst:
            dst.write(values)
            dst.descriptions=('ST_B6','SR_B3','SR_B4','QA_PIXEL','QA_RADSAT')
        with rasterio.open(mask,'w',count=2,dtype='uint8',**profile) as dst:
            dst.write(np.array([gap,np.ones_like(gap)]))
        records.append({**scene(str(raw.stem),date,cloud=i%2), 'file':str(raw),'gap_mask_file':str(mask)})
    config=dict(start='2014-01-01',end='2014-12-31',min_common=16,max_window=9,
        validation=dict(years=[2014],minimum_passed_scenes=2,crop_pixels=40,stripe_shift_rows=7,border_pixels=2,
            min_scored_pixels=5,minimum_fill_fraction=.8,LST_RMSE_max_K=3.,LST_absolute_bias_max_K=1.,NDVI_RMSE_max=.1))
    outputs=prepare_l7(records,tmp_path/'processed',config,development_end='2014-12-31')
    product=read_verified_product(outputs[records[0]['file']],records[0]['file'])
    assert product.lst_K_provenance.values[20,20] == 2
    from pysweb.ssebop.api import process_landsat_scene
    from pysweb.io.grids import open_template
    template=open_template(records[0]['file'])
    delta=(template.astype(float)+10).expand_dims(dayofyear=[32])
    (tmp_path/'etf').mkdir()
    (tmp_path/'ndvi').mkdir()
    process_landsat_scene(records[0]['file'],'auto','ndvi','SR_B3','SR_B4',False,None,
        delta,template.rio.crs,str(tmp_path/'etf'),str(tmp_path/'ndvi'),gapfill_product=outputs[records[0]['file']])
    with xr.open_dataset(tmp_path/'etf/auxiliary_2014-02-01.nc') as ds:
        assert ds.lst_K_provenance.values[20,20] == 2
        assert ds.original_support.values[20,20] == 0
    with rasterio.open(records[0]['file'],'r+') as dst:
        data=dst.read(1)
        data[0,0]+=1
        dst.write(data,1)
    with pytest.raises(ValueError,match='inputs changed'):
        read_verified_product(outputs[records[0]['file']],records[0]['file'])


def test_calendar_climatology_requires_full_source_and_ignores_holdout():
    from tests.io.test_native_grids import raster
    from rasterio import Affine
    from pysweb.ssebop.scene_protocol import calendar_day_dt
    dates=pd.date_range('2014-01-01','2015-12-31')
    grid=raster(np.ones((2,2)),Affine(.1,0,146,0,-.1,-34),'EPSG:4326')
    fields={k:(grid*v).expand_dims(time=dates).copy() for k,v in dict(tmax=30.,tmin=15.,rs=20.,ea=1.5).items()}
    for field in fields.values():
        field.loc[dict(time=slice('2015-01-01',None))]=np.nan
    dt=calendar_day_dt(fields,grid*100,grid*-34,'2025-03-01',start='2014-01-01',end='2014-12-31')
    assert np.isfinite(dt).all()
    fields['tmax']=fields['tmax'].drop_sel(time='2014-03-01')
    with pytest.raises(ValueError,match='missing calendar'):
        calendar_day_dt(fields,grid*100,grid*-34,'2025-03-01',start='2014-01-01',end='2014-12-31')


def test_screened_scene_workflow_offline(tmp_path):
    import rasterio
    from rasterio import Affine
    from rasterio.warp import transform_bounds
    from tests.io.test_native_grids import raster
    from pysweb.ssebop.scene_protocol import run_scene_workflow
    transform=Affine(30,0,496160,0,-30,6103840)
    source_grid=raster(np.ones((256,256)),transform)
    target=source_grid.isel(x=slice(122,134),y=slice(122,134))
    extent=list(transform_bounds(target.rio.crs,'EPSG:4326',*target.rio.bounds(),densify_pts=21))
    scenes=[]
    for day in ['2014-06-01','2014-06-05']:
        path=tmp_path/f'LC08_001001_{day.replace("-","")}.tif'
        data=np.stack([np.full((256,256),42000),np.full((256,256),12000),np.full((256,256),23000),
            np.zeros((256,256)),np.zeros((256,256))]).astype('uint16')
        with rasterio.open(path,'w',driver='GTiff',width=256,height=256,count=5,dtype='uint16',crs=source_grid.rio.crs,transform=transform) as dst:
            dst.write(data)
            dst.descriptions=('ST_B10','SR_B4','SR_B5','QA_PIXEL','QA_RADSAT')
        scenes.append(str(path))
    dem=tmp_path/'dem.tif'
    (source_grid*100).rio.to_raster(dem)
    geographic=raster(np.ones((6,6)),Affine(.1,0,8.7,0,-.1,55.3),'EPSG:4326')
    dates=pd.date_range('2014-01-01','2014-12-31')
    files={}
    for name,value in dict(tmax=30.,tmin=15.,rs=20.,ea=1.5,et_tall_crop=4.).items():
        path=tmp_path/f'{name}.nc'
        (geographic*value).expand_dims(time=dates).rename(name).to_netcdf(path)
        files[name]=str(path)
    padding=dict(primary_bounds=[497120,6097120,502880,6102880],comparison_bounds=list(source_grid.rio.bounds()),
        crs='EPSG:32632',maximum_fallback_fraction=1e-8,minimum_agreement=.99,maximum_p99_difference=.01)
    protocol=dict(development_end='2014-12-31',climatology_start='2014-01-01',climatology_end='2014-12-31',
        meteorology_files=files,scene_cache_dir=str(tmp_path/'cache'),padding=padding)
    output=run_scene_workflow(date_range='2014-06-01 to 2014-06-05',scene_files=scenes,gapfill_products={},
        dem=str(dem),output_dir=tmp_path/'seb',extent=extent,protocol=protocol,
        reconstruction=dict(acquisition_max_gap_days=64,boundary_carry=True),workers=1)
    with xr.open_dataset(output,decode_coords='all') as ds:
        assert ds.sizes['time']==5 and ds.sizes['scene_time']==2
        assert ds.ET.notnull().any()
        assert {'original_supported','reconstructed','dt_K','cold_boundary_fallback_fraction'} <= set(ds)
        assert ds.rio.crs == source_grid.rio.crs
    with pytest.raises(ValueError,match='before first supported scene'):
        run_scene_workflow(date_range='2014-05-31 to 2014-06-05',scene_files=scenes,gapfill_products={},
            dem=str(dem),output_dir=tmp_path/'early',extent=extent,
            protocol={**protocol,'require_supported_start':True},
            reconstruction=dict(acquisition_max_gap_days=64,boundary_carry=True),workers=1)


def test_configured_spinup_products_and_endpoint_operator(tmp_path):
    """Run real orchestration, convergence, fitting, reproduction and extraction."""
    from tests.io.test_native_grids import raster
    from rasterio import Affine
    from pyproj import Transformer
    from pysweb.workflow_native import run_native
    from pysweb.workflow_protocol import FORCING_FILES, write_support
    from pysweb.swb.calibrate import SOIL_FILE_STEMS
    from pysweb.evaluation import extract_point
    from pysweb.io.provenance import atomic_json
    import json
    base = tmp_path/'spinup'
    coarse, fine = base/'calibration_inputs', base/'run_inputs'
    dates = pd.date_range('2014-01-01','2015-01-05')
    geographic = raster(np.ones((2,2)),Affine(.01,0,9,0,-.01,55.04),'EPSG:4326')
    projected = raster(np.ones((2,2)),Affine(30,0,500000,0,-30,6100000))
    properties = dict(porosity=.45,wilting_point=.12,available_water_capacity=.2,
                      b_coefficient=4.5,conductivity_sat=8.)
    for folder,grid,calendar in ((coarse,geographic,dates[dates.year==2014]),(fine,projected,dates)):
        folder.mkdir(parents=True)
        if folder == coarse:
            grid=grid.rename(y='lat',x='lon').rio.set_spatial_dims(x_dim='lon',y_dim='lat')
        tag=f'{calendar[0]:%Y%m%d}_{calendar[-1]:%Y%m%d}'
        for name,(stem,variable) in FORCING_FILES.items():
            value={'et':3.,'E_t':2.,'ndvi':.6,'E_s':1.}.get(name,0.)
            array=(grid*value).expand_dims(time=calendar).rename(variable)
            array.attrs['units']='1' if name=='ndvi' else 'mm day-1'
            array.to_netcdf(folder/f'{stem}_daily_{tag}.nc')
        for name,value in properties.items():
            array=(grid*value).expand_dims(layer=[1,2]).rename(name)
            array=array.assign_coords(layer_depth=('layer',[50.,200.]))
            array.attrs['layer_bottoms_mm']=[50.,200.]
            array.to_netcdf(folder/SOIL_FILE_STEMS[name])
        if folder==coarse:
            reference=(grid*.12).expand_dims(time=calendar).rename('reference_ssm')
            reference.attrs['units']='m3 m-3'
            reference.to_netcdf(folder/f'reference_ssm_daily_{tag}.nc')
        write_support(folder,'2014-01-01',str(calendar[-1].date()),'2014-12-31')
    (base/'products').mkdir()
    (projected*3).expand_dims(time=dates).rename('ET').to_netcdf(base/'products/SSEBop_ET.nc')
    config=dict(run=dict(name='spinup',root=str(tmp_path),start='2014-01-01',end='2015-01-05',
        extent=[9,55.01,9.02,55.04],workers=1,state_timing='start',grid_mode='native',
        sensors=['LC08'],holdout_start='2015-01-01'),soil=dict(source='custom'),
        calibration=dict(enabled=True,start='2014-01-01',end='2014-12-31',max_iter=2,
            no_polish=True,serial_polish=True,solver_backend=SOLVER_BACKEND,seed=0,diff_bounds=[1000.,1000.],
            sm_max_bounds=[1.,1.],sm_min_bounds=[1.,1.],beta_bounds=[.96,.96]),
        model=dict(solver_backend=SOLVER_BACKEND),forcing=dict(peer_runs=[]),
        spinup=dict(source_year=2014,repeats=2,tolerance=.005,minimum_fraction=.99,solver_backend=SOLVER_BACKEND))
    run_native(config,['benchmark'])
    assert not (base/'calibration.csv').exists()
    benchmark=json.loads((base/'calibration-benchmark.json').read_text())
    assert benchmark['cells']==4 and benchmark['simulated_days']==1095
    run_native(config,['calibrate','run'])
    acceptance=json.loads((base/'calibration-acceptance.json').read_text())
    assert acceptance['passed'] and acceptance['same_pair_count']
    for folder in (coarse,fine):
        for label in ('default','calibrated'):
            gate=json.loads((folder/'initialized'/f'{label}_convergence.json').read_text())
            assert gate['passed'] and gate['initialization_days']==730
    product=base/'products/SWEB_RZSM.nc'
    longitude,latitude=Transformer.from_crs(projected.rio.crs,4326,always_xy=True).transform(
        projected.x.values[0],projected.y.values[0])
    table,operator=extract_point(product,longitude=longitude,latitude=latitude,depth_interval_mm=[30,200])
    assert len(table)==len(dates) and table.continuous_trajectory_valid.all()
    with xr.open_dataset(product) as ds:
        assert ds.sizes['time']==len(dates)
        assert ds.initial_history_valid_default.astype(bool).all()
        last=ds.rzsm_calibrated.isel(time=-1,y=0,x=0).values
        final=np.array([ds[f'final_sm_layer_{i}_calibrated'].isel(y=0,x=0).item() for i in (1,2)])
        assert table.storage_dailymean_mm.iloc[-1]==pytest.approx(((last+final)/2)@np.array([20.,150.]))
    # Valid acceptance cannot be reused after changing development support.
    path=coarse/'development_support.nc'
    with xr.open_dataset(path) as ds:
        modified=ds.load()
    modified.development_support[0,0]=0
    modified.to_netcdf(path)
    with pytest.raises(ValueError,match='changed since acceptance'):
        run_native(config,['run'])


def test_seasonal_fraction_support_and_holdout_isolation(tmp_path):
    from pysweb.forcing import seasonal_fraction,write_seasonal_seb
    from tests.io.test_native_grids import raster
    from rasterio import Affine
    dates=pd.date_range('2018-01-01','2020-03-01')
    values=np.ones((len(dates),2,2))*.2
    values[dates.year==2020]=.9
    actual=seasonal_fraction(values,dates,'2018-01-01','2019-12-31',dates)
    np.testing.assert_allclose(actual,.2)
    missing=values.copy();missing[dates.year==2019,0,0]=np.nan
    result=seasonal_fraction(missing,dates,'2018-01-01','2019-12-31',dates)
    assert np.isnan(result[:,0,0]).all()
    grid=raster(np.ones((2,2)),Affine(30,0,500000,0,-30,6100000))
    fraction=grid.expand_dims(time=dates).copy(data=values)
    ds=xr.Dataset(dict(ET=fraction*5,E_t=fraction*3,E_s=fraction*2,
        ndvi_interp=xr.ones_like(fraction)*.65,f_t=xr.ones_like(fraction)*.6,
        etf_interp=fraction,reference_et=xr.ones_like(fraction)*5))
    source=tmp_path/'source.nc';destination=tmp_path/'seasonal.nc'
    ds.to_netcdf(source)
    write_seasonal_seb(source,destination,development_start='2018-01-01',development_end='2019-12-31',block_rows=1)
    with xr.open_dataset(destination,decode_coords='all') as control:
        assert control.rio.crs==grid.rio.crs
        np.testing.assert_allclose(control.ET,1.)
        np.testing.assert_allclose(control.E_t,.6)
        np.testing.assert_allclose(control.ET,control.E_t+control.E_s)
        np.testing.assert_array_equal(control.ndvi_interp,ds.ndvi_interp)
