"""
Script: test_site_workflow.py
Objective: Verify explicit scoring, portable operators, restart integrity and site isolation.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-23
Last updated: 2026-10-23
Inputs: Small synthetic fixtures and mocked stage calls; no external data or downloads.
Outputs: Assertions for the portable site workflow contracts.
Usage: python -m pytest tests/workflows/test_site_workflow.py
Dependencies: pytest, numpy, pandas, pysweb.
"""
from pathlib import Path
import copy
import importlib.util
import json
import socket
import numpy as np
import pandas as pd
import pytest
from pysweb.workflow_native import native_plan
from pysweb.site_evaluation import align_day,observation_groups
from pysweb.site_workflow import file_records,verified_records,site_lock,offline,science_identity,run_plan


def config(tmp_path):
    return dict(run=dict(root=str(tmp_path),name='site',start='2014-01-01',end='2025-12-31',
        analysis_start='2016-01-01',holdout_start='2021-01-01',state_timing='start',workers=1,
        sensors=['LC08','LC09']),calibration=dict(enabled=False),spinup=dict(enabled=False),
        forcing=dict(retrieval={}),soil={},model={},landsat={})


def test_explicit_actual_prefix(tmp_path):
    c=config(tmp_path)
    result=native_plan(c,['run'])
    assert result['initialization']['actual_warmup_days']==730
    assert result['initialization']['artificial_warmup_days']==0
    assert result['analysis_window']==['2016-01-01','2025-12-31']
    assert result['initialization_window']==['2014-01-01','2015-12-31']
    c['run']['analysis_start']='2021-01-01'
    with pytest.raises(ValueError,match='holdout'):
        native_plan(c,['run'])
    c['run']['analysis_start']='2013-01-01'
    with pytest.raises(ValueError,match='within'):
        native_plan(c,['run'])


def test_defaults_preserved(tmp_path):
    c=config(tmp_path);c['run'].pop('analysis_start')
    assert native_plan(c,['run'])['initialization']['analysis_start']=='2014-01-01'


def test_linear_day_integral_and_gap():
    dates=pd.date_range('2020-01-01',periods=4)
    f=pd.DataFrame(dict(continuous_trajectory_valid=[True,True,False,False],actual_et=[1,2,3,4],et_forcing=[2,4,6,8]),index=dates)
    for layer in range(1,6):
        f[f'start_layer_{layer}']=np.arange(4)*.1
        f[f'dailymean_layer_{layer}']=np.arange(4)*.1+.05
    result=align_day(f,10)
    np.testing.assert_allclose(result.aligned_layer_1.iloc[1],.15-.1*10/24)
    np.testing.assert_allclose(result.actual_et_aligned.iloc[1],2-10/24)
    assert result.paired_valid.tolist()==[False,True,False,False]
    np.testing.assert_array_equal(align_day(f,0).actual_et_aligned,f.actual_et)


def test_unknown_depth_and_footprint_excluded():
    raw=pd.DataFrame(dict(variable=['soil_moisture']*2,sensor_id=['cosmos','point'],
        depth_from_cm=[0,10],depth_to_cm=[100,10],support=['sensor_footprint_requires_review','nominal_point'],layer=['footprint_review','5-15cm']))
    groups,excluded=observation_groups(raw,'ISMN')
    assert len(excluded)==1 and excluded[0]['n']==0
    assert groups[0]['interval_mm']==[50.,150.] and groups[0]['layer']==2


def test_integrity_and_lock(tmp_path):
    path=tmp_path/'a';path.write_text('original')
    manifest=file_records([path]);assert verified_records(manifest)
    path.write_text('modified');assert not verified_records(manifest)
    with site_lock(tmp_path):
        with pytest.raises(RuntimeError,match='Another writer'):
            with site_lock(tmp_path):pass
    with site_lock(tmp_path):pass
    with offline():
        with pytest.raises(RuntimeError,match='Network'):
            socket.socket().connect(('127.0.0.1',1234))


def test_relocation_does_not_change_science(tmp_path):
    manifest=tmp_path/'scenes.json';manifest.write_text(json.dumps({'scenes':[{'scene_id':'immutable','file':'old/file.tif'}]}))
    c=config(tmp_path);c['landsat']['manifest_file']=str(manifest)
    a=dict(config_values=c,scientific_baseline='baseline')
    b=copy.deepcopy(a);b['config_values']['run'].update(root='/another/path',workers=4,name='relocated')
    assert science_identity(a)==science_identity(b)
    b['config_values']['model']['diff_factor']=99
    assert science_identity(a)!=science_identity(b)


def test_reference_selection_policy_changes_science_identity(tmp_path):
    manifest=tmp_path/'scenes.json';manifest.write_text(json.dumps({'scenes':[]}))
    c=config(tmp_path);c['landsat']['manifest_file']=str(manifest)
    a=dict(config_values=c,scientific_baseline='baseline',derive_physical_reference=True)
    b={**a,'reference_coverage_policy':dict(minimum_valid_fraction=.99,minimum_area_fraction=.5)}
    assert science_identity(a)!=science_identity(b)


def test_resume_after_failure_and_corruption(tmp_path,monkeypatch):
    import pysweb.site_workflow as w
    c=config(tmp_path)
    p=dict(config_values=c,base=str(tmp_path/'work'),destination=str(tmp_path/'dataset'),implementation_commit='test')
    dependency=tmp_path/'input';dependency.write_text('input')
    monkeypatch.setattr(w,'science_identity',lambda _: {'scientific':'same'})
    monkeypatch.setattr(w,'source_identity',lambda:'software')
    monkeypatch.setattr(w,'external_inputs',lambda _: [dependency])
    counts={'calls':0}
    def acquire(plan):
        counts['calls']+=1
        out=Path(plan['base'])/'output';out.write_text('accepted')
        return [out]
    monkeypatch.setattr(w,'acquire',acquire)
    run_plan(p,['acquire']);run_plan(p,['acquire']);assert counts['calls']==1
    (Path(p['base'])/'output').write_text('corruption')
    run_plan(p,['acquire']);assert counts['calls']==2
    def failed(plan):raise RuntimeError('interruption')
    monkeypatch.setattr(w,'retrieve',failed)
    with pytest.raises(RuntimeError,match='interruption'):run_plan(p,['retrieve'])
    assert json.loads((Path(p['base'])/'receipts/acquire.json').read_text())['passed']
    run_plan(p,['acquire']);assert counts['calls']==2


def test_scheduler_scripts_and_frozen_scope(tmp_path):
    path=Path(__file__).resolve().parents[2]/'workflows/hpc/gadi/submit_sites.py'
    spec=importlib.util.spec_from_file_location('gadi_submit',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    plan=tmp_path/'site.json';plan.write_text(json.dumps(dict(site_key='AU_site',region='AU',config='config.toml',destination=str(tmp_path/'dataset'))))
    execution=dict(resources={'compute':dict(queue='normalsr',ncpus=4,mem_gib=192,jobfs_gib=128,walltime='06:00:00')},
        storage=['gdata/project','scratch/project'],project='project',modules=['configured/1'],logs=str(tmp_path),
        python='/configured/python',resource_helper='/configured/resources.py',source_archive='/configured/source.tar.gz',source_sha256='abc')
    text=module.pbs_script(execution,plan,'compute')
    assert 'ncpus=4,mem=192GB,jobfs=128GB' in text
    assert '--stages compute' in text and 'depend=' not in text
    assert '#PBS -l storage=gdata/project+scratch/project' in text
    assert '--storage gdata/project scratch/project' in text
    cohort=tmp_path/'cohort.json';cohort.write_text(json.dumps({'sites':[dict(name='AU_site',region='AU')]}))
    execution['cohort']=str(cohort)
    with pytest.raises(ValueError,match='outside'):module.submit(execution,['NA_site'])


def test_isolated_stage_rejects_stale_ancestor(tmp_path,monkeypatch):
    import pysweb.site_workflow as w
    c=config(tmp_path)
    p=dict(config_values=c,base=str(tmp_path/'work'),destination=str(tmp_path/'dataset'),implementation_commit='test')
    dependency=tmp_path/'input';dependency.write_text('original')
    monkeypatch.setattr(w,'science_identity',lambda _: {'scientific':'same'})
    monkeypatch.setattr(w,'source_identity',lambda:'software')
    monkeypatch.setattr(w,'external_inputs',lambda _: [dependency])
    def output(name):
        def stage(plan):
            path=Path(plan['base'])/name;path.write_text('accepted');return [path]
        return stage
    for name in ('acquire','retrieve','preprocess'):monkeypatch.setattr(w,name,output(name))
    run_plan(p,['acquire','retrieve'])
    (Path(p['base'])/'acquire').write_text('corrupt ancestor')
    with pytest.raises(ValueError,match='Prerequisite outputs changed: acquire'):run_plan(p,['preprocess'])
    (Path(p['base'])/'acquire').write_text('accepted')
    dependency.write_text('changed forcing')
    with pytest.raises(ValueError,match='Prerequisite inputs changed: acquire'):run_plan(p,['preprocess'])
    run_plan(p,['acquire','retrieve','preprocess'])
    assert json.loads((Path(p['base'])/'receipts/preprocess.json').read_text())['passed']


def test_figure_bounds_check_only_drawn_ticks(tmp_path):
    from pysweb.visualisation.site_figures import save_figure
    import matplotlib.pyplot as plt
    with plt.rc_context({'pdf.fonttype':42,'font.family':'DejaVu Sans'}):
        fig,ax=plt.subplots(figsize=(4,3))
        ax.set_xticks([0,1,2]);ax.set_xlim(0,1.1)
        save_figure(fig,tmp_path,'ticks',[])
        fig,ax=plt.subplots(figsize=(4,3));fig.text(2,1,'outside')
        with pytest.raises(ValueError,match='outside canvas'):save_figure(fig,tmp_path,'invalid',[])
        plt.close(fig)
