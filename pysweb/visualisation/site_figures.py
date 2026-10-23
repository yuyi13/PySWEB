"""
Script: site_figures.py
Objective: Render concise, numerically checked figures for one accepted site trajectory.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-23
Last updated: 2026-10-23
Inputs: Exact evaluated series, pairs, native maps and five-layer heatmap input.
Outputs: PDF and 300-dpi PNG pages, inspection crops and automatic render receipts.
Usage: from pysweb.visualisation.site_figures import make_figures
Dependencies: matplotlib, numpy, pandas, Pillow, pysweb; optional Ghostscript.
"""
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.lines import Line2D
from PIL import Image
from pysweb.io.provenance import atomic_json
from pysweb.evaluation import metrics
from pysweb.visualisation import plot_heatmap

COLOURS = dict(default='#E18727',forcing='#0072B5',observed='#000000')
FS, TS, DPI = 10, 18, 300


def save_figure(fig, directory, name, checks):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width,height = fig.canvas.get_width_height()
    for item in fig.findobj(Text):
        if item.get_visible() and item.get_text():
            b = item.get_window_extent(renderer)
            if b.width and b.height and (b.x0 < -1 or b.y0 < -1 or b.x1 > width+1 or b.y1 > height+1):
                raise ValueError(f'{name}: text outside canvas: {item.get_text()}')
    fig.savefig(directory/f'{name}.png',dpi=DPI,facecolor='white')
    fig.savefig(directory/f'{name}.pdf',facecolor='white')
    raw = (directory/f'{name}.pdf').read_bytes()
    assert b'/Subtype /Type3' not in raw and b'/FontFile2 ' in raw
    if shutil.which('gs'):
        subprocess.run(['gs','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=nullpage',str(directory/f'{name}.pdf')],check=True,capture_output=True)
    inspect = directory/'inspection'
    inspect.mkdir(exist_ok=True)
    with Image.open(directory/f'{name}.png') as im:
        assert im.size == tuple(round(v*DPI) for v in fig.get_size_inches())
        thumb=im.copy()
        thumb.thumbnail((1500,1500))
        thumb.save(inspect/f'{name}_overview.png')
        w,h=im.size
        for tag,box in [('top',(0,0,w,min(h,700))),('bottom',(0,max(0,h-700),w,h))]:
            im.crop(box).save(inspect/f'{name}_{tag}.png')
    checks.append(dict(page=name,size_inches=fig.get_size_inches().tolist(),dpi=DPI,
                       canvas_bounds=True,embedded_font=True))
    plt.close(fig)


def time_axis(ax, run):
    ax.axvspan(pd.Timestamp(run['start']),pd.Timestamp(run['analysis_start']),color='#FFF3CF',zorder=0)
    ax.axvspan(pd.Timestamp(run['holdout_start']),pd.Timestamp(run['end']),color='#EAEAEA',zorder=0)
    ax.set_xlim(pd.Timestamp(run['start']),pd.Timestamp(run['end']))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(axis='y',color='.9',lw=.5)


def finite_limits(values, zero=True):
    a = np.concatenate([np.asarray(v,dtype=float).ravel() for v in values])
    a = a[np.isfinite(a)]
    if not len(a):
        return [0,1]
    low,high = float(a.min()),float(a.max())
    if zero:
        low = min(0,low)
    pad = max((high-low)*.05,.001)
    return [low-pad,high+pad]


def make_figures(plan):
    base = Path(plan['base'])
    data,out = base/'evaluation',base/'figures'
    out.mkdir(parents=True,exist_ok=True)
    available = {f.name for f in font_manager.fontManager.ttflist}
    required = {ord(c) for c in '³⁻¹–·'}
    font = next(n for n in ('Arial','DejaVu Sans') if n in available and required <= set(font_manager.get_font(font_manager.findfont(n)).get_charmap()))
    plt.rcParams.update({'font.family':font,'font.size':FS,'axes.labelsize':FS,'axes.titlesize':TS,
        'xtick.labelsize':FS,'ytick.labelsize':FS,'legend.fontsize':FS,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':None})
    frame = pd.read_csv(data/'point_series.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    pairs = pd.read_csv(data/'observation_pairs.csv.gz',parse_dates=['date'],float_precision='round_trip')
    evaluation = json.loads((data/'observation_evaluation.json').read_text())
    operators = evaluation['operators']
    point = json.loads((data/'point_operator.json').read_text())
    hydraulic = json.loads((data/'hydraulics.json').read_text())
    run = plan['config_values']['run']
    label = plan.get('label',run['name'])
    checks,statistics = [],[]
    storage_ops = [op for op in operators if op['units']=='mm']
    rows = 5 if storage_ops else 4
    fig,axes = plt.subplots(rows,1,figsize=(14,15 if storage_ops else 12),sharex=True)
    fig.subplots_adjust(left=.09,right=.98,bottom=.095,top=.88,hspace=.50)
    fig.text(.09,.975,label+' · uncalibrated',fontsize=TS,va='top')
    handles = [Line2D([],[],color=COLOURS[k],lw=2,label=v) for k,v in [('default','SWEB'),('forcing','SSEBop ET forcing')]]
    handles.append(Line2D([],[],color='black',ls='none',marker='o',ms=5,label='Matched primary-QC observations'))
    fig.legend(handles=handles,ncol=3,loc='upper left',bbox_to_anchor=(.085,.945),frameon=False)
    for layer,ax in enumerate(axes[:3],1):
        ax.plot(frame.index,frame[f'aligned_layer_{layer}'].where(frame.paired_valid),color=COLOURS['default'],lw=.65)
        for op in operators:
            if op['layer']==layer:
                selected = pairs.loc[pairs.operator_id.eq(op['operator_id'])&pairs.qc.eq('clean')]
                ax.scatter(selected.date,selected.observed,s=2,c='black',rasterized=True)
        ax.set_ylim(0,max(hydraulic['porosity'][:3])+.02)
        ax.set_ylabel('Soil moisture (m³ m⁻³)')
        ax.set_title(f'({chr(96+layer)}) Layer {layer} · '+['0–5 cm','5–15 cm','15–30 cm'][layer-1],loc='left',pad=10)
        time_axis(ax,run)
    ax=axes[3]
    for col,key in [('actual_et_aligned','default'),('et_forcing_aligned','forcing')]:
        ax.plot(frame.index,frame[col].where(frame.paired_valid),color=COLOURS[key],lw=.65)
    for op in operators:
        if op['interval_mm'] is None:
            selected=pairs.loc[pairs.operator_id.eq(op['operator_id'])&pairs.qc.eq('clean')]
            ax.scatter(selected.date,selected.observed,s=2,c='black',rasterized=True)
    ax.set_ylabel('ET (mm day⁻¹)')
    ax.set_title('(d) ET · forcing and realised flux',loc='left',pad=10)
    time_axis(ax,run)
    if storage_ops:
        ax=axes[4]
        series=pd.read_csv(data/'additional_storage_series.csv',index_col=0,parse_dates=True,float_precision='round_trip').iloc[:,0]
        ax.plot(series.index,series,color=COLOURS['default'],lw=.65)
        for op in storage_ops:
            selected=pairs.loc[pairs.operator_id.eq(op['operator_id'])&pairs.qc.eq('clean')]
            ax.scatter(selected.date,selected.observed,s=2,c='black',rasterized=True)
        ax.set_ylabel('Profile storage (mm)')
        ax.set_title('(e) Cheverelis · 0–90 cm storage · UTC',loc='left',pad=10)
        time_axis(ax,run)
    axes[-1].set_xlabel('Year · gold: initialisation; grey: evaluation')
    basis = 'UTC' if plan['observation_operator']['source']=='ISMN' else f"Local standard time (UTC+{plan['observation_operator']['utc_offset_hours']:g})"
    note='Centre diagnostics only; station outside accepted support.' if point['diagnostic_only'] else 'Fixed station pixel; depth and temporal approximations are recorded in evaluation.'
    basis_note=basis+' daily means/totals.'+(' Cheverelis storage uses UTC.' if storage_ops else '')
    fig.text(.09,.027,basis_note+'\n'+note+'\nOnly supported matched observations are overlaid; gaps remain missing.',fontsize=FS)
    save_figure(fig,out,'daily',checks)
    supported = [op for op in operators if (pairs.operator_id.eq(op['operator_id'])&pairs.qc.eq('clean')).any()]
    for page,begin in enumerate(range(0,len(supported),2),1):
        selected_ops=supported[begin:begin+2]
        fig,axes=plt.subplots(len(selected_ops),2,figsize=(12,6 if len(selected_ops)==1 else 11),squeeze=False)
        fig.subplots_adjust(left=.09,right=.97,bottom=.12,top=.85,wspace=.34,hspace=.48)
        fig.text(.09,.96,label+' · matched observations',fontsize=TS,va='top')
        for row,op in enumerate(selected_ops):
            selected=pairs.loc[pairs.operator_id.eq(op['operator_id'])&pairs.qc.eq('clean')]
            model_columns=['default']+(['SSEBop_ET_forcing'] if op['interval_mm'] is None else [])
            limits=finite_limits([selected[['observed',*model_columns]]])
            for col,phase in enumerate(('development','evaluation')):
                ax=axes[row,col]
                sample=selected.loc[selected.phase.eq(phase)]
                ax.plot(limits,limits,color='.5',lw=.8)
                notes=[]
                for variable,colour,marker in [('default',COLOURS['default'],'o'),('SSEBop_ET_forcing',COLOURS['forcing'],'s')]:
                    if variable not in model_columns:
                        continue
                    ax.scatter(sample.observed,sample[variable],s=8,c=colour,marker=marker,alpha=.6,rasterized=True)
                    result=metrics(sample[variable],sample.observed)
                    statistics.append(dict(operator_id=op['operator_id'],qc='clean',phase=phase,model=variable,**result))
                    r='NA' if result['pearson_r'] is None else f"{result['pearson_r']:.2f}"
                    notes.append(('SWEB' if variable=='default' else 'Forcing')+f": R = {r} (N = {result['n']})")
                ax.text(.03,.97,'\n'.join(notes),transform=ax.transAxes,va='top',fontsize=FS,
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
                ax.set_xlim(limits);ax.set_ylim(limits);ax.set_aspect('equal',adjustable='box')
                unit='mm day⁻¹' if op['interval_mm'] is None else ('mm' if op['units']=='mm' else 'm³ m⁻³')
                ax.set_xlabel(f'Observed ({unit})');ax.set_ylabel(f'Model ({unit})')
                interval='ET' if op['interval_mm'] is None else f"{'Storage' if op['units']=='mm' else 'SM'} {op['interval_mm'][0]/10:g}–{op['interval_mm'][1]/10:g} cm"
                ax.set_title(f'{op["operator_id"]} · {interval}\n{phase.capitalize()}',fontsize=FS,pad=8)
        fig.text(.09,.025,'Primary QC · each sensor/deployment has its own exact pairs. Strict-QC metrics and pairs are retained in evaluation.\nOrange: SWEB; blue: ET forcing. Unknown correlation is shown as NA.',fontsize=FS)
        save_figure(fig,out,f'scatter_{page:02d}',checks)
    maps=dict(np.load(data/f'maps_{plan.get("map_year",2025)}.npz'))
    meta=json.loads((data/'map_metadata.json').read_text())
    for name,variables,titles,unit,cmap in [
        ('storage_maps',['storage'],['Storage · 3–75 cm'],'Storage (mm)','YlGnBu'),
        ('et_maps',['et_forcing','actual_et'],['SSEBop ET forcing','Realised SWEB ET'],'ET (mm day⁻¹)','YlGnBu')]:
        limits=finite_limits([maps[v] for v in variables])
        limits[0]=max(0,limits[0])
        fig,axes=plt.subplots(len(variables),4,figsize=(12,4*len(variables)+1.3),squeeze=False)
        fig.subplots_adjust(left=.08,right=.98,bottom=.24,top=.81,wspace=.15,hspace=.55)
        fig.text(.08,.975,label+f' · {meta["year"]}',fontsize=TS,va='top')
        for row,(variable,title) in enumerate(zip(variables,titles)):
            for col,month in enumerate(('January','April','July','October')):
                ax=axes[row,col]
                colour=plt.get_cmap(cmap).copy();colour.set_bad('#E5E5E5')
                im=ax.imshow(maps[variable][col],origin='upper',extent=meta['extent_km'],vmin=limits[0],vmax=limits[1],cmap=colour)
                if meta['extent_km'][0] <= 0 <= meta['extent_km'][1] and meta['extent_km'][2] <= 0 <= meta['extent_km'][3]:
                    ax.plot(0,0,marker='+',color='black',ms=9,mew=3)
                    ax.plot(0,0,marker='+',color='white',ms=8,mew=1.5)
                ax.set_title(month,fontsize=FS,pad=6)
                if col:ax.set_yticklabels([])
                else:ax.set_ylabel('North offset (km)')
                if row==len(variables)-1:ax.set_xlabel('East offset (km)')
                else:ax.set_xticklabels([])
            pos=axes[row,0].get_position()
            fig.text(pos.x0,pos.y1+.065,title,fontsize=TS)
        cax=fig.add_axes([.22,.13,.58,.025])
        fig.colorbar(im,cax=cax,orientation='horizontal',label='Monthly mean '+unit)
        fig.text(.5,.025,'Native 30 m grid · complete-month history support · grey = missing · + = fixed station\nCoordinates are offsets from the true station location; forcing and realised ET share one scale.',ha='center',fontsize=FS)
        save_figure(fig,out,name,checks)
    captured=[]
    original_close=plt.close
    try:
        plt.close=lambda f:captured.append(f)
        plot_heatmap.main(['--sweb-path',str(data/'heatmap_input.nc'),'--domain-mean','--start-date',run['start'],
            '--end-date',run['end'],'--cmap','esa_soil','--vmin','0','--vmax',str(max(.5,max(hydraulic['porosity']))),
            '--figsize','14','6','--output',str(out/'inspection/heatmap_upstream.png'),
            '--csv-out',str(out/'heatmap_exact_data.csv'),'--title',label])
    finally:
        plt.close=original_close
    fig=captured[-1];ax,cax=fig.axes[:2]
    for t in list(ax.texts):t.remove()
    for t in fig.findobj(Text):t.set_fontsize(FS);t.set_fontfamily(font)
    fig._suptitle.set_text(label+' · five-layer trajectory');fig._suptitle.set_fontsize(TS);fig._suptitle.set_y(.96)
    ax.set_position([.14,.32,.83,.49]);cax.set_axes_locator(None);cax.set_box_aspect(None);cax.set_aspect('auto');cax.set_position([.24,.17,.64,.035])
    ax.set_ylabel('Soil depth (mm)');ax.set_xlabel('Year')
    years=range(pd.Timestamp(run['start']).year,pd.Timestamp(run['end']).year+1,2)
    ax.set_xticks([int(np.flatnonzero(frame.index==pd.Timestamp(f'{y}-01-01'))[0])+.5 for y in years],list(map(str,years)),rotation=0,ha='center')
    for date in (run['analysis_start'],run['holdout_start']):ax.axvline(int(np.flatnonzero(frame.index==pd.Timestamp(date))[0]),color='black',lw=1)
    ax.collections[0].set_rasterized(True);ax.collections[0].cmap.set_bad('#E5E5E5')
    cax.set_xlabel('Start-of-day soil moisture (m³ m⁻³)')
    fig.text(.5,.025,'Repository heatmap function · exact native pixel · physical layer thicknesses · grey = missing\nVertical lines mark the end of initialisation and the start of evaluation.',ha='center',fontsize=FS)
    plotted=pd.read_csv(out/'heatmap_exact_data.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    for layer in range(1,6):np.testing.assert_allclose(plotted[f'sweb_rzsm_layer_{layer}'],frame[f'start_layer_{layer}'].where(frame.continuous_trajectory_valid),atol=1e-14,rtol=0,equal_nan=True)
    save_figure(fig,out,'heatmap',checks)
    atomic_json(out/'figure_contract.json',dict(font=font,ordinary_font_pt=FS,title_font_pt=TS,dpi=DPI,colours=COLOURS,
        pages=checks,missing='NaNs preserved; unavailable observation panels omitted with evaluation coverage records',
        point_operator=point,visual_review='pending; automatic rendering is not visual acceptance'))
    atomic_json(out/'scatter_statistics.json',statistics)
    atomic_json(out/'figure_validation.json',dict(passed=True,pages=len(checks),heatmap_exact_data=True,visual_review_passed=False))
