#!/usr/bin/env python3
"""
Script: submit_sites.py
Objective: Plan or submit independent site jobs with only same-site acquisition prerequisites.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-23
Last updated: 2026-10-23
Inputs: Frozen cohort, site plans, execution JSON and a coordinator-owned submission ledger.
Outputs: Explicit PBS scripts, dry-run plans and append-preserving submission records.
Usage: python workflows/hpc/gadi/submit_sites.py --execution execution.json --sites SITE [--submit]
Dependencies: Python standard library, POSIX locks; PBS commands only for submission.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import tempfile


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):
            h.update(block)
    return h.hexdigest()


def atomic(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');temporary=Path(f.name)
    temporary.replace(path)


def pbs_script(execution, plan_path, stage):
    plan=json.loads(Path(plan_path).read_text())
    resources=execution['resources']['acquire' if stage=='acquire' else 'compute']
    mounts=sorted(set(execution['storage'])|set(plan.get('storage',[])))
    quote=shlex.quote
    name='sweb_'+plan['site_key'][-28:]+'_'+stage
    module_line='module load '+' '.join(quote(v) for v in execution['modules'])
    return f'''#!/usr/bin/env bash
# Script: {plan['site_key']}_{stage}.pbs
# Objective: Run one independently restartable site {'acquisition' if stage=='acquire' else 'offline computation'}.
# Author: Yi Yu (with assistance from Codex)
# Created: 2026-10-23
# Last updated: 2026-10-23
# Inputs: Frozen software archive, configured site plan and verified prerequisite receipts.
# Outputs: Site products, stage receipts and resource telemetry on configured storage.
# Usage: qsub this-script.pbs
# Requirements: Gadi PBS allocation and the configured runtime/modules.
#PBS -P {execution['project']}
#PBS -q {resources['queue']}
#PBS -N {name}
#PBS -l ncpus={resources['ncpus']},mem={resources['mem_gib']}GB,jobfs={resources['jobfs_gib']}GB,walltime={resources['walltime']}
#PBS -l storage={'+'.join(mounts)}
#PBS -j oe
#PBS -o {execution['logs']}
set -euo pipefail
{module_line}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PBS_JOBFS" NUMBA_CACHE_DIR="$PBS_JOBFS/numba"
SWEB_PYTHON={quote(execution['python'])}
SWEB_PLAN={quote(str(plan_path))}
SWEB_SITE_LOGS={quote(str(Path(plan['destination'])/'logs'))}
mkdir -p "$SWEB_SITE_LOGS"
export SWEB_RESOURCE_FILE="$SWEB_SITE_LOGS/resources_${{PBS_JOBID}}.json"
SWEB_SOURCE={quote(execution['source_archive'])}
"$SWEB_PYTHON" -c 'import hashlib,sys; assert hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest() == sys.argv[2]' "$SWEB_SOURCE" {quote(execution['source_sha256'])}
mkdir "$PBS_JOBFS/source"
tar -xzf "$SWEB_SOURCE" -C "$PBS_JOBFS/source"
export PYTHONPATH="$PBS_JOBFS/source"
cd "$PBS_JOBFS/source"
"$SWEB_PYTHON" workflows/hpc/gadi/verify_allocation.py --helper {quote(execution['resource_helper'])} --output "$SWEB_RESOURCE_FILE" --ncpus {resources['ncpus']} --memory-gib {resources['mem_gib']} --jobfs-gib {resources['jobfs_gib']} --storage {' '.join(quote(mount) for mount in mounts)}
"$SWEB_PYTHON" workflows/run_site.py --plan "$SWEB_PLAN" --stages {stage} 2>&1 | tee "$SWEB_SITE_LOGS/${{PBS_JOBID}}.stages.log"
'''


def job_state(job_id):
    result=subprocess.run(['qstat','-xf','-F','json',job_id],capture_output=True,text=True)
    if result.returncode:
        if 'Unknown Job Id' in result.stderr:
            return None
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)['Jobs'][job_id]


def submit(execution, sites, execute=False):
    cohort=json.loads(Path(execution['cohort']).read_text())['sites']
    selected={s['name']:s for s in cohort}
    if set(sites)-set(selected):
        raise ValueError('A requested site is outside the frozen cohort')
    if len(set(sites))!=len(sites):
        raise ValueError('Duplicate site selection')
    output=Path(execution['operations'])
    output.mkdir(parents=True,exist_ok=True)
    plan_records=[]
    for site in sites:
        path=Path(execution['site_plans'])/(site+'.json')
        plan=json.loads(path.read_text())
        if plan['site_key']!=site or plan['region']!=selected[site]['region']:
            raise ValueError('Site plan identity differs from frozen cohort')
        for stage in (['compute'] if plan.get('reuse_pilot') else ['acquire','compute']):
            script=output/f'{site}_{stage}.pbs'
            script.write_text(pbs_script(execution,path,stage))
            subprocess.run(['bash','-n',str(script)],check=True)
            plan_records.append(dict(site=site,stage=stage,script=str(script),script_sha256=digest(script),
                config_sha256=digest(plan['config']),plan_sha256=digest(path),
                source_sha256=execution['source_sha256'],implementation_commit=execution['implementation_commit'],
                scientific_source_identity=execution.get('scientific_source_identity'),
                resource_request=execution['resources'][stage],dependency_policy='same-site acquire only' if stage=='compute' else 'none'))
    atomic(output/'submission_plan.json',dict(submit=execute,sites=sites,jobs=plan_records))
    if not execute:
        return plan_records
    ledger_path=Path(execution['ledger'])
    with ledger_path.with_suffix('.lock').open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=json.loads(ledger_path.read_text())
        for record in plan_records:
            prior=[r for r in ledger['submissions'] if r.get('site')==record['site'] and r['stage']==record['stage']]
            active=[]
            for r in prior:
                state=job_state(r['job_id'])
                if state is not None:
                    r['scheduler']={key:state[key] for key in ('job_state','Exit_status','resources_used','Resource_List','stime','mtime','obittime','comment') if key in state}
                    if state.get('job_state') not in ('F','X'):
                        active.append(r)
            if active:
                print('Already active:',record['site'],record['stage'],active[-1]['job_id'],flush=True)
                continue
            if record['stage']=='acquire' and prior:
                previous=prior[-1]
                state=previous.get('scheduler',{})
                if (state.get('job_state')=='F' and state.get('Exit_status')==0
                        and previous.get('config_sha256')==record['config_sha256']
                        and previous.get('scientific_source_identity')
                        and previous['scientific_source_identity']==record['scientific_source_identity']):
                    print('Reuse successful acquisition:',record['site'],previous['job_id'],flush=True)
                    continue
            plan=json.loads((Path(execution['site_plans'])/(record['site']+'.json')).read_text())
            accepted=Path(plan['destination'])/'metadata.json'
            if accepted.exists() and json.loads(accepted.read_text()).get('status')=='accepted':
                print('Already accepted:',record['site'],flush=True)
                continue
            dependencies=[]
            if record['stage']=='compute' and not plan.get('reuse_pilot'):
                acquisitions=[r for r in ledger['submissions'] if r.get('site')==record['site'] and r['stage']=='acquire']
                if not acquisitions:
                    raise ValueError('No same-site acquisition submission')
                predecessor=acquisitions[-1]
                state=job_state(predecessor['job_id'])
                if state is not None and state.get('job_state') in ('F','X'):
                    if state.get('Exit_status')!=0:
                        raise ValueError('Same-site acquisition failed; inspect before retry')
                else:
                    dependencies=[predecessor['job_id']]
            command=['qsub']+(['-W','depend=afterok:'+dependencies[0]] if dependencies else [])+[record['script']]
            # Persist intent before qsub; an interrupted transaction must be reconciled.
            intent={**record,'state':'submission_pending','command':command,'dependencies':dependencies,
                    'submitted_at':datetime.now(timezone.utc).isoformat(),'attempt':len(prior)+1}
            pending=output/f'{record["site"]}_{record["stage"]}_submission_pending.json'
            if pending.exists():
                raise RuntimeError(f'Unreconciled submission intent: {pending}')
            atomic(pending,intent)
            job=subprocess.check_output(command,text=True).strip()
            intent.update(job_id=job,state='submitted')
            ledger['submissions'].append(intent);ledger['status']='submitted'
            atomic(ledger_path,ledger)
            pending.unlink()
            print(record['site'],record['stage'],job,'dependencies',dependencies,flush=True)
        atomic(ledger_path,ledger)
    return plan_records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution',required=True)
    p.add_argument('--sites',nargs='+',required=True)
    p.add_argument('--submit',action='store_true')
    args=p.parse_args()
    execution=json.loads(Path(args.execution).read_text())
    submit(execution,args.sites,args.submit)


if __name__=='__main__':
    main()
