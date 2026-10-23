# Site workflows

The maintained package owns the scientific implementation. Existing numbered adapters and `pysweb workflow` remain available. The site workflow adds independently restartable acquisition, retrieval, preprocessing, simulation, evaluation, figures and finalisation around those APIs. This is a maintained-software profile with an external scientific reproduction layer.

These site workflows are **runnable with external data** on POSIX hosts. They require a scientific TOML and a JSON site plan; no dataset, institution path, account or scheduler is embedded in scientific code. Local execution uses the same stage functions as Gadi. Calibration remains available through the existing general workflow, while this composition explicitly requires an uncalibrated actual-prefix trajectory.

```bash
python workflows/run_site.py --plan /path/to/site.json --dry-run
bash workflows/local/run_site.sh --plan /path/to/site.json --stages acquire
bash workflows/local/run_site.sh --plan /path/to/site.json --stages compute
# Individual steps use the same plan and receipts:
python workflows/run_site.py --plan /path/to/site.json --stages retrieve,preprocess
python workflows/evaluate_site.py --plan /path/to/site.json
python workflows/finalise_site.py --plan /path/to/site.json
```

Set `SWEB_PYTHON` to an existing compatible interpreter. `compute` calls retrieval → preprocessing → simulation → evaluation → figures → finalisation. Acquisition supplies local native soil predictors, meteorology, physical references, immutable Landsat scenes and tiled WorldCover. Compute stages reject network fallback. Reusing a pilot requires an explicit compatibility configuration and a passing parity receipt for the current implementation.

A site plan supplies `config`, `destination`, `retained_soil`, `scientific_baseline`, `implementation_commit`, `run_id`, `development_end`, `observation_file`, its expected `observation_sha256`, `observation_manifest`, its expected hash, and `observation_operator`. The operator supplies source (`FLUXNET` or `ISMN`), fixed location and standard UTC offset in hours. Paths in these plan fields resolve relative to the JSON file. The TOML keeps scientific settings and data paths; `run.root/run.name` is the working directory. Dataset destination must differ from that directory. `scene_workers` defaults to two; model concurrency follows `run.workers`.

`run.analysis_start` explicitly separates actual-prefix initialisation from scoring even when calibration is disabled. For example, a 2014–2025 simulation with `analysis_start="2016-01-01"` and `holdout_start="2021-01-01"` retains 730 initialisation days, excludes them from scores and continues without resetting state at the evaluation boundary. Omitting the setting preserves the previous general-workflow defaults. Repeated spin-up and calibration settings cannot conflict with an explicit analysis start.

Stage receipts bind configuration, input contents, source identity, dependencies and output checksums. Kernel locks prevent concurrent site writers; failed attempts remain recorded. A retry reuses only receipts whose outputs still verify. Final dataset files are copied through temporary files and atomically promoted; previous runs remain external immutable evidence. A finalised site remains `numerically_validated_pending_visual_review` until its exported figure pages have been inspected and a separate review recorded. Missing observational support is recorded with reasons and zero sample counts.

## Gadi adapter

```bash
python workflows/hpc/gadi/submit_sites.py --execution /path/to/execution.json --sites SITE_A SITE_B
# Inspect the generated submission_plan.json and PBS files, then execute when authorised:
python workflows/hpc/gadi/submit_sites.py --execution /path/to/execution.json --sites SITE_A SITE_B --submit
```

The default is a dry run. Execution JSON specifies the frozen `cohort`, `site_plans` directory, `operations`, `logs`, `ledger`, project, storage mounts, tested interpreter, modules, Gadi resource helper, frozen source archive and SHA-256, implementation commit and separate acquisition/compute resource requests. Requests contain queue, ncpus, mem_gib, jobfs_gib and walltime. The adapter creates site-specific PBS scripts with explicit resources and verified allocations. It checks active jobs and serialises ledger writes. A pending submission intent requires reconciliation before retry to prevent duplicate jobs after an interrupted qsub transaction.

Each compute job has at most its own acquisition prerequisite. There are no cross-site chains or cohort-wide barriers. Select sites explicitly from the frozen cohort; dataset directory discovery never expands scope. The coordinator should limit acquisition concurrency to the project and service allowances, examine initial telemetry, and update resource requests before broad dispatch.

## Scientific and observational limits

Refactoring does not change retrieval thresholds, hydraulics, meteorology, reconstruction, or the solver. The finite FANO cold-reference/clipping limitation remains; no ET floor or rainfall correction is added. Static WorldCover represents one epoch. ET forcing and realised ET are labelled separately. Evaluation uses exact exported finite pairs for each variable, sensor/deployment, QC and phase. Point probes are approximated by containing layers; depth intervals use overlap weights. Unknown depths and footprint measurements without an appropriate operator are excluded. Positive-offset local-day alignment approximates unresolved subdaily behaviour; the current operator supports UTC and offsets from zero to less than 24 hours. Additional global regions require their own explicit time/operator configurations.
