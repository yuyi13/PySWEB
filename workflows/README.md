# Site workflows

The maintained package owns the scientific implementation. Existing numbered adapters and `pysweb workflow` remain available. The site workflow adds independently restartable acquisition, retrieval, preprocessing, physical references, simulation, evaluation, figures and finalisation around those APIs. This is a maintained-software profile with an external scientific reproduction layer.

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

Set `SWEB_PYTHON` to an existing compatible interpreter. `compute` calls retrieval → preprocessing → physical-reference verification/derivation → simulation → evaluation → figures → finalisation. Acquisition supplies local native soil predictors, meteorology, physical references, immutable Landsat scenes and tiled WorldCover. Compute stages reject network fallback. Reusing a pilot requires an explicit compatibility configuration and a passing parity receipt for the current implementation.

A site plan supplies `config`, `destination`, `retained_soil`, `scientific_baseline`, `implementation_commit`, `run_id`, `development_end`, `observation_file`, its expected `observation_sha256`, `observation_manifest`, its expected hash, and `observation_operator`. The operator supplies source (`FLUXNET` or `ISMN`), fixed location and signed standard UTC offset in hours. Paths in these plan fields resolve relative to the JSON file. The TOML keeps scientific settings and data paths; `run.root/run.name` is the working directory. Dataset destination must differ from that directory. `scene_workers` defaults to two; model concurrency follows `run.workers`.

Set `acquire_missing_sources=true` for a new site without retained inputs. Its own acquisition invocation freezes the Landsat 8/9 L2SP metadata query at the configured centre/WRS, then calls the established native source APIs for meteorology, NASADEM, GSSM and OpenLandMap. The catalogue starts at `catalogue_start` and ends before 2026-01-01 for this 2014–2025 protocol. The first catalogue scene supplies the grid template. Source receipts verify checksums before reusing completed acquisition substeps after failure. Acquired source files become outputs of the acquisition receipt, so offline stages verify them through the prerequisite chain. A temporary directory (controlled by `TMPDIR` locally and jobfs on Gadi) contains the daily meteorology expansion. The default retained-input workflow is unchanged.

`run.analysis_start` explicitly separates actual-prefix initialisation from scoring even when calibration is disabled. For example, a 2014–2025 simulation with `analysis_start="2016-01-01"` and `holdout_start="2021-01-01"` retains 730 initialisation days, excludes them from scores and continues without resetting state at the evaluation boundary. Omitting the setting preserves the previous general-workflow defaults. Repeated spin-up and calibration settings cannot conflict with an explicit analysis start.

Stage receipts bind configuration, input contents, source identity, dependencies and output checksums. Kernel locks prevent concurrent site writers; failed attempts remain recorded. A retry reuses only receipts whose outputs still verify. Final dataset files are copied through temporary files and atomically promoted; previous runs remain external immutable evidence. A finalised site remains `numerically_validated_pending_visual_review` until its exported figure pages have been inspected and a separate review recorded. Missing observational support is recorded with reasons and zero sample counts.

An individual downstream stage verifies its complete prerequisite chain, including external inputs, observation operators and relevant evaluation/figure code. A stale ancestor must be resumed first. Use `compute` for an ordinary compute restart, or `all` when acquisition inputs changed. Files are verified once per invocation under the site lock. Plot-only changes retain valid numerical receipts. Changing `label` reruns figures and finalisation; changing `map_year` reruns evaluation and its downstream stages. A changed destination or exported plan/configuration metadata reruns finalisation. These options do not invalidate unchanged numerical stages.

## Gadi adapter

```bash
python workflows/hpc/gadi/submit_sites.py --execution /path/to/execution.json --sites SITE_A SITE_B
# Inspect the generated submission_plan.json and PBS files, then execute when authorised:
python workflows/hpc/gadi/submit_sites.py --execution /path/to/execution.json --sites SITE_A SITE_B --submit
```

The default is a dry run. Execution JSON specifies the frozen `cohort`, `site_plans` directory, `operations`, `logs`, `ledger`, project, storage mounts, tested interpreter, modules, Gadi resource helper, frozen source archive and SHA-256, implementation commit and separate acquisition/compute resource requests. Requests contain queue, ncpus, mem_gib, jobfs_gib and walltime. The adapter creates site-specific PBS scripts with explicit resources and verified allocations. It checks active jobs and serialises ledger writes. A pending submission intent requires reconciliation before retry to prevent duplicate jobs after an interrupted qsub transaction.

Each compute job has at most its own acquisition prerequisite. There are no cross-site chains or cohort-wide barriers. Select sites explicitly from the frozen cohort; dataset directory discovery never expands scope. The site plan's implementation commit must match its execution archive configuration. Keep earlier execution configurations when a run contains more than one validated implementation. The coordinator should limit acquisition concurrency to the project and service allowances, examine initial telemetry, and update resource requests before broad dispatch.

At startup, the adapter verifies the small source archive and extracts its allocation checker before scientific execution. Scheduler lookup timeouts are retried up to four times, with ten seconds between attempts. Every report is retained in the site logs. Scientific stages start only after the checker verifies the allocated CPUs, memory, jobfs and required storage mounts; other verification failures stop immediately.

## Scientific and observational limits

Refactoring does not change retrieval thresholds, hydraulics, meteorology, reconstruction, or the solver. The finite FANO cold-reference/clipping limitation remains; no ET floor or rainfall correction is added. Static WorldCover represents one epoch. ET forcing and realised ET are labelled separately. Evaluation uses exact exported finite pairs for each variable, sensor/deployment, QC and phase. Point probes are approximated by containing layers; depth intervals use overlap weights. Unknown depths and footprint measurements without an appropriate operator are excluded. Signed-offset local-day alignment approximates unresolved subdaily behaviour. Offsets must be finite and strictly between -24 and 24 hours; use documented station standard time with daylight saving excluded. Negative offsets integrate the current and following UTC day; positive offsets retain the established previous/current-day arithmetic. Required neighbouring dates must be valid, and an unavailable boundary remains missing. Scoring phases refer to the observation date label. Additional regions require their own explicit time/operator configurations.

Missing physical references can be derived with `derive_physical_reference=true` using the baseline recipe: complete-development native GSSM support and native hydraulic predictors, temporal 2011–2020 p01/p99 per cell followed by area means. The preprocessing-only reference grid is generated without calibration or parameter fitting. Existing reference artifacts remain fixed. Derived artifacts are frozen before simulation and retained with source hashes.

An explicit `reference_coverage_policy` in the site JSON can relax the all-cells requirement for a new, versioned reference derivation:

```json
"reference_coverage_policy": {
  "minimum_valid_fraction": 0.99,
  "minimum_area_fraction": 0.50
}
```

This example retains cells with at least 99% valid days over the unchanged 2011–2020 calendar (3,653 days), excludes deficient cells without filling them, and requires at least 50% of the original candidate spherical area to remain. GSSM quantiles and native hydraulic means use exactly the same retained footprint. Empty support always fails. The `gssm-native-p01-p99-support-v2` artifact records every candidate cell, valid-day count, exclusion and retained-area fraction. Existing strict v1 artifacts remain readable, and omitting the policy preserves the previous strict derivation byte for byte. This is an explicit scientific support-selection change, not calibration or a solver change. Freeze a new artifact and preserve the original run records; policy changes invalidate affected workflow receipts. It does not establish that the retained subset represents excluded land.

Yanco can explicitly include its documented Cheverelis storage comparison through `observation_operator.additional_storage`, supplying the accepted `contract`, `contract_sha256` and processed `observations` CSV. The contract and its native observation inputs are checksum-verified. The 0–90 cm profile is evaluated at the separate Cheverelis pixel in UTC, with primary/strict QC and exact pairs; it is not inherited by other sites.
