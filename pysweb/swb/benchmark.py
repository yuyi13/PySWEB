"""
Script: benchmark.py
Objective: Measure complete coarse objectives and short populations before resource planning.
Author: Yi Yu (with assistance from Codex)
Created: 2026-10-17
Last updated: 2026-10-17
Inputs: Explicit objective arrays, fixed observation support and optimizer bounds.
Outputs: Measured objective/population times and qualified calibration projections.
Usage: from pysweb.swb.benchmark import benchmark_objective
Dependencies: numpy, scipy, pysweb; caller-provided compute allocation.
"""
import time
import numpy as np
from pysweb.swb.calibrate import _compute_rmse, _objective_function
from pysweb.swb.calibration_runtime import shared_differential_evolution


def benchmark_objective(arguments, context, bounds, *, workers, parameters,
                        max_generations=100, population_multiplier=10, seed=0,
                        planning_margin=1.5, data_status="actual prepared calibration grid"):
    """Measure warmed full objectives and two complete population evaluations."""
    if workers < 1 or planning_margin < 1 or max_generations < 1:
        raise ValueError("Invalid benchmark resource/planning options")
    warm_start = time.perf_counter()
    rmse, count = _compute_rmse(parameters, *arguments, context)
    warm_seconds = time.perf_counter()-warm_start
    if not np.isfinite(rmse) or count <= 0:
        raise ValueError("Benchmark has no valid objective support")
    times = []
    for _ in range(3):
        start = time.perf_counter()
        repeated, pairs = _compute_rmse(parameters, *arguments, context)
        times.append(time.perf_counter()-start)
        if repeated != rmse or pairs != count:
            raise ValueError("Repeated objective changed its value or support")
    start = time.perf_counter()
    result = shared_differential_evolution(_objective_function, bounds,
        args=(*arguments, context), workers=workers, maxiter=1, popsize=population_multiplier,
        seed=seed, polish=False, updating='deferred', tol=0, atol=0,
        mutation=(.5, 1.), recombination=.7, strategy='best1bin')
    elapsed = time.perf_counter()-start
    population = max(5, population_multiplier*sum(low != high for low,high in bounds))
    evaluations = int(result.nfev)
    population_seconds = elapsed*population/evaluations
    projected = population_seconds*(max_generations+1)
    return dict(data_status=data_status, workers=workers, cells=int(np.asarray(arguments[6]).sum()),
        simulated_days=len(arguments[4]), observation_cell_days=count, baseline_rmse=rmse,
        compile_and_first_objective_seconds=warm_seconds, warmed_objective_seconds=times,
        measured_search_seconds=elapsed, measured_evaluations=evaluations,
        population_members=population, population_seconds=population_seconds,
        planned_max_generations=max_generations, projected_search_seconds=projected,
        planning_margin=planning_margin, conservative_search_seconds=projected*planning_margin,
        excluded_costs=['queue time','input acquisition/preparation','serial polishing','fine-grid simulation','acceptance checks'],
        limitation='Projection is conditional on observed throughput and the generation limit; convergence is not guaranteed')
