# Optional calibration acceleration

The default reference solver remains available. `solver_backend="numba"`
selects the optional `speed` extra, with identical default equations and explicit
fallback to the reference implementation for unsupported process options.
The regression suite compares every layer, budget term, final state and forcing
validity over wet, dry, missing and extreme fixtures. No fast-math is enabled.

Calibration caches the fixed observation support and weighted reference means
once. Missing reference dates stay missing; model trajectories remain continuous.
`shared_workers=True` uses local fork processes to share the immutable arrays;
request all workers and memory from the scheduler on a single host. It requires
a fork-capable platform. The default worker implementation is unchanged.

`no_polish=True` saves the converged differential-evolution solution without
holding a parallel allocation during serial polishing. A separately recorded
serial optimisation can refine that candidate. This option does not waive
convergence requirements. Record both stages and their resource use.

The gridded runner also accepts `solver_backend="numba"` (CLI
`--solver-backend numba`) and records the selected backend in its output.
Serial and parallel accelerated demo trajectories, final states and budgets
are checked against the reference runner. Completed row futures are released
as results are copied into the output arrays, limiting retained worker results
for long records. The runner still holds full time/grid output arrays; memory
must be measured and allocated before long simulations.
