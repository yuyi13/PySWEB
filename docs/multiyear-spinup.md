# Repeated calendar-block initialization

Use a complete block at the beginning of the forcing record to initialize a
daily SWB run. For example:

```toml
[run]
start = "2014-01-01"
end = "2025-12-31"
holdout_start = "2021-01-01"
state_timing = "start"

[calibration]
enabled = true
start = "2016-01-01"
end = "2020-12-31"

[spinup]
source_years = [2014, 2015]
repeats = 10
tolerance = 0.005
minimum_fraction = 0.99
convergence_mode = "wet_dry"
convergence_action = "require"
```

This example integrates the 730-day block ten times (twenty simulated years).
The count includes the retained 2014–2015 pass. The solver first integrates nine
extra passes, discarding their trajectories, then continues through the actual
2014–2025 record without resetting the soil state. Output dates remain unchanged.
All 2014–2015 outputs are initialization and are excluded from fitting and
evaluation. Calibration observations are never repeated.

`source_years` must be consecutive integer years forming a complete initial daily
forcing block. Leap years retain every day. Calibration must begin after the
block. The native workflow requires a development boundary before the holdout.
The existing single-year `source_year` interface retains its earlier semantics.
Do not specify both interfaces.

The reference and Numba solvers share this implementation. Their public
`soil_water_balance_1d` functions accept
`spinup={"source_years": [2014, 2015], "repeats": 10}`. Gridded run and calibration
APIs accept `spinup_source_years` and `spinup_repeats`; the command-line flags are
`--spinup-source-years 2014 2015 --spinup-repeats 10`. Direct calibration also
requires `--spinup-days 730` to exclude the retained block.

Only one column's current cycle is allocated during initialization. Neither
forcing grids nor spatial warm-up outputs are expanded with repetition count.
Runtime still increases with the number of simulated days.

## Interpreting convergence

`wet_dry` compares empty and saturated initial states at the scoring boundary.
`field_capacity_cycles` compares field-capacity starts with N and 2N total passes;
passing it does not establish independence from other initial states. Each test
requires every layer to meet the tolerance in the requested fraction of admitted
cells. Default and newly calibrated parameters are checked separately.

The default `convergence_action = "require"` stops on a failed diagnostic.
An explicitly labelled fixed-duration research experiment may use `"record"`.
This retains a failed receipt, emits a warning and records the interpretation in
product initialization metadata. It does **not** turn the diagnostic into a pass
or establish readiness for regional inference. Numerical conservation and
calibration reproduction remain mandatory in either mode.

Repeated forcing can reduce sensitivity to assumed initial water storage. It
does not correct bias in forcing, hydraulic properties, rooting assumptions or
the observation operator, and a longer initialization need not converge for
every parameter setting.
