# Repeated calendar-block initialization

Repeated spin-up is disabled by default. Set `spinup.enabled = true` to use a
complete block at the beginning of the forcing record for repeated
initialisation. For example:

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
enabled = true
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

To disable repetition while retaining the other settings for later use, set:

```toml
[spinup]
enabled = false
```

An omitted table, an empty table or an omitted `enabled` key also disables
repetition. `enabled` must be a boolean. Disabled settings do not require source
years, complete calendar blocks or convergence diagnostics. The ordinary run
starts at physical field capacity and processes its actual dates once. Explicit
calibration dates and any actual-forcing prefix remain in effect; the switch
does not remove ordinary initialisation or rewrite the requested date range.

Configurations created before the explicit switch need `enabled = true` to
reproduce their repeated-year experiment with this version. Preserve their
original source snapshot when reproducing an archived accepted run.

`source_years` must be consecutive integer years forming a complete initial daily
forcing block. Leap years retain every day. Calibration must begin after the
block. The native workflow requires a development boundary before the holdout.
The existing single-year `source_year` interface retains its earlier semantics.
Do not specify both interfaces.

The reference and Numba solvers share this implementation. Their public
`soil_water_balance_1d` functions accept
`spinup={"enabled": True, "source_years": [2014, 2015], "repeats": 10}`.
Passing `None`, an empty mapping or `{"enabled": False}` disables repetition;
omitting `enabled` in a supplied mapping also leaves it disabled.
Gridded run and calibration
APIs accept `spinup_source_years` and `spinup_repeats`; the command-line flags are
`--spinup-source-years 2014 2015 --spinup-repeats 10`. Direct calibration also
requires `--spinup-days 730` to exclude the retained block.
Supplying `--spinup-source-years` explicitly enables repetition in these
component APIs; omit the two repetition flags for the default disabled path.
`--spinup-days` alone excludes an actual forcing prefix from calibration and
does not repeat any forcing.

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
