# Explicit scan-gap reconstruction

`pysweb.ssebop.gapfill.local_linear_gapfill` provides adaptive local matching
for a floating-point primary band, ordered donor bands and an explicit
eligible-gap mask. It follows the radiometric matching in the public-domain
[USGS Landsat 7 Phase Two methodology](https://www.usgs.gov/media/files/landsat-7-slc-gap-filled-products-phase-two-methodology).

The smallest odd window up to 31 pixels requires 144 common observed samples.
OLS gains outside [1/3, 3] trigger standard-deviation matching, followed by a
mean-offset fallback. The result retains original finite values, residual
missing values, per-pixel donor indices, support counts, window sizes and
estimator flags. Donor indices are 0 for missing, 1 for original and 2 onward
for supplied donors; method flags are 1 for OLS, 2 for standard-deviation
matching and 3 for mean-offset matching.

This implementation requires sufficient support even at the largest window
and fits every donor against original target observations. It operates in
supplied units without historical 8-bit clipping. These are explicit local
adaptations. The function does not select dates, identify scan gaps, apply QA,
register grids, harmonise sensors, or establish thermal/field accuracy.
Callers must verify those contracts, restrict temporal donors to allowed
periods, retain provenance and validate estimates on withheld observations.
Cloud masks and persistent temperature-only holes should not be mistaken for
scan-gap eligibility. Suggested physical bounds reject implausible predictions
without clipping them into apparently valid values.

The module is an explicit preprocessing option. No automatic L7 filling is
enabled in the standard SSEBop API. The two-site pilot requires a separate
field-scene validation gate before its filled products enter retrieval.
