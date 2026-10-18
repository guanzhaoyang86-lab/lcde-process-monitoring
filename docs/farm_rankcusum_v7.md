# FARM-RankCUSUM v7: frozen method specification

## Status and scope

`farm-rankcusum-v7-fixed` is the single v7 development candidate.  It is a
genuinely different successor to the terminally aborted PCF-CycleRank v6
candidate: v7 is a deterministic, healthy-only empirical-CDF monitoring
method and contains no learned neural network, residual classifier, fault
supervision, or candidate grid.

Development uses only the already-consumed Extended Tennessee Eastman Mode2
magnitude-50 pool.  Its result is adaptive exploratory feasibility evidence,
not an unbiased confirmation result.  Extended TEP Mode1 magnitude 25 and 75
remain sealed.  No v7 development operation authorizes a Mode1 HDF5 open.

## Primary method provenance and project contribution

The monitoring core follows the eCDF model in Miraliakbar et al., *An online
fault detection, diagnosis, and isolation framework for industrial
processes*, Computers & Chemical Engineering 198 (2025) 109177,
<https://doi.org/10.1016/j.compchemeng.2025.109177>.  The authors' public
implementation is <https://github.com/checlams/FARM_paper>.

The upstream paper supplies the Bayesian empirical-CDF estimate, two-sided
CUSUM, and top-r aggregation, and reports strong TEP results for Faults 3, 9,
and 15.  Those published rates are not treated as guarantees here because the
simulation counts, monitoring horizon, and false-alarm protocol differ.  The
project-level contribution is the fixed combination with collision-safe
run-wise cross-validation, run-maximum calibration, strict no-fault-use
fitting, CUDA replay auditing, and a prospectively held-out severity-cell
confirmation protocol.

## Frozen data contract

- Source: `data/extended_tep_dtu_mode2/TEP_Mode2.h5`, exact source identity
  bound by the preregistration.
- Regime: Mode2, `SingleFault`, `SimulationCompleted`, magnitude 50.
- Conditions: IDV1 as the project healthy-control condition and IDV3, IDV9,
  and IDV15 as the three target faults.
- Runs: simulation run IDs 1 through 100 for every condition.
- Healthy local window: source rows `[100, 600)`.
- Fault local window: source rows `[440, 940)`; local index 160 is the frozen
  fault onset.
- Every selected run has exactly 500 rows and 52 process channels after the
  source time column is removed.  Values must be finite float64.
- The monitor consumes local endpoints 160 through 499 inclusive.  A missed
  alarm receives delay 340.

The exact 400 selectors, not a wildcard, are committed in the access intent.
Only the 60 fit-role healthy runs may fit a fold reference.  Only the 20
inner-role healthy runs may set its threshold.  No fault value is permitted
to fit either object.

For each selector, the only numeric projections are
`processdata[start:stop, 0:53]` and `idv_init[0:3, 0:28]`.  The only other
source reads are the group `seed` attribute and object type, shape, and dtype
metadata.  Root-label values, rows outside the selected 500-row window, and
all other Mode2 branches are forbidden.

## Frozen five-fold roles

The run-ID folds are inherited unchanged from the collision-safe v5.1/v6
assignment.  For outer fold `f`, inner fold is `(f + 1) mod 5`; the remaining
three folds supply 60 healthy reference runs.  The inner fold supplies 20
healthy threshold runs, and the outer fold supplies 20 runs from each of the
four conditions for evaluation.  All connected duplicate-seed components
remain in one fold.

## Frozen 46-channel order

The retained zero-based 52-channel indices are

`[0, 1, ..., 40, 46, 47, 48, 50, 51]`.

They correspond to XMEAS1--XMEAS41 followed by XMV6, XMV7, XMV8, XMV10, and
XMV11.  Constant manipulated variables are excluded.  The order is part of
the saved reference state.

No normalization is applied.  A common strictly increasing affine
normalization fitted on healthy data would preserve every empirical rank, so
omitting it is exactly rank-equivalent while removing unnecessary state.

## Frozen reference and score

For each retained channel `j`, concatenate all 500 samples from the 60
fit-role healthy runs and stable-sort them.  The fold reference therefore has
shape `46 x 30000`.  A GO-only final refit may instead use all 100 healthy
runs and has shape `46 x 50000`; no other run count is valid.

At a monitored observation `x[j,t]`, let

```
a[j,t]  = searchsorted_left(reference[j], x[j,t])
mu[j,t] = (a[j,t] + 1) / (N + 2)
```

where `N` is 30000 or 50000.  At local time 160 reset both accumulators to
zero, then update for `t = 160, ..., 499`:

```
Wplus[j,t]  = max(Wplus[j,t-1]  - log(1 - mu[j,t]) - 1.3, 0)
Wminus[j,t] = max(Wminus[j,t-1] - log(mu[j,t])     - 1.3, 0)
W[j,t]      = max(Wplus[j,t], Wminus[j,t])
V[t]        = sum(largest 4 values of W[:,t])
```

All reference, rank, logarithm, recursion, aggregation, and threshold values
are float64.  The run score is `M = max(V[160:500])`.

## Frozen inner threshold and alarm

Within each fold, sort the 20 inner healthy run maxima and set the threshold
to ascending rank 20, namely their maximum.  The comparison is strictly
`M > threshold`.  The first endpoint whose `V[t]` strictly exceeds the
threshold is the alarm; its delay is `t - 160`.  There is no persistence,
clipping, smoothing, or post-hoc threshold adjustment.

The fold fails closed unless all 20 maxima are finite, at least 18 are exactly
distinct, and the rank-20 minus rank-19 gap is greater than
`100 * max(authoritative CUDA repeat error, 1e-10)`.  The threshold must also
be strictly below the analytically reachable 340-step upper bound
`340 * 4 * (log(N + 2) - 1.3)`.

## Frozen development GO gates

All conditions are required:

- Healthy: at most 10 alarms among 100 outer-fold observations and at most 4
  among the 20 observations in every fold.
- Each of Fault 3, Fault 9, and Fault 15: detection rate at least 0.85 over 100
  observations and at least 0.70 in every 20-run fold.
- Each fault: median delay at most 140 samples and nearest-rank 90th-percentile
  delay at most 220 samples, assigning 340 to misses.
- Every fold passes its threshold non-degeneracy and reachable-upper-bound
  checks.
- For every fold, the complete inner-healthy CUDA score -> run-maximum ->
  threshold -> decision path is replayed from the already fitted reference;
  training/reference fitting and outer evaluation are not replayed.  Discrete
  results must match exactly and maximum absolute score difference must not
  exceed `1e-6`.
- Every required value is finite and every exact-selector/access audit passes.

CPU-versus-CUDA scoring is recorded but is report-only.  Failure of any
required gate yields `DEVELOPMENT_NO_GO`; there is no selected candidate and
no final reference artifact.  Only an all-gate `DEVELOPMENT_GO` permits a
single reference refit from all 100 Mode2 healthy runs.

## Governance and hash domains

The frozen JSON protocol is canonical UTF-8 JSON plus exactly one terminal LF.
Its commit and sidecar bind `file_bytes_sha256`, which includes that LF.  The
runner/backend semantic handshake binds `canonical_json_sha256`, which hashes
the canonical parsed JSON without a line terminator.  These identities are
separately named and may never be compared across domains.

The actual persisted protocol is validated by the production backend and the
RTX 5090 CUDA runtime before the one-use development claim is written.  This
pre-claim handshake is protected by a meta-path blocker that fails on any
attempted `h5py` import, including an attempt caught inside backend code.  It
must not import h5py, open an HDF5 file, or decode a data value.  The exact
backend module object that validates the persisted protocol is retained; that
same object performs the numeric run after the claim and access intent, with
no second import or substitute backend.  The access intent is committed only
after that handshake and immediately before the first authorized HDF5 open.

The implementation manifest hashes the runner, backend, monitor, evaluation,
tests, the hash-contract helper, and all local modules executed transitively
by backend imports.  The runner loads the stdlib-only hash helper directly by
file path so the governance hash path does not execute `mate_spc.__init__` or
load NumPy before the backend's deliberate torch-first import order.
It also hash-binds the pre-result conditional v8 MiniRocket charter; that
charter is neither a second v7 candidate nor an authorization to access HDF5,
but prevents a v7 NO_GO successor from being chosen after seeing its result.

The exclusive claim has one fixed project-global path; changing the artifact
output root cannot create a second claim.  No automatic rescue or renamed
replay is permitted after it exists.  Failure records distinguish pre-claim
failure, post-claim/pre-intent failure, post-intent/pre-HDF5 failure,
HDF5-open-started/not-completed failure, HDF5-open/no-slice failure, failure
after a numeric slice was read, and an uncertain phase when the persisted
backend state is absent or inconsistent.  Every accepted backend result must
bind the exact source, selector and fold hashes, zero Mode1/Rieth opens, zero
fault fit/threshold runs, all 400 process-data plus 400 declaration slices,
the five folds, CUDA replay, gate/decision consistency, and its completed
access-state artifact.  A GO additionally binds both the 100-healthy final
reference and its manifest; a NO_GO must contain neither.

After the five-fold gates determine GO or NO_GO, the backend first writes the
append-only `development_decision_before_final_refit.json`, binding the
protocol, candidate, folds, aggregate, CUDA replay, CPU report, decision, and
the fact that final refit has not started.  Only after this record exists may
a GO begin the 100-healthy final refit.  Every GO or NO_GO result, including a
numeric NO_GO, must return its file record.

## GO-only next stages

A development GO does not itself authorize target-domain testing.  Before any
Mode1 access, a separate frozen protocol must bind the saved 100-healthy
reference and implementation.  It may then use Mode1 magnitude-25 IDV1
runs 1--100 exactly once for healthy threshold calibration.  Only a valid
calibration may unlock a one-time Mode1 magnitude-75 IDV1/3/9/15 confirmation.
Magnitude-75 is a prospectively held-out target-domain severity cell, not a
wholly unseen dataset-level or zero-shot external validation set.
