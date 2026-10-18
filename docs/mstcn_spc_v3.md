# MSCD-TCN-SPC v3: method and audit protocol

## Status and scope

This document specifies the successor to PCLR-CUSUM v2 for early detection of
Tennessee Eastman faults 3, 9, and 15.  The v2 result was a valid development
no-go: its best candidate detected fault 3 well but did not meet the joint
fault-9/fault-15 and healthy-alarm gates.  v3 therefore changes the
representation, learning objective, and sequential statistic rather than
reusing a failed energy/CUSUM construction.

At the time this document is written, v3 is a development method.  Neither a
formal v3 protocol nor any v3 numeric access to the 25% calibration or 75%
confirmation partitions is authorized by this document alone.  A formal audit
starts only after a passing 50% development iteration has been hash-bound by
`freeze-formal`.

The primary endpoint is fault detection.  Fault attribution is reported
separately and is not allowed to turn a correctly detected but misattributed
fault into a detection miss.  This separation follows the development finding
that faults 9 and 15 were often confused with another fault rather than with
healthy operation.

## Scientific method

The method is **Multi-Scale Causal Dynamics TCN with SPC top-tail scanning**
(MSCD-TCN-SPC).

### Causal 512-channel representation

Each 500-sample run is normalized from its fixed commissioning prefix
`[0,128)`.  The transform is fitted only on training-run prefixes and uses the
feature schema `mstcn_causal_haar_loop_v2`.  It yields exactly 512 float64
channels containing signed robust levels, increments, increment energy,
causal Haar level/energy contrasts, correctly paired controlled-variable loop
errors and hysteresis, and separate physical flow-response cross/hysteresis
terms.  Every feature at endpoint `t` uses only the commissioning prefix and
rows at or before `t`.

The code and the formal protocol both bind the feature schema, channel count,
feature implementation hash, and complete model configuration.  A geometry or
schema change fails closed.

### Dual-head causal TCN

The frozen architecture uses:

- causal windows of 128 samples;
- 32 channels, kernel size 3, dilations `[1,2,4,8,16]`;
- the causal endpoint state plus 8/32/96 trailing mean/RMS summaries of the
  1x1 pointwise projection, followed by hidden width 64;
- one binary detector head and one three-class attribution head for faults
  `[3,9,15]`.

Dense inference endpoints are `160..499`.  MIL training endpoints are
`160..496` with step 4.  No future row, validation threshold, calibration row,
or confirmation row is used in a training update.

### Fault-bag MIL objective

Faults 9 and 15 can be intermittent, so the method does not require every
post-onset point to look faulty.  It uses the top fraction `q` of causal
instances in a run bag for the detector loss.  The only allowed values are
`q in {0.125, 0.25}`.  Every fold trains exactly one model for each q.

The optimizer is AdamW with learning rate `1e-3`, weight decay `1e-4`, 30 fixed
epochs, bag batch size 8, and two runs per each of four classes per batch.  The
per-class epoch cap is 100 (all 80 available training runs in a fold and all
100 in the final refit).  Gradient norm is clipped at 1.  Binary detection loss
weights are exactly 0.5 for healthy and 1/6 for each of faults 3, 9, and 15;
the attribution loss weight is 0.25.  There is no validation early stopping.

### Robust multi-scale top-tail scan

The detector logit is standardized endpoint-wise using robust statistics fit
only to healthy training paths.  For each trailing window
`W in {16,32,64,128}`, the largest `rho` fraction of standardized evidence is
averaged and robustly studentized using healthy training runs.  The four W
branches are combined by their maximum.  The only allowed tail fractions are
`rho in {0.125,0.25}`.

The four development candidates are exactly the Cartesian product of the two
q values and two rho values.  An alarm requires two consecutive samples
strictly greater than the threshold.  There is no finite score ceiling.  CPU
and CUDA float64 score errors must be at most `1e-9`, and alarm/discrete outputs
must be identical.

## Partition governance

All source runs are from the byte-exact official Extended TEP Mode 1 archive.
The same run number across IDVs is a balanced stratification block, not a
common-random-number identity; the stored RNG seed identifies an independent
simulation.

| Partition | Allowed role | Numeric access policy |
|---|---|---|
| magnitude 50%, IDV 1/3/9/15, runs 1-100 | development | already consumed; repeatable iterations permitted |
| magnitude 25%, IDV1, runs 1-100, pre-fault window only | calibration | sealed until formal freeze; one time |
| magnitude 25%, IDV3/9/15, runs 1-100 | none | quarantined; never accessible in v3 |
| magnitude 75%, IDV1/3/9/15, runs 1-100 | final confirmation | sealed until valid calibration; one time even on failure |
| magnitude 100%, IDV1/3/9/15, runs 1-200 | historical diagnostics only | consumed; forbidden for v3 fitting or selection |

Healthy sequences use global source window `[100,600)`, which ends before the
first affected global sample.  Fault sequences use `[440,940)`, placing fault
onset at local index 160.  Each sequence therefore has length 500 and a missed
fault receives delay 340.  Extended TEP Mode 2 is reserved for a later
cross-mode portability audit and is not part of this main protocol.

## Repeatable development iteration

`development-iteration` is intentionally not a formal freeze.  It constructs
only the magnitude-50 access plan and creates a new, exclusive iteration
directory.  An iteration ID cannot be overwritten; another ID may be run after
reviewing prior development results.

Five deterministic 20-block folds are assigned by a frozen salted hash of run
number.  In each fold, all transforms, both q models, robust pointwise
standardizers, and branch studentizers are fitted again using the 80 training
blocks only.  Validation blocks are transformed only afterward.

For every q/rho candidate and fold, the alarm threshold is the noninterpolated
79th ascending order statistic of 80 training-healthy run maxima.  The 20
validation healthy runs are never used to set that fold threshold.  A fold is
numerically valid only when:

- at least 76 of 80 maxima are numerically distinct;
- rank 79 is strictly below rank 80;
- the adjacent gap `M_(80)-M_(79)` is greater than 100 times the larger of the
  observed parity error and the fixed float tolerance;
- CPU/CUDA float64 error is at most `1e-9`, with identical discrete output.

A candidate is development-eligible only if all numerical checks and all of
the following gates pass:

- out-of-fold healthy alarm runs: at most 10/100 overall and at most 4/20 in
  every fold;
- each fault detection rate: at least 0.85 overall and at least 0.70 in every
  fold;
- for every fault, median delay at most 140 and nearest-rank p90 delay at most
  220 samples, counting misses as delay 340.

Eligible candidates are selected deterministically by: maximum minimum overall
fault detection rate, maximum minimum fold-by-fault detection rate, minimum
worst p90 delay, minimum worst median delay, minimum healthy alarms, larger
rho, larger q, then canonical candidate ID.  If none is eligible, the result is
`DEVELOPMENT_NO_GO`, no model is emitted, and formal freeze is forbidden.

## Formal freeze and hash binding

`freeze-formal` is metadata/hash-only.  It requires a completed passing
development iteration and binds:

- the complete model/training/scan configuration and selected q/rho;
- the final model bytes (including canonical metadata with the final-refit
  training record and fitted pointwise/branch monitor state) and development
  result/completion/config artifacts;
- this design document, runner, features, model, monitor, and numeric backend;
- the official Mode 1 byte identity and metadata-only topology record;
- the absolute output root and all partition/access contracts.

Publication uses an exclusive freeze claim and three-file atomic bundle:
canonical protocol JSON, SHA-256 sidecar, and commit marker binding both.  A
partial or existing bundle is rejected.  Later phases require the digest on the
command line and rehash every live input.  Formal freeze is HDF5-numeric-free:
it performs Torch-first safe model validation and may use NumPy while checking
metadata topology, but it never decodes `processdata` or `idv_init` values.

## One-time calibration

Calibration is allowed to access only magnitude-25 IDV1 runs 1-100 through the
pre-fault source window.  Its access intent is durably written after an
exclusive phase claim and before the backend can open the source.

The formal threshold is the noninterpolated 98th ascending order statistic of
100 healthy run maxima, with strict `>` comparison.  This corresponds to the
finite-sample marginal exceedance bound `3/101` under exchangeability.  It also
supports the pre-registered conditional statement “run-level false-alarm
probability below 0.10 with confidence at least 0.99” under the stated iid/
exchangeability assumptions; it does **not** certify a `<0.05` claim.

Calibration is valid only with 100 finite maxima, at least 96 numerically
distinct maxima, and rank 98 strictly below rank 100.  Rank-100 minus rank-96
spread is recorded as a diagnostic only and is not an additional acceptance
gate.  Invalid calibration is still a consumed, completed one-time phase and
forbids confirmation.

## One-time confirmation

Confirmation is possible only after semantic replay of a valid, hash-bound
calibration.  It then claims the phase and writes its exact magnitude-75 access
intent before any source access.  It runs exactly once even if the answer is
FAIL.  The pre-registered gates are:

- healthy alarm runs at most 4/100;
- at least 90/100 detections for each fault and at least 270/300 in total;
- median delay at most 140 and nearest-rank p90 delay at most 220 samples for
  each fault, with misses assigned delay 340.

Attribution accuracy among all runs and detected runs is reported separately.
No calibration or confirmation observation can alter weights, q, rho,
standardizers, scan windows, persistence, threshold rule, or any gate.
Calibration and confirmation load the monitor embedded in the bound model NPZ;
they never refit its standardizers from 25% or 75% data.

## Runtime and failure semantics

All numeric phases require an available CUDA device and float64.  On the
target Windows runtime the production backend imports Torch before NumPy,
HDF5/data adapters, model, or monitor modules.  The runner itself has a
standard-library-only top level; merely importing it cannot load Torch or
NumPy.

Claims use exclusive creation.  Any exception after a claim retains the claim
and staging directory as visible evidence of a partial/consumed attempt.  The
runner never silently deletes, retries, or overwrites a formal phase.  A
successful phase is moved atomically into place and contains a completion
manifest with size/SHA-256 records for every artifact.  Unexpected files,
changed hashes, non-finite JSON, missing fields, extra fields, wrong device,
wrong dtype, parity failure, split drift, or threshold replay disagreement all
fail closed.
