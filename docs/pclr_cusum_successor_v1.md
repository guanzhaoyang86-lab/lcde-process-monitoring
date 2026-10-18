# PCLR-CUSUM successor protocol (v1)

## Why a successor is required

The frozen Hybrid Directional-Energy external audit is complete and cannot be
retuned.  Its external-domain calibration and all audit paths hit the exact
finite score ceiling

```
(500 - 63 - 3 + 1) * (8.0 - 0.75) = 3153.75.
```

Consequently, strict `>` could never alarm and `>=` would alarm on every
healthy run.  Runs 1--200 are therefore treated as consumed diagnostic data and
are permanently excluded from successor fitting, selection, calibration, and
confirmation.

## Primary method

The successor is **Prefix-Conditioned Loop-aware Signed Residual
Density-Ratio CUSUM (PCLR-CUSUM)**.  It assumes that each monitored run has a
certified healthy commissioning prefix.  It is not an unconditional detector
for a process that may already be faulty at its first sample.

For every 500-sample run:

- local `[0, 128)` is the only run-specific personalization baseline;
- local `[128, 160)` is causal lag warm-up;
- local `[160, 500)` is the monitored interval;
- a miss is assigned delay 340 samples; one sample is three minutes.

The 500 samples come from exact normalized source slices: IDV 1 healthy
controls use `[100, 600)`, which excludes the first fault-affected sample at
global index 600, while IDV 3/9/15 fault runs use `[440, 940)`, placing the
first fault-affected sample at local index 160.  The same mapping is used in
development, calibration, and final confirmation.  Because normalized index
`k` maps to raw `processdata` row `k + 1`, the only authorized raw row slices
are `[101:601]` for healthy controls and `[441:941]` for faults (both with
columns `[1:53]`).  Thus the healthy slice ends at normalized global index 599,
whereas the fault monitoring interval begins exactly at global index 600.

Each channel is centered and scaled using only its own baseline median and
MAD, with a per-channel scale floor learned only from development baselines.
The standardized value is transformed with `asinh`; no positive hard clip is
permitted.  A nominal causal ridge-VARX model with lags
`[1, 2, 4, 8, 16, 32]` produces signed XMEAS and XMV innovations.  Features
retain signed residual direction, multiscale dynamics, and the following
zero-based XMV--XMEAS control-loop pairs:

```
(41,1), (42,2), (43,0), (44,3), (45,4), (46,9),
(47,13), (48,16), (49,18), (50,20), (51,21)
```

Three supervised one-vs-healthy density-ratio heads target Faults 3, 9, and
15.  Training is run-weighted and combines point evidence, top-quartile bag
evidence, and causal local `[128,160)` prefix-mean versus post-`160` top-quartile
ranking.  Raw finite logits feed a
one-sided Page recursion with no ceiling and no positive clipping.  A
three-sample persistence envelope defines alarm episodes.

## Immutable data partitions

All conditions sharing a simulation run number are one group.  A healthy run
must never be used as the paired template for the same-numbered fault run.
The shared run number is treated as a common-random-number (CRN) cluster:
IDV 1/3/9/15 outcomes with the same number are dependent, while distinct run
numbers are assumed to be independent simulator replicates.  No analysis may
count the four conditions within a run as four independent units.

| Status | Conditions | Runs | Authorized purpose |
|---|---|---:|---|
| consumed | IDV 1/3/9/15 | 1--200 | descriptive diagnosis only |
| development | IDV 1/3/9/15 | 201--300 | grouped fitting, CV, ablation, selection |
| calibration | healthy prefix from IDV 1 only | 301--400 | frozen threshold only |
| quarantined | IDV 3/9/15 | 301--400 | no access in v1 |
| final confirmation | IDV 1/3/9/15 | 401--500 | one irreversible audit |

Development uses deterministic five-fold splitting by simulation-run group.
Each fold fits its MAD floors, VARX model, feature standardizers, and heads from
the 80 raw training groups only, then transforms the 20 validation groups; a
full-development prefit must never be reused inside CV.
The fixed search grid is the Cartesian product of VARX ridge
`[1e-3, 1e-2, 1e-1]`, classifier ridge `[1e-3, 1e-2, 1e-1]`, and Page drift
`[0.0, 0.25, 0.5]`.  Page drift changes only scoring and must reuse the fitted
fold model for the corresponding two ridge values.  Selection maximizes the
minimum fault detection rate, then minimizes the worst fault p90 delay, then
chooses the more strongly regularized/simpler candidate.  The final model is
refit on all development groups for a frozen number of optimization steps.
Calibration data cannot be used for early stopping.

Within each fold, the monitoring threshold is derived only from the 80
training-fold healthy run maxima: the ascending 79th value (second largest),
without interpolation and with strict `>`.  The 20 validation healthy runs do
not participate in threshold fitting.  A candidate is eligible for selection
only if every fold has at least 76 distinct training maxima, `q79 < q80`, no
known-limit hit or top tie, and `q80-q76` exceeds 100 times the numeric-error
reference (the maximum of full-path CPU/GPU error, run-maxima CPU/GPU error,
the frozen floating tolerance, and parity tolerance); full CPU/GPU parity must
also hold,
there are at most 10 out-of-fold healthy alarms over all 100 runs, and no fold
has more than 4 healthy alarms among its 20 validation runs.  Eligible
candidates are ordered by minimum fault detection rate, worst fault p90 delay,
out-of-fold healthy alarm count, and finally the stronger/simpler
regularization setting.

If all 27 candidates fail selection eligibility, development completes as an
immutable `DEVELOPMENT_NO_GO` result.  The full cross-validation artifact is
retained, no candidate is selected, no final model is generated, and
confirmation freeze is forbidden.  Re-entry with the same v1 protocol only
verifies and returns that completed no-go result; another method or revised
search requires a new versioned protocol and output root.

The development go/no-go gate requires, in grouped cross-validation:

- detection rate at least 0.85 for each fault;
- median delay at most 140 for each fault;
- nearest-rank p90 delay at most 220 for each fault;
- every fold/fault detection rate at least 0.70;
- at most 10/100 out-of-fold healthy alarm runs and at most 4/20 in each fold;
- full CUDA-float64 versus NumPy-float64 scoring parity.

No calibration or final sample may be opened until this gate passes and the
model, implementation, source identity, and confirmation protocol are frozen.

## Calibration and non-degeneracy

The operational threshold is the 99th ascending run maximum among the 100
healthy calibration runs, with strict `>`.  Assume that the 100 calibration
simulation-run groups are IID draws from one fixed operational healthy-run
distribution and that a future healthy run is an independent draw from that
same distribution.  (Finite exchangeability alone is not used to claim this
conditional tolerance guarantee.)  Then its one-sided order-statistic
certificate for a true run-level family-wise alarm probability above 0.10 has
failure probability, over the calibration sample,

```
P(Binomial(100, 0.90) >= 99)
  = 0.000321688053194116 < 0.01.
```

Equivalently, q99 is a one-sided 90%-content tolerance bound with confidence
at least `0.999678311946806`.  The strict `>` alarm rule makes ties
conservative for this statement.

The 99th rank is chosen before numeric access as a power-preserving buffer
inside the final healthy gate; it is not selected from observed results.

Calibration fails closed before final access unless all of these hold:

- at least 95 of 100 run maxima are numerically distinct;
- q99 is strictly below q100;
- the `q100-q95` upper-tail spread is strictly above 100 times the maximum of
  observed CPU/GPU error over standardized logits, Page-CUSUM states,
  persistence states, and the final envelope, frozen floating tolerance, and
  parity tolerance;
- no run maximum equals the Page reset floor `0`, the prior finite ceiling
  `3153.75`, or another frozen numerical/algorithmic limit;
- all paths are finite, and CUDA/NumPy eligibility and alarm arrays match
  exactly with numeric error at most `1e-10`.

## One-time final gate

Runs 401--500 are opened once.  PASS requires all of:

- healthy alarm runs at most 4/100;
- at least 90/100 detected runs for each of Faults 3, 9, and 15;
- at least 270/300 total detected fault runs;
- each fault median restricted delay at most 120 samples;
- each fault nearest-rank p90 restricted delay at most 180 samples;
- exact CPU/GPU alarm and attribution parity.

Only a new alarm episode starting at or after local index 160 counts as a
detection.  Attribution accuracy is secondary and report-only.  Run IDs, not
time points, are the independent units for uncertainty intervals.  At the
healthy PASS boundary of 4 alarms in 100 independent run IDs, the exact
one-sided 95% Clopper--Pearson upper bound is `0.089196250159`; the gate can
therefore support a run-level family-wise alarm claim below 0.10, but not below
0.05.  A supplemental below-0.05 statement is permitted only if at most 1 of
100 healthy runs alarms (upper bound `0.046559811454`) and must be reported as
an observed stronger result, not as the predeclared primary gate.

For each individual fault, the 100 distinct simulation-run IDs may support a
run-level binomial interval under the stated independence assumption.  The
three fault outcomes sharing one run ID remain a CRN cluster, so the 270/300
total-detection requirement is a deterministic PASS criterion only: it must
not be assigned a `Binomial(n=300)` confidence interval.  Any uncertainty for
the aggregate must resample or otherwise model the 100 run-ID clusters.
Regardless of PASS or FAIL, final rows cannot be used to modify and rerun v1.

## Governance

Each numeric phase writes its access intent and immutable manifest before the
first authorized dataset value is decoded.  Partial or staging directories
block re-entry.  A completed phase is verified before any HDF5 stat, hash, or
open.  Protocol files use a same-directory temporary write plus a verified
SHA-256 sidecar commit marker.  Completion verification replays the semantic
intent--manifest--source--model--runtime chain, not merely self-signed hashes.
