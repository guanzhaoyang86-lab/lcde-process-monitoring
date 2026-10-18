# Severity-stratified PCLR-CUSUM v2 protocol

## Purpose and claim boundary

The previous absolute-logit external audit failed because every calibration
maximum and every audit maximum reached the same finite ceiling, 3153.75.
PCLR-CUSUM removes that clipping mechanism and tests signed, persistent
residual evidence.  The v2 study is a prospective **severity-shift
confirmation within Extended TEP Mode 1**.  It is not an external-dataset,
zero-shot, cross-mode, or full-magnitude confirmation.

The old Rieth TEP values are never used.  The already consumed Extended TEP
Mode 1 cells at magnitude 100%, IDV 1/3/9/15, Runs 1--200 are also never
reopened by a v2 numeric phase.

## Source topology and independent simulation identity

The byte-exact official DTU/Figshare file is `TEP_Mode1.h5`, DOI
`10.11583/DTU.13385936.v1`, size 23,924,710,848 bytes, MD5
`c79ecc6d12a5ccb49caffcd8f531feae`, and SHA-256
`d6b0ed0ef3974ff571afd6717ba0418a5b8eb58a7822c84a05735572d52b3294`.

Metadata-only inspection established that each target IDV has 100 completed
runs at each of 25%, 50%, and 75%, plus 200 runs at 100%.  The HDF5 `seed`
attribute is unique within and across the 16 target IDV-by-magnitude branches.
The RNG seed, not the displayed Run number, is therefore the independent
simulation identity.  Matching Run numbers across IDVs are used only to build
balanced stratification blocks containing one example of each class; they are
not described or analysed as common-random-number clusters.

The immutable allocation is:

| Role | Magnitude | IDVs | Runs | Numeric use |
|---|---:|---|---:|---|
| consumed diagnosis | 100% | 1, 3, 9, 15 | 1--200 | permanently forbidden |
| development | 50% | 1, 3, 9, 15 | 1--100 | grouped five-fold selection and refit |
| calibration | 25% | 1 only | 1--100 | pre-fault control windows only |
| calibration quarantine | 25% | 3, 9, 15 | 1--100 | never opened |
| final confirmation | 75% | 1, 3, 9, 15 | 1--100 | opened once after calibration passes |

The 50%, 25%, and 75% branch seed sets are disjoint.  This permits a
development/calibration/confirmation separation even though each branch
reuses display labels Run1--Run100.

## Exact windows and data contract

Low-magnitude `processdata` datasets have exact shape `(2000, 54)` and
float64 dtype.  They contain no extra initial-state row: normalized row `k`
maps to raw row `k`.  The 52 inputs are raw columns `[1:53]` in the exact
official label order.

Every model input is a 500-sample sequence:

- IDV1 pre-fault control: normalized/raw rows `[100:600)`;
- IDV3/9/15 fault sequence: normalized/raw rows `[440:940)`;
- local `[0:128)`: commissioning baseline;
- local `[128:160)`: causal lag warm-up;
- local `[160:500)`: monitored interval.

The fault declaration must be exactly one active IDV with the frozen
low-magnitude profile `[0, magnitude/100, 30]`.  At three-minute sampling,
the first fault-affected sample in the fault sequence is local index 160.  The
IDV1 control slice ends at global index 599, before the first affected global
index 600.  Exact time spacing is verified when an authorized numeric slice
is materialized.

## Method

For each run, PCLR-CUSUM:

1. estimates robust median/MAD scaling only from local `[0:128)`, applies a
   development-fitted positive scale floor, and uses an `asinh` transform
   without a positive hard clip;
2. fits a ridge VARX baseline with lags `[1,2,4,8,16,32]`;
3. retains signed residual direction and constructs 548 causal multiscale
   features, including XMV--XMEAS loop pairs, hysteresis, limit-cycle, slope,
   and cross-correlation evidence;
4. fits three one-vs-control density-ratio heads for Faults 3, 9, and 15 with
   point, top-quartile bag, and causal prefix-versus-post ranking losses;
5. standardizes logits from development control evidence and feeds them to
   three one-sided Page CUSUM recursions;
6. uses the maximum three-sample persistent component as the run score.

The Page state has no finite cap.  Threshold comparisons are strictly `>`.
Only a new alarm episode starting at or after local index 160 counts.

## Development selection

The 100 Run-number stratification blocks are deterministically assigned to
five folds of 20.  Each fold refits MAD floors, VARX coefficients, feature
standardizers, and density-ratio heads using only its 80 training blocks.
Validation values never determine a fold threshold.

The frozen grid is:

- VARX ridge: `1e-3, 1e-2, 1e-1`;
- classifier ridge: `1e-3, 1e-2, 1e-1`;
- Page drift: `0, 0.25, 0.5`.

For each fold, the operational development threshold is the 79th ascending
order statistic among 80 training-control run maxima.  It must be below the
80th, have at least 76 numerically distinct maxima, avoid known finite limits,
and have a top-tail spread safely above measured CPU/CUDA and floating-point
error.

A candidate is eligible only if:

- each fault has out-of-fold detection rate at least 0.85;
- each fault has restricted median delay at most 140 samples and nearest-rank
  p90 delay at most 220;
- every fold/fault cell detects at least 0.70;
- at most 10/100 out-of-fold control runs alarm, and at most 4/20 in any fold;
- complete NumPy/CUDA float64 parity passes at absolute tolerance `1e-10`.

Selection then maximizes the minimum fault detection rate, minimizes the worst
fault p90 delay and control alarms, and uses the frozen regularization
tie-breaks.  The selected configuration is refit from scratch on all 100
development blocks for exactly 500 optimization steps.  If no candidate is
eligible, the phase terminates as `DEVELOPMENT_NO_GO`; it emits no model and
does not authorize calibration or confirmation.

## Calibration

After development passes, a second immutable protocol binds the model and
opens only the 100 magnitude-25 IDV1 pre-fault control windows.  The
operational threshold is their non-interpolated 99th ascending run maximum.
Under the stated IID/exchangeability assumption for distinct RNG-seed
simulations from the fixed Mode 1 nominal prefix distribution,

`P[Binomial(100, 0.9) >= 99] = 0.000321688053194116 < 0.01`.

Calibration must also have at least 95 distinct maxima, q99 below q100, no
known finite ceiling, exact alarm parity, and a q100-minus-q95 tail spread
greater than 100 times the numeric error reference.  Failure forbids opening
the final cells.

## One-time confirmation and reporting

Magnitude-75 final cells are opened once whether the outcome is PASS or FAIL.
PASS requires all of:

- no more than 4/100 control-window alarm runs;
- at least 90/100 detections for each of Faults 3, 9, and 15;
- at least 270/300 fault detections in total;
- restricted median delay at most 120 samples for every fault;
- nearest-rank restricted p90 delay at most 180 for every fault;
- complete CPU/CUDA float64 path, eligibility, alarm, and attribution parity.

Because the three faults are different target distributions, the pooled
270/300 count is a deterministic gate and must not be presented as a
Binomial(300) confidence statement.  For 4 alarms in 100 independent control
runs, the one-sided 95% Clopper--Pearson upper bound is about 0.08920 and
supports only a false-alarm claim below 0.10.  A descriptive claim below 0.05
requires at most 1/100 alarms, whose upper bound is about 0.04656.

All numeric phases require CUDA float64 and independently replay the full
NumPy path.  Protocols, access intents, source/implementation hashes, models,
evidence arrays, RNG-seed identities, results, and completion manifests are
atomically written and hash-bound.
