# CRIS-v9: covariate-regulated innovation scanning

## Status and scope

CRIS-v9 is a new scientific hypothesis following the terminal v8 protocol
handshake abort.  It is not a repair, rename, or rescue iteration of the v8
MiniROCKET classifier.  It does not use the consumed Extended TEP Mode2 pool
or Rieth data for fitting, calibration, selection, or evaluation.

The working title is **Covariate-Regulated Innovation Scanning for Weak and
Intermittent Process Faults with Trajectory-Level False-Alarm Control**.
CRIS learns healthy conditional dynamics, rather than a boundary between
named fault classes.  Its intended contribution is the combination of:

1. covariate-regulated, multi-horizon Student-t innovations;
2. separate location, scale, and temporal-dependence evidence;
3. a fixed mixture of intermittent two-state HMM betting scans; and
4. an outer trajectory-maximum calibration that includes all channels,
   horizons, scales, and scan states in the false-alarm unit.

No individual ingredient is claimed as new.  The contribution claim is the
causal combination and its run-level calibration/evaluation protocol.

## Healthy conditional model

For target signals `x_t`, observed mask `m_t`, elapsed-time feature `dt_t`,
and permitted operating covariates `c_t`, a two-layer GRU receives only the
causal context ending before a target:

```text
h_t = GRU([x, c, m, dt] over the previous 60 samples)
```

In the four-tank study, the six targets are the four measured levels and two
pump commands.  Only the two externally assigned setpoints are covariates.
Pump commands are closed-loop responses and are therefore forbidden as
conditioning variables; retaining them as monitored targets prevents the
conditional model from explaining away a fault through a post-treatment
control action.

For horizons `d in {1, 5, 15}`, the frozen head returns diagonal Student-t
parameters `mu[t,d]` and `scale[t,d]` with `nu=5`.  The production v9
implementation deliberately uses a positive diagonal conditional scale; it
does not claim a low-rank conditional covariance.

The model is trained only on healthy runs with Student-t negative
log-likelihood.  Fixed training defaults are two GRU layers, hidden width 64,
AdamW learning rate `3e-4`, weight decay `1e-4`, 40 epochs, batch size 1024,
gradient clipping at 1.0, seed 20260915, deterministic CUDA, and no online
refitting after deployment.  Inference uses batches of 2048 windows.
Dataset-specific weights may be fitted from that dataset's authorized healthy
fit partition, but architecture, training rule, monitoring rule, and candidate
set are fixed beforehand.

The unit-variance innovation is

```text
u[t,d,j] = sqrt((nu - 2) / nu) * (x[t,j] - mu[t,d,j]) / scale[t,d,j].
```

## Multi-moment evidence

CRIS retains three fault mechanisms instead of reducing every innovation to
one squared energy:

```text
location   : u
scale      : (u^2 - 1) / sqrt(2)
dependence : u[t] * u[t-1]
```

Both signs are monitored.  Causal rolling summaries use window lengths
`{4, 16, 64}`.  Within each moment/horizon/scale, the largest `top_k` absolute
sensor contributions form a sparse score.  A point reference fitted only to
authorized healthy reference scores returns the empirical upper-tail rank

```text
p = (1 + count(reference >= score)) / (n + 1).
```

The fixed power transform is `e = eta * p^(eta-1)` with `eta=0.5`.  Because
the point reference pools serially dependent time points, `p` is not claimed
to be a conformal p-value and `e` is not claimed to be a valid e-value.
Finite-sample false-alarm control is attributed only to the outer rank of
independent healthy-trajectory maxima.

## Intermittent scan

For each component, two HMM modes are mixed with equal prior weight:

```text
(rho, lambda) = (0.02, 0.20)  # sparse, longer episodes
(rho, lambda) = (0.10, 0.50)  # frequent, shorter episodes
```

If `q(0), q(1)` are the previous no-fault/fault masses, one causal update is

```text
q_tilde(0) = (1-rho) q(0) + lambda q(1)
q_tilde(1) = rho q(0) + (1-lambda) q(1)
q_new(0)   = q_tilde(0)
q_new(1)   = q_tilde(1) * e
```

Updates start from `q=(1,0)` and are implemented in the log domain.  All
moments, horizons, scales, and HMM modes are combined with frozen weights
before thresholding.  A missing component holds its own HMM state while
available enabled components continue, with the combination denominator kept
fixed.  Sixty-four consecutive missing samples reset that component.  A time
with no available enabled component is `DATA_QUALITY_UNKNOWN`; UNKNOWN fault
time is counted as a miss.

## Trajectory-level threshold

For each independent healthy calibration run, compute the maximum valid scan
score `M_i`.  Given `n` runs and target `alpha=0.05`, use

```text
k = ceil((n + 1) * (1 - alpha))
h = kth ascending M_i
alarm iff score > h.
```

If `k > n`, the requested finite-sample threshold is unavailable and the
implementation must refuse rather than substitute the sample maximum without
disclosure.  Under exchangeable healthy trajectories this rank controls the
next-trajectory alarm probability at the corresponding finite-sample level.
Rows, windows, channels, horizons, and repeated views are never counted as
independent trajectories.

## Three frozen development candidates

The method family contains only these monitoring candidates; the neural model
and all other settings are shared:

| id | top_k | location/scale/dependence weights |
|---|---:|---|
| `cris-k1-equal` | 1 | `1/3, 1/3, 1/3` |
| `cris-k2-equal` | 2 | `1/3, 1/3, 1/3` |
| `cris-k2-location` | 2 | `1/2, 1/4, 1/4` |

Selection on authorized development runs is lexicographic: satisfy all
healthy/covariate-shift gates; maximize the minimum fault-family recall;
minimize pre-onset false-alarm runs; minimize worst-family median relative
delay; then prefer the table's earlier candidate.  Exactly one selected
configuration may enter independent confirmation.

## Synthetic closed-loop development and confirmation

The primary controlled study uses a nonlinear closed-loop four-tank process.
The simulator, controller, noise law, fault definitions, seed ranges, code,
and source hashes are frozen before independent confirmation generation.
Fault families are persistent sensor bias, intermittent pump-efficiency loss,
valve stiction/deadband, and innovation variance/dependence change.  Healthy
training and calibration trajectories cover setpoints 10.5--15.0 with
75-sample randomized blocks.  The covariate-shift-only condition follows a new
trajectory inside that same operating envelope, so it tests conditioning
rather than neural extrapolation.

Development can use designated training, score-reference, threshold, and
development seeds.  Independent confirmation seeds must not be generated or
read until the selected method and result validator are frozen.

Confirmation gates use complete trajectories:

- healthy alarm runs at most 11/200 and one-sided 95% Clopper-Pearson upper
  bound at most 0.10;
- each persistent fault-family recall at least 0.80;
- intermittent-pump recall at least 0.75;
- each family median delay at most 25% of the monitored post-onset length;
- covariate-shift-only alarm rate at most 0.10;
- exact repeated CUDA alarm decisions and score error at most `1e-6`.

For a direct sensor bias, `fault_mask` begins at the intervention sample.  For
dynamic pump, valve, and dependence interventions at `t`, it begins at the
first affected observation `t+1`.  Persistent alarms receive credit only
where that mask is active.  An intermittent-pump alarm receives credit at an
active sample or within a fixed five-sample right grace period; inactive-gap
alarm onsets are reported separately and receive no credit.  A continuous
alarm already active before the first affected observation cannot acquire
detection credit without a new false-to-true onset.

The frozen baselines are standardized PCA `T2+Q` retaining at least 90%
variance, standardized six-target energy CUSUM with allowance 0.5, and mean
squared CRIS-innovation CUSUM with allowance 0.5.  Each uses the same 0.05
complete-trajectory maximum calibration role as CRIS.  A superiority claim
additionally requires at least a ten
percentage-point recall improvement at no larger healthy alarm rate for at
least three fault families.

## Prospective SKAB audit

SKAB v0.9.0 is a small external real-testbed audit, not the source of a strong
run-level FAR theorem.  Its one anomaly-free experiment cannot substitute for
many independent healthy runs.  The preaccess charter fixes whole-file
development/final roles before download.  Final files are not opened unless
the fixed development gate passes.  Event recall, first-alarm delay, pre-event
false-alarm experiments, alarm occupancy, and a right-sided 60-second
changepoint metric are reported; point-adjusted F1 is forbidden.

The primary statistical unit is a complete CSV experiment.  Fault subgroups
with only two final experiments are descriptive and cannot support subgroup
generalization claims.

## Required tests and reporting

Tests must cover causal prefix invariance, horizon alignment, Student-t scale
positivity, read-only array handling, no-pickle persistence, CPU/CUDA replay,
HMM recursion against enumeration, rank ties and strict comparison, UNKNOWN
pause/reset, complete-run metrics, simulator counterfactual equality before
onset, and an actual producer-consumer integration path for every committed
authorization schema.

Every paper/report must disclose v2-v5 and v7 numerical NO-GO outcomes and the
v6/v8 infrastructure aborts.  The v8 abort is documented in
`gpu_minirocket_logodds_v8_abort_audit.md`.
