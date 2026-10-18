# PCF-CycleRank-Scan v6 pre-access method and governance specification

## Status and claim boundary

This document defines the single fixed v6 candidate
`pcf-cyclerank-scan-v6-fixed` before any v6 HDF5 open.
It does not itself authorize numeric access.  Authorization exists only after
`scripts/run_pcf_cyclerank_gpu_v6.py freeze-development` has verified every
required implementation and test file and atomically created the development
protocol, SHA-256 sidecar, and commit record.

The study fits only Extended TEP Mode 2 magnitude-50 data.  Earlier Rieth and
Extended TEP Mode 1 results are historical design provenance.  They are not
v6 fitting, selection, calibration, or confirmation data.  Mode 1 magnitude
25 and 75 remain unavailable during development.

Because prior methods used Mode 1 magnitude 50 and 100, a later Mode 1
magnitude-75 result is a prospectively held-out **target-domain severity
cell**, not a wholly unseen dataset-level, zero-shot, or independent external
validation.  All v2--v5.1 no-go results and earlier failed confirmations must
be disclosed with any v6 result.

## Immutable development allocation

The source is the official `TEP_Mode2.h5`, SHA-256
`8462c0249761ae762a99028c7f8391620b68c69aaea4ecc961fe482799c974b4`,
size 23,897,125,720 bytes.  Development uses only
`Mode2/SingleFault/SimulationCompleted`, magnitude 50, IDV 1/3/9/15,
runs 1--100.

Each of the 400 selectors authorizes exactly its named `processdata` rows and
columns `[0,53)` (time plus the 52 process channels), the named `idv_init`
array `[0,3)x[0,28)`, the Run group's `seed` attribute, and object
type/shape/dtype metadata.  It does not authorize any root-dataset value,
including `Processdata_Labels`, or any `processdata` value outside the selected
row window.

Each analytic sequence has 500 rows and 52 channels:

- IDV1 healthy control: normalized rows `[100,600)`;
- IDV3/9/15 fault: normalized rows `[440,940)`;
- local fault onset: index 160;
- miss delay: 340 samples.

The displayed run number is a balanced four-class stratification block, not a
paired random seed.  Seed-connected run-number components are never split.
The five exact folds are:

```text
F0 = 8,14,15,20,26,31,37,42,43,48,53,58,63,68,73,79,84,90,95,98
F1 = 3,6,10,16,21,27,32,38,44,49,54,59,64,69,74,76,80,85,91,96
F2 = 4,9,11,17,22,28,34,39,45,50,55,60,65,70,75,81,86,88,92,97
F3 = 5,12,18,23,24,29,33,35,40,46,51,56,61,66,71,77,82,87,93,99
F4 = 1,2,7,13,19,25,30,36,41,47,52,57,62,67,72,78,83,89,94,100
```

For outer fold `j`, `Fj` is the 20-run-per-class outer test set,
`F((j+1) mod 5)` contributes only its 20 healthy runs to the inner threshold,
and the remaining three folds contribute 60 runs per class to fitting.
Inner-fold fault values are forbidden.  The candidate count is exactly one;
there is no hyperparameter grid, ensemble, early stopping, or outcome-based
retry.

Mode 2 data are already adaptive exploratory data after v5.1.  The five-fold
v6 result is therefore a feasibility gate, not an unbiased generalization
estimate.

## Paired domains

Every endpoint window is causal and inclusive:
`current(t)=x[t-127:t+1]`.  The fixed same-run reference is `x[0:128]`.
Training endpoints are `128..496` with stride four, giving 93 endpoints:

- prefix endpoints `128..156` with stride four: 8;
- monitor endpoints `160..496` with stride four: 85;
- ignored transition endpoints for stage-2 labels: `160..188`: 8;
- mature fault endpoints `192..496`: 77.

Online inference uses the eight prefix endpoints above and every integer
monitor endpoint `160..499`.

All network tensors are strict float32.  Within an outer-fold fit, channel
normalization is fitted only on rows 0--127 of the 60 fit-fold healthy runs.
For every raw channel, pool
those 7,680 fit-healthy prefix values, fit the median and
`max(1.4826*MAD,1e-6)`, standardize, and clip to `[-10,10]`.  Domain `A` is the
elementwise standardized `current-reference` difference with shape
`[B,52,128]`; it receives no second normalization.

### Frozen 604-dimensional Bfeat order

The first 572 entries are channel-major.  For channel 0 through 51, append the
following 11 current-minus-reference statistics in this order:

1. mean;
2. `log(population variance+1e-8)`;
3. `log(mean(diff(x)^2)+1e-8)`;
4. six normalized PSD log shares in bands `[1,2)`, `[2,4)`, `[4,8)`,
   `[8,16)`, `[16,32)`, `[32,65)`;
5. spectral entropy;
6. spectral centroid.

For every length-128 window, subtract its own mean and compute the unwindowed
rFFT power at positive bins 1--64.  Add `1e-12` to every bin and normalize by
the resulting 64-bin sum to get `q_k`.  A band value is
`log(sum(q_k in band)+1e-8)`, entropy is
`-sum(q_k log(q_k))/log(64)`, and centroid is
`sum(k*q_k)/64` for `k=1..64`.  Every listed statistic is calculated for the
current and fixed reference windows separately, then differenced.

The next 24 entries are pair-major: first XMV11/channel51 with
XMEAS11/channel10, then XMV11/channel51 with XMEAS22/channel21.  Each pair has
this exact 12-entry order:

1. current dwell, current-minus-reference dwell;
2. current stick-slip, current-minus-reference stick-slip;
3. current reversal, current-minus-reference reversal;
4. current oriented area, current-minus-reference oriented area;
5. current-minus-reference lag asymmetry;
6. log current/reference ARX residual MSE ratio;
7. current-minus-reference Welch coherence;
8. current-minus-reference actuator PSD concentration.

Dwell is `mean(abs(diff(u))<=0.05)`; stick-slip is the fraction of adjacent
increments with current absolute change at least 0.50 after one at most 0.05;
reversal is the fraction of adjacent increment products below zero.  Lag
asymmetry is the difference between the two lag-one increment correlations.
The reference-window ARX regression uses an intercept, lag-one response and
lag-one actuator, with diagonal ridge `1e-6`.  Welch coherence uses length-32
periodic-Hann segments starting every 16 samples and averages positive-bin
coherence.  PSD concentration is the largest normalized positive-bin power.

The final eight entries are these causal current-window statistics (they are
not current-minus-reference differences):

1. PCA T2;
2. PCA Q;
3. lag-4 DPCA T2;
4. lag-4 DPCA Q;
5. SFA slow energy;
6. SFA fast energy;
7. VAR8 state diagonal-Mahalanobis energy;
8. VAR8 residual full-Mahalanobis energy.

Within an outer-fold fit, these eight health coordinates are fitted only from
the normalized first 128 rows of the 60 fit-healthy runs.  PCA uses fixed rank
16.  DPCA concatenates
the current vector and lags 1--4, also at fixed rank 16.  Retained eigenvalues
have floor `1e-6`.  SFA diagonalizes the first-difference covariance in the
whitened PCA-16 space and averages squared, delta-variance-standardized energy
over the four slowest or four fastest directions.  VAR order is eight.  Its
centered lag-state regression uses ridge `1e-6`; the state statistic uses
diagonal fitted state variances and the residual statistic uses the fitted
52-by-52 covariance plus `1e-6 I`.  Each coordinate is a causal within-window
mean of its pointwise statistic.  There is no reference subtraction and no
additional `log1p` transform in this final block.

Within an outer-fold fit, fit a separate median and
`max(1.4826*MAD,1e-6)` for each of the 604 features using exactly the
`60*85=5100` fit-healthy sparse monitor windows at
endpoints 160--496 with stride four, standardize, and clip to `[-10,10]`.
The same frozen transform applies to fit, inner and outer values and is stored
in the checkpoint.  This standardized vector is the network's `Bfeat`.

## Stage 1: health-only paired cycle model

Within an outer-fold fit, Stage 1 uses exactly the 5,100 fit-healthy
`(A,Bfeat)` pairs at the 85 sparse monitor endpoints.  The eight prefix
endpoints are excluded from Stage 1.

`G_AB` is:

```text
Conv1d 52->128 k7 s2 p3, GroupNorm(8), GELU
Conv1d 128->256 k5 s2 p2, GroupNorm(16), GELU
Conv1d 256->256 k3 s2 p1, GroupNorm(16), GELU
time mean/max concatenate: 512
Linear 512->256, GELU, Linear 256->604
```

`G_BA` is:

```text
Linear 604->512, GELU, Linear 512->4096, GELU
reshape [256,16]
ConvTranspose1d 256->256 k4 s2 p1, GroupNorm(16), GELU
ConvTranspose1d 256->128 k4 s2 p1, GroupNorm(8), GELU
ConvTranspose1d 128->52 k4 s2 p1, no final activation
```

`D_A` is a PatchGAN:

```text
Conv1d 52->64 k4 s2 p1, LeakyReLU(0.2)
Conv1d 64->128 k4 s2 p1, GroupNorm(8), LeakyReLU(0.2)
Conv1d 128->256 k4 s2 p1, GroupNorm(16), LeakyReLU(0.2)
Conv1d 256->1 k3 s1 p1, output [B,1,16]
```

`D_B` is `Linear 604->256->64->1` with LeakyReLU 0.2 after the first two
linear layers.

The LSGAN discriminator loss in each domain is
`0.5*(MSE(D(real),1)+MSE(D(fake.detach()),0))`.  Each generator adversarial
term is `0.5*MSE(D(fake),1)`.  The generator reconstruction terms are:

```text
10*L1(G_BA(G_AB(A)), A)
+ 10*L1(G_AB(G_BA(Bfeat)), Bfeat)
+ 5*L1(G_AB(A), Bfeat)
+ 5*L1(G_BA(Bfeat), A)
```

The total generator objective is both adversarial terms plus all four listed
reconstruction terms.  Use separate AdamW optimizers with learning rate
`1e-4`, weight decay `1e-4`, betas `(0.9,0.999)`, epsilon
`1e-8`, 120 fixed epochs, shuffled batch size 128 without dropping the final
partial batch, seed 20260910, and the last epoch.  No early stopping is
permitted.  Freeze both generators after stage 1; stage 2 cannot update them.

## Stage 2: residual detector and CycleRank objective

The residual signature has exactly 656 entries.  Its first 52 are
`log1p(mean_time((A-G_BA(G_AB(A)))^2))`; its final 604 are
`abs(Bfeat-G_AB(G_BA(Bfeat)))`.

The encoder is `Linear 656->128, GELU, Dropout(0.1), Linear 128->32, GELU`.
The detection head is `Linear 32->1`; the attribution head is
`Linear 32->3`, with IDs 3/9/15 mapped to 0/1/2.

The fixed total loss is:

```text
0.40 * balanced endpoint BCE
+ 0.30 * paired rank loss
+ 0.20 * supervised contrastive loss
+ 0.10 * mature-fault attribution cross entropy
```

Balanced BCE gives total weight 0.5 to the negative population (the 85
healthy monitor endpoints and the eight fault prefix endpoints) and total
weight 0.5 to the 77 mature fault endpoints; faults 3, 9 and 15 contribute
equally.  Fault prefix endpoints are also normal baselines for ranking; fault
endpoints 160--188 are ignored by BCE, attribution and contrastive losses.
Mature endpoints 192--496 are positives.

For every fault run, take the mathematical median of its eight prefix logits.
The paired rank term is the mean over its 77 mature logits of
`softplus(1-(post_logit-prefix_median))`, with the three faults equally
weighted.  Supervised contrastive loss uses L2-normalized 32-dimensional
embeddings, labels `{0,3,9,15}`, temperature 0.1, excludes self-pairs and
requires at least two examples per class.  Attribution cross entropy uses
only mature fault endpoints.

Use AdamW learning rate `3e-4`, weight decay `1e-4`, betas `(0.9,0.999)` and
epsilon `1e-8`, 80 fixed epochs, two runs per class and four classes per joint batch
(eight runs total), exactly 30 balanced batches per outer-fold-fit epoch, seed
20260910, no early stopping, and the last epoch checkpoint.

## GO-only final refit

Only after all five frozen folds satisfy every development GO gate, refit the
same fixed candidate on the same already-authorized 400 Mode 2 records, now
using all 100 runs per class.  Refit the channel and health models from the 100
healthy prefixes, fit the B standardizer and Stage 1 from `100*85=8500`
healthy monitor rows, and run Stage 2 for 50 two-runs-per-class batches per
epoch.  From the final model's 100 healthy paths, fit and save the scan
calibrator using all `100*340` dense monitor logits after each run's eight-logit
prefix median is removed.  This calibrator must be frozen before any Mode 1
access.  Seed, endpoint definitions, architectures, losses, epochs, optimizer
settings and scan rules remain unchanged.  This refit cannot change the
development decision, performs no target-domain calibration, and does not
authorize or access Mode 1.

## Dense run-relative scan

CUDA inference is authoritative.  For each fit-healthy run, let `b_i` be the
mathematical median of its eight prefix logits and let
`v_it=monitor_logit_it-b_i` for all 340 dense endpoints 160--499.  Pool all
`60*340=20,400` fit-healthy values and freeze

```text
m = median(v)
s = max(1.4826*median(abs(v-m)), 1e-3)
u_it = (v_it-m)/s
```

Inner and outer paths use the frozen `m,s` without updating them.  For endpoint
`t`, with drift `d=0.25`, define the sustained branch

```text
G_t = max over w in {4,8,16,32,64} that is fully available
      sum(last w values of (u-d))/sqrt(w)
```

and, once 16 values are available, the intermittent branch

```text
I_t = 0.5 * sum(the largest 4 values of max(u-d,0) in the latest 16)
```

Before 16 points, only available sustained windows participate.  Raw score is
`R_t=max(G_t,I_t)` over defined branches.  The first raw score is at endpoint
163 after one full four-point window.  Two-point persistence is
`P_t=min(R_(t-1),R_t)`, so the first alarm-eligible endpoint is 164.  There is
no clipping, ECDF, or finite ceiling.  Run maximum is `max(P_t)` and the alarm
is the earliest endpoint whose persistent score is strictly greater than the
threshold.

For each outer fold, transform the 20 inner healthy runs with the frozen
fit-health model and scan calibration, then use their maximum run maximum
(ascending rank 20) as `h`.  Require at least 18 exactly distinct maxima and

```text
rank20-rank19 > 100*max(CUDA deterministic replay error, 1e-7).
```

Any nonfinite value or failed nondegeneracy check is a fold-level no-go.

## Development GO gates

`DEVELOPMENT_GO` requires every condition:

- healthy outer alarms no more than 10/100 and no more than 4/20 in every fold;
- each fault detects at least 85/100 overall and at least 14/20 in every fold;
- each fault's restricted median delay is at most 140 samples;
- each fault's nearest-rank p90 restricted delay is at most 220 samples;
- misses are coded as 340;
- all five thresholds pass the 18-distinct and top-gap rule;
- in every fold, replay the frozen model's complete 20-run inner-healthy CUDA
  score -> frozen-calibrator transform -> scan -> decision path twice; those
  paths must be discrete-identical and their maximum absolute difference must
  not exceed the frozen `1e-6` tolerance;
- training and outer scoring are not replayed;
- cross-device CPU results are reported, but they are not a GO gate.

Any one failure produces immutable `DEVELOPMENT_NO_GO`.  The same candidate
cannot be rescued with a new iteration ID, seed, threshold, endpoint set,
loss, epoch count, fold assignment, or implementation patch.

## Pre-access and completion governance

The governance runner uses strict JSON (`NaN` and infinities forbidden),
exclusive atomic writes and one fixed global claim.  The required order is:

1. verify the history ledger and all required implementation/test files;
2. create the immutable development protocol, SHA sidecar and commit record;
3. run metadata-only protocol, source-byte and CUDA preflight;
4. atomically create the global development claim;
5. atomically commit an access intent enumerating all 400 exact HDF5 dataset
   paths and slices;
6. only then import and call the numeric backend;
7. write result and completion for GO, NO_GO, or failure.

No wildcard selector is permitted.  A prior claim, partial directory, hash
mismatch, missing completion dependency or backend exception fails closed.
After intent, an exception is conservatively recorded as
`ABORTED_AFTER_NUMERIC_ACCESS`; the partial directory is retained.

During development, Mode 1 access is absolutely unauthorized.  A development
GO does not itself open Mode 1.  A separate protocol must first bind the final
model, transforms, scan, target threshold rank and final gates.  Only then may
Mode 1 magnitude-25 IDV1 runs 1--100 be used once to set a threshold.  The
magnitude-25 fault branches stay quarantined.  Invalid calibration leaves
magnitude 75 sealed.  A valid calibration may authorize one irreversible
magnitude-75 confirmation; any pass, fail, partial read or infrastructure
failure permanently consumes it.

## Statistical reporting

The internal Mode 2 result is adaptive exploratory evidence.  It receives no
claim of unbiased out-of-sample performance after the v2--v6 method search.

If a later frozen target calibration uses ascending rank 96 of 100 healthy run
maxima and strict `>`, exchangeability supports only a marginal exceedance
bound of `5/101`; it is not a conditional PAC or automatic cross-mode
guarantee.  In final 100-run healthy confirmation, report an exact one-sided
95% Clopper--Pearson upper bound.  At most 4/100 alarms supports a claim below
0.10; at most 1/100 supports below 0.05.  An operational gate allowing 10/100
does not by itself support either claim.

Report exact per-fault intervals using 100 simulator runs per fault under the
stated independence assumption.  Do not attach a `Binomial(300)` interval to
the pooled three-fault count.  Report delay by fault with misses fixed at 340
and resample only at the run level if uncertainty intervals are added.
