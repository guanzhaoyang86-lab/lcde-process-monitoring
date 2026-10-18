# MetroPT CARED-BC development method

## Status and purpose

This document specifies the development-only CARED-BC method for UCI
MetroPT-3. It is subordinate to
`protocols/metropt_cared_bc_v0_development_charter.json`. It is not a final
calibration or confirmation protocol.

The pre-access clarifications and implementation safeguards in
`protocols/metropt_cared_bc_v0_preaccess_amendment_001.json` are binding and
take precedence where they are more specific. That amendment was frozen before
any MetroPT sensor numeric access.

At this stage, sensor numeric access is permitted only for `fit`,
`dev_healthy`, and the first reported failure, `F1`. Calibration A,
Calibration B, both embargo intervals, and the complete confirmation interval
remain sealed. Timestamp-only integrity auditing is permitted.

CARED-BC means **control-aware residual encoder with directional GLR and
block-conformal CUSUM**. The method targets reported high-stress air leaks. It
does not claim to be a universal detector for every compressor failure type.

## Bound source data

The source is the [UCI MetroPT-3 dataset](https://archive.ics.uci.edu/dataset/791/metropt%203%20dataset.)
(DOI `10.24432/C5VW3R`, CC BY 4.0).

| Artifact | Size | SHA-256 |
|---|---:|---|
| `data/metropt.zip` | 218,381,995 bytes | `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a` |
| `MetroPT3(AirCompressor).csv` | 218,300,507 bytes | `db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24` |
| `Data Description_Metro.pdf` | 81,208 bytes | `b00fac0e8899854078309bef4adaa480d82ecf14dc81c5097c3646973e824127` |

The timestamp-only audit found 1,516,948 rows from
`2020-02-01 00:00:00` through `2020-09-01 03:59:50`. Timestamps are monotone,
there are no duplicates or nonpositive increments, and the modal and median
positive increment is 10 seconds. There are 51,147 gaps longer than 10 seconds
and 331 gaps longer than 60 seconds.

The UCI description contains both 1 Hz and 0.1 Hz statements. Consequently,
all temporal operations below are defined from audited timestamps and not from
either prose sampling-rate statement.

## Locked partitions

Every interval is left-closed and right-open in the dataset's native naive
local time.

| Partition | Interval | Rows | Current numeric access |
|---|---|---:|---|
| Fit | Feb 1–Feb 21 | 151,297 | Allowed |
| Development healthy | Feb 21–Mar 1 | 63,553 | Allowed |
| Calibration A | Mar 1–Apr 11 | 300,769 | Sealed |
| Embargo A | Apr 11–Apr 18 | 46,945 | Permanently excluded |
| Development F1 | Apr 18–Apr 19 | 8,663 | Allowed |
| Embargo B | Apr 19–Apr 22 | 22,834 | Permanently excluded |
| Calibration B | Apr 22–May 21 | 188,937 | Sealed |
| Confirmation | May 21–Sep 1 00:00 | 733,420 | Sealed |
| Post-study tail | Sep 1 00:00–04:00 | 530 | Permanently outside study |

The official report intervals contain 8,663, 2,360, 17,315, and 1,622 rows
for F1 through F4 respectively. Only F1 is a development event. F2, F3, and
F4 are confirmation events and cannot be numerically inspected during
development.

## Causal one-minute representation

### Resampling

Raw observations are aggregated to dataset-native wall-clock minutes. A
completed minute can use only timestamps within that minute.

For each of the seven continuous channels, retain the within-minute median. For
`COMP`, `DV_eletric`, `Towers`, `MPG`, and `Pressure_switch`, retain the
final observation in the completed minute. Sum `Caudal_impulses` within each
minute. These 13 values are the fixed TCN input. Partial minutes retain an
eligibility coverage record, but neither coverage nor missingness masks are
model inputs. Empty minutes remain explicit missing rows; they are not deleted
or compressed out of the timeline.

Any raw gap strictly longer than 60 seconds resets the neural history and all
online statistics. No interpolation may cross such a gap.

The frozen causal input window is 10 minutes. Forecast horizons are 1, 3, and
5 minutes. A score timestamped at minute `t` may compare an observation at `t`
with predictions issued at `t-1`, `t-3`, and `t-5`; it may never use an
observation after `t`.

### Inputs and held-out direct alarms

Let the seven continuous measurements be

\[
x_t=(TP2,TP3,H1,DV_{pressure},Reservoirs,OilTemp,MotorCurrent)_t.
\]

The control state is

\[
s_t=(COMP,DV_{electric},TOWERS,MPG,PressureSwitch)_t.
\]

The source CSV spells `DV_eletric`; code may map that source spelling to the
canonical name `DV_electric`, but the mapping must be explicit. The input also
contains the within-minute sum of `Caudal_impulses`. Coverage masks and
timestamps govern eligibility and reset logic but are not predictors.

`LPS` directly indicates pressure below the existing alarm limit, while
`Oil_level` is another direct status indication. Both are excluded from every
primary-model input, residual, training loss, score, threshold, and selection
operation. They may be opened only with the corresponding partition and only
for a locked comparator or secondary lead-time audit.

## Self-supervised causal encoder

The encoder is a causal TCN with four residual blocks, width 64, kernel size 3,
dilations `(1,2,4,8)`, latent width 32, and dropout 0.1. It predicts the seven
continuous channels at 1, 3, and 5 minutes. It has no auxiliary prediction
head.

Training uses a deterministically control-state-balanced multi-horizon Huber
loss; there is no auxiliary prediction head. AdamW uses learning rate
`1e-3` and weight decay `1e-4`, with at most 50 epochs and early-stopping
patience 8. Seeds are fixed to 17, 29, and 43, and the elementwise median of the
three forecasts is used. Training must execute on CUDA.

The encoder is self-supervised in the narrow, verifiable sense that its targets
are future continuous measurements derived from the allowed fit data; it does
not receive a fault label. “Causal” here means past-only temporal computation,
not causal-effect identification.

## Residual and physics vector

At minute `t`, collect the 21 state-scaled forecast errors for seven channels
and three horizons. Append the following ten predeclared physics quantities:

1. `TP2 - TP3`;
2. `TP3 - Reservoirs`;
3. one-minute TP3 slope;
4. one-minute Reservoirs slope;
5. pressure rise per unit motor current;
6. loaded fraction over the causal 10-minute window;
7. DV-pressure discharge-pulse amplitude;
8. DV-pressure discharge-pulse duration;
9. caudal impulses per loaded minute;
10. five-minute oil-temperature slope.

This gives the residual-physics vector \(r_t\). All scaling locations and
scales are fit-only robust estimates. For each control state, estimate a
Ledoit-Wolf shrinkage covariance \(\Sigma_s\). A state represented by fewer
than 500 fit minutes is pooled to the frequent state with minimum Hamming
distance; lexical state order breaks ties.

## F1 weak-supervision direction

F1 is one failure event, not 8,663 independent events. For each sufficiently
represented state, estimate

\[
v_s=\operatorname{HuberMean}(r_t\mid F1,s)
    -\operatorname{HuberMean}(r_t\mid dev\text{-}healthy,s).
\]

Sparse state directions are shrunk toward the global direction:

\[
\rho_s=\frac{n_s}{n_s+500},\qquad
\widetilde v_s=\rho_s v_s+(1-\rho_s)v_{global}.
\]

F1 may establish this direction and may support the development power gate.
It may not be split into pseudo-independent folds to claim sensitivity or a
confidence interval.

## Detection channels

The generic energy channel is

\[
E_t=r_t^\top(\Sigma_{s_t}+\lambda I)^{-1}r_t.
\]

The one-sided direction-preserving GLR channel is

\[
D_t=\left[
\frac{\widetilde v_{s_t}^\top\Sigma_{s_t}^{-1}r_t}
{\sqrt{\widetilde v_{s_t}^\top\Sigma_{s_t}^{-1}\widetilde v_{s_t}}}
\right]_+.
\]

Separate empirical-rank Gaussian transforms fitted on `dev_healthy` map the
two channels to \(Z_t^E\) and \(Z_t^D\), clipped to `[-4,8]`. The Page
recursions are

\[
G_t^j=\max\{0,G_{t-1}^j+Z_t^j-0.5\},
\qquad j\in\{E,D\}.
\]

The fixed reference drift is 0.5. CUSUM state resets at dataset-native
midnight and after every raw gap longer than 60 seconds. The complete-day run
loss is

\[
M_d=\max_{t\in d,\;j\in\{E,D\}}G_t^j.
\]

Taking the maximum over both channels before threshold calibration accounts
for their multiplicity. No channel may be selected after confirmation.

## Development power gate

Calibration remains sealed unless all conditions hold:

1. TCN `dev_healthy` multi-horizon Huber loss is at least 5% lower than a
   causal persistence predictor.
2. With the provisional boundary equal to the maximum complete-day CUSUM
   maximum in `dev_healthy`, F1 is detected.
3. F1 first-alarm latency is at most 60 minutes.
4. The F1 maximum CUSUM is at least 1.25 times the provisional boundary.
5. Causality, gap-reset, partition-access, and direct-alarm-exclusion tests all
   pass.
6. CPU and CUDA development alarm decisions agree, with a numerical margin
   larger than the parity tolerance that will be frozen in the final protocol.

Failure stops the method before any calibration numeric access. Failure does
not authorize opening sealed data or changing to another method family under
this charter. Implementation bugs and training-instability defects may be
fixed using only the three open development partitions and synthetic tests.

## Future block-conformal boundary

This section defines the intended calculation but does not grant calibration
access. After the power gate passes, a final analysis protocol, model hashes,
test results, and a calibration access-intent record must be frozen first.

For \(n\) qualified, non-overlapping calibration days and target daily
any-alarm probability \(\alpha=0.05\), set

\[
k=\lceil(n+1)(1-\alpha)\rceil,\qquad h=M_{(k)}.
\]

At least 39 qualified days are required. If all planned 70 days qualify,
\(k=68\): the raw boundary is the third-largest calibration-day maximum, and
the rank-based marginal upper bound is

\[
\frac{71-68}{71}=0.0422535.
\]

The final protocol must freeze a numerical guard and `nextafter` boundary rule
before confirmation. Alarms use a strict greater-than comparison.

The finite-sample rank statement is exact only if calibration and future day
blocks are exchangeable. One train observed over time does not establish that
assumption. Non-overlapping days, control-state conditioning, gaps, and
embargoes reduce dependence; they do not provide distribution-free validity
under arbitrary temporal drift. Under weak dependence, the result should be
described as approximate. See
[Chernozhukov, Wüthrich, and Zhu (2018)](https://proceedings.mlr.press/v75/chernozhukov18a.html).

## Future locked confirmation

Confirmation cannot start until the calibration certificate, raw and effective
boundaries, model artifacts, code, baselines, ablations, and all success gates
are hashed and frozen. A durable access ledger must be created before any
confirmation numeric value is parsed.

The primary gates are:

- F2, F3, and F4 all detected inside their official report intervals;
- per-event first-alarm latency no greater than 60 minutes;
- median first-alarm latency no greater than 30 minutes;
- alarm incidence on qualified healthy confirmation days no greater than 5%;
- its one-sided 95% Clopper-Pearson upper bound no greater than 10%;
- healthy monitored-time alarm fraction no greater than 1%; and
- exact CPU/CUDA alarm-decision agreement.

A qualified healthy day cannot overlap a reported failure or its following
72-hour recovery embargo. Secondary results include official interval overlap,
false alarms per 100 monitored hours, monitored time between false alarms,
lead relative to the first LPS activation, and fixed sensor/physics
attributions.

The inferential units are days for false alarms and maintenance-report events
for detection. Rows and overlapping windows are never treated as independent
replicates. Even if all three events are detected, the one-sided 95% exact
lower bound for an event-detection probability is only

\[
0.05^{1/3}=0.3684.
\]

Accordingly, a successful result is a locked three-event forward feasibility
confirmation, not a precise estimate of population sensitivity. A failed
confirmation is terminal for this frozen analysis and cannot be followed by
same-data retuning.

## Locked comparisons

All baselines and ablations intended for confirmation must be implemented and
frozen before the confirmation ledger is created. Their thresholds use the
same complete-day alarm budget.

Baselines are the existing LPS/low-pressure rule, the published sparse
autoencoder plus sequential alarm, control-state PCA-SPE+CUSUM, an Isolation
Forest on physics features, and a causal TCN energy-CUSUM without control-state
conditioning. The introductory sparse-autoencoder study is
[Davari et al. (2021)](https://doi.org/10.1109/DSAA53316.2021.9564181).

The fixed ablations remove, one at a time, control conditioning, physics
features, the direction channel, the TCN encoder, CUSUM accumulation, or the
complete-day block-conformal boundary. Confirmation results may compare these
pre-frozen variants but may not be used to revise them.
