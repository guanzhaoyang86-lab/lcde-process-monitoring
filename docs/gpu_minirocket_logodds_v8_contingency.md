# GPU MiniROCKET Log-Odds CUSUM v8: contingent method charter

## 1. Status and authorization boundary

- Canonical candidate ID: `gpu-minirocket-logodds-cusum-v8-fixed`.
- This document is a **design-level contingent method charter**, frozen before any formal v7 data access or v7 numerical result is viewed.
- The only activation trigger is the exact categorical result `farm-rankcusum-v7-fixed = DEVELOPMENT_NO_GO`.
- The charter does not authorize opening any H5 file, implementing the method, executing an experiment, creating a checkpoint, or making a numerical performance claim.
- Before any data access, v8 still requires a reviewed implementation snapshot, synthetic golden tests, environment capture, immutable hashes, and a separate pre-access execution protocol.
- A v7 `GO`, invalid/incomplete v7 result, implementation failure, or any other observation does not activate v8. No v7 metric, fault-specific result, plot, or failure pattern may change this charter.
- v8 is one candidate with no rescue grid. A formal v8 `DEVELOPMENT_NO_GO` terminates method search on the current development pool.

## 2. Why this is the sole fallback

The sole fallback is upstream-recognized GPU MiniROCKET followed by a fixed linear four-class head and a run-relative log-odds sequential monitor. It is materially different from v7: v7 ranks each raw channel against healthy marginal eCDFs, whereas v8 applies a large fixed bank of dilated convolutions to multivariate causal windows and learns only a convex linear classifier on the resulting PPV features.

Alternatives are excluded in advance:

1. A handwritten ROCKET implementation has a larger stochastic and implementation-validation surface without a compensating protocol advantage.
2. An enhanced FARM is not an independent fallback after v7 because it retains the same marginal-rank monitoring core.
3. A trained TCN/Transformer adds optimization variance and a tuning surface that are unnecessary for a one-candidate fallback.

The original MiniROCKET authors identify MiniROCKET as the usual default variant and explicitly point to the tsai GPU implementation. Primary references are the [MiniROCKET paper](https://doi.org/10.1145/3447548.3467231), the [authors' repository](https://github.com/angus924/minirocket), and the [tsai source](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/MINIROCKETPlus_Pytorch.py).

## 3. Upstream source, vendoring, and license

- Fixed upstream repository: `https://github.com/timeseriesAI/tsai`.
- Fixed tag: `v1.0.1`.
- Fixed upstream path: `tsai/models/MINIROCKETPlus_Pytorch.py`.
- Vendor only `MiniRocketFeaturesPlus` and the minimal pure-PyTorch helpers it transitively requires. Do not install or import the tsai package at runtime.
- Preserve the upstream Apache-2.0 license and required notices, identify every local modification, and add the vendored source and license to the implementation snapshot.
- The full commit SHA for tag `v1.0.1` has **not been asserted by this design charter**. Before implementation, resolve the tag from the official repository, verify that it is an annotated or lightweight tag pointing to a commit, and bind the resolved full commit SHA plus SHA-256 hashes of the exact source and license bytes. Do not substitute a branch head or guess a SHA.
- If the tagged package metadata is incompatible with the installed PyTorch build, compatibility is achieved only by vendoring the necessary pure-PyTorch module. The numerical algorithm and frozen settings below may not be rewritten.

## 4. Data isolation and five-fold roles

Only Mode2 development runs 1--100 are in scope. Mode1, Rieth, sealed calibration/final-confirmation data, and all other sources remain unread and unauthorized.

The immutable run blocks are:

```text
fold 0: 8,14,15,20,26,31,37,42,43,48,53,58,63,68,73,79,84,90,95,98
fold 1: 3,6,10,16,21,27,32,38,44,49,54,59,64,69,74,76,80,85,91,96
fold 2: 4,9,11,17,22,28,34,39,45,50,55,60,65,70,75,81,86,88,92,97
fold 3: 5,12,18,23,24,29,33,35,40,46,51,56,61,66,71,77,82,87,93,99
fold 4: 1,2,7,13,19,25,30,36,41,47,52,57,62,67,72,78,83,89,94,100
```

For outer fold `f`, outer runs are block `f`, inner healthy runs are block `(f+1) mod 5`, and fit runs are the other three blocks. Thus each fold has exactly 60 fit runs per class, 20 inner healthy runs, and 20 outer runs per class. Inner-fault values are forbidden. The established canonical run-to-fold mapping SHA-256 is `64a5aee8bc61f0865072e05be3f4cdd25f1e6f39e4061b44add2ab7d1e08eb13`.

All rows/windows from one run remain in one role. Endpoints are never independent split units.

## 5. Input and fit-only normalization

- Use all 52 Mode2 channels in their stored order.
- At endpoint `t`, input is the inclusive causal window `x[t-127:t+1]`, shaped `[52,128]`; no future sample is visible.
- Fit each channel's location and scale using only the 60 fit-healthy runs and all 500 time points per run.
- Location is the float64 median. Scale is `max(1.4826 * median(abs(x - median(x))), 1e-6)`.
- Normalize `(x-location)/scale`, clip to `[-10,10]`, verify finiteness, then cast to float32 for MiniROCKET.
- Save the 52 locations, scales, ordering, and hashes in the future implementation snapshot.

## 6. Fixed MiniROCKET transform

- Global seed: `20260912` for Python, NumPy, CPU PyTorch, and all CUDA generators.
- Enable deterministic PyTorch algorithms; set cuDNN deterministic true and benchmark false; set the deterministic cuBLAS workspace configuration to `:4096:8` before CUDA initialization.
- Kernel length is 9. Use all `C(9,3)=84` zero-sum kernels: three coefficients are `+2` and six are `-1`.
- Requested feature count is 10,000; upstream divisibility yields exactly 9,996 PPV features.
- `max_dilations_per_kernel=32`, sequence length 128, and the upstream algorithm therefore uses its fixed dilation allocation over integer dilations 1 through 15.
- `max_num_channels=9`, `add_lsaz=False`, and preserve the upstream padded/unpadded feature allocation.
- Bias quantiles use the upstream low-discrepancy golden-ratio sequence. PPV uses the strict comparison `convolution > bias`; ties are not positive.
- Preserve the tagged upstream random generation of channel subsets and bias-sampling indices under seed `20260912`; serialize the resulting kernel/dilation/channel-mask/bias arrays and their hashes. Do not regenerate them during scoring.
- Fit biases from the 20,400 fold-training windows only, in the deterministic ordering specified below. Use the upstream default chunk size `84 * 15 = 1260`, sampled without replacement under the fixed generator. No inner or outer window participates.
- Feature extraction batch size is 128, `drop_last=False`, float32 on CUDA. Concatenation order is deterministic.

Before any H5 access, the implementation must pass a golden handshake against the unmodified tagged upstream class on a fixed synthetic tensor:

1. Save the upstream-generated kernel, dilation, channel-mask, bias, and PPV outputs as a no-pickle golden NPZ with provenance hashes.
2. Vendored CPU output and buffers must match exact integer/boolean values and floating values within absolute and relative tolerance `1e-7`.
3. Hand-calculated scalar fixtures must cover padding, unpadding, quantiles, and strict `>` ties.
4. Vendored CUDA output must match the CPU golden within absolute and relative tolerance `1e-6`.
5. Two same-CUDA complete replays must have maximum absolute error at most `1e-6`.

Failure of any handshake is an implementation blocker, not a reason to change the method.

## 7. Fold-training examples

Training windows are concatenated in the stable order `(class order, source/run ID, endpoint)`, where class order is `H,F3,F9,F15`.

- Healthy-class label `H`:
  - 60 healthy fit runs at 85 endpoints `160,164,...,496`: 5,100 examples.
  - Each of F3, F9, and F15 fit runs at eight pre-fault endpoints `128,132,...,156`: 1,440 examples.
  - Total `H`: 6,540.
- Each fault-class label F3, F9, or F15:
  - Its 60 fit runs at 77 mature endpoints `192,196,...,496`: 4,620 examples per class.
- Fault endpoints `160,164,...,188` are ignored in training and bias fitting.
- Total training examples: 20,400.

After the frozen transform, compute each of the 9,996 feature means and population standard deviations over all 20,400 training examples in float64. Apply `max(std,1e-6)`, standardize, verify finiteness, and cast to float32. These statistics are fit-fold-only state.

## 8. Fixed linear head and optimizer

- Head: `Linear(9996,4,bias=True)` with logits ordered `H,F3,F9,F15`.
- Initialize all 39,988 weights and biases to zero.
- Objective is PyTorch weighted cross-entropy with mean reduction plus `(1e-4/2) * ||W||_F^2`; the bias is not penalized.
- Class weights are exactly `H=85/109` and `F3=F9=F15=85/77`. This equalizes the total endpoint weight of each class.
- Optimize on CUDA in one full batch with PyTorch LBFGS: `lr=1.0`, `max_iter=100`, `max_eval=125`, `history_size=20`, `tolerance_grad=1e-7`, `tolerance_change=1e-9`, `line_search_fn='strong_wolfe'`.
- There is no validation selection, hyperparameter grid, epoch choice, early stopping, dropout, hidden layer, batch normalization, or alternate optimizer.
- Training fails closed unless initial/final loss, every parameter, and every gradient are finite and final objective is no greater than its zero-initialization objective (within `1e-10` numerical tolerance).

## 9. Dense log-odds monitor

Score eight prefix endpoints `128,132,...,156` and every monitor endpoint `160,161,...,499`. For logits `l_H,l_3,l_9,l_15`, compute on CPU in float64:

```text
q_it = logsumexp(l_3,l_9,l_15) - l_H
b_i  = median(q_i,128, q_i,132, ..., q_i,156)
v_it = q_it - b_i
```

Using only the 60 fit-healthy runs and all dense monitor endpoints, pool the `60*340` values `v_it` and calculate:

```text
m = median(v)
s = max(1.4826 * median(abs(v-m)), 1e-3)
u_it = (v_it-m)/s
```

Medians use sorted float64 values and the arithmetic mean of the two central values for even counts.

For every run, reset `C_i,159=0` and calculate:

```text
C_it = max(0, C_i,t-1 + u_it - 0.25),                    t=160,...,499
I_it = 0.5 * sum(largest 4 values of max(u_i,t-15:t,0)), t=175,...,499
S_it = C_it,                                              t=160,...,174
S_it = max(C_it,I_it),                                    t=175,...,499
M_i  = max_t S_it
```

The slice in `I_it` contains 16 endpoints including `t`. There is no persistence rule, clipping, eCDF transform, or alternative scan. The first eligible alarm is endpoint 160.

## 10. Inner threshold and numerical validity

- Compute the 20 inner-healthy run maxima with the fit-fold transform, classifier, and calibration.
- The threshold `h_f` is rank 20, i.e. the maximum of these 20 finite maxima.
- A run alarms only when `S_it > h_f`; equality is negative. Alarm time is the earliest strict crossing.
- Detection delay is `alarm_time-160`; a miss receives delay 340.
- Require at least 18 distinct inner maxima.
- Require `rank20-rank19 > 100 * max(full_same_cuda_replay_error,1e-6)`.
- Repeat the complete inner CUDA score, float64 log-odds, fit-health center/scale, and scan twice. Any nonfinite result or violation makes the fold numerically invalid and forces overall `DEVELOPMENT_NO_GO`.
- CPU execution is a reference report only; the formal result is the deterministic CUDA path.

## 11. Frozen outer gates

The single v8 candidate is `GO` only if every condition below is true across the five untouched outer folds:

- All five folds are numerically valid.
- Healthy alarms are at most 10 of 100 overall and at most 4 of 20 in every fold.
- F3, F9, and F15 each have detection rate at least 0.85 over 100 outer runs and at least 0.70 in every 20-run fold.
- Pooling the 300 fault-run delays with misses assigned 340, median delay is at most 140 and nearest-rank 90th percentile is at most 220.

No component metric can compensate for a failed gate. Do not create confidence-interval, diagnostic, or fault-specific rescue rules after inspection.

## 12. Post-GO boundary and expected resources

Only after all five development folds pass may the same method be refit with 100 authorized runs per class. That refit is a later protocol stage and does not authorize opening Mode1 or any sealed confirmation set under this charter.

No new runtime dependency is required beyond NumPy and CUDA-enabled PyTorch. The 20,400-by-9,996 float32 development feature cache is about 0.76 GiB; the analogous 100-run cache is about 1.27 GiB. With extraction batch size 128, expected peak memory is below 8 GiB, so an RTX 5090 is sufficient. Estimated implementation plus synthetic verification is 10--16 working hours; a formal five-fold development run is expected to take approximately 20--60 minutes, recorded as an estimate rather than a performance claim.

## 13. Required future artifacts before activation

The following must exist and be reviewed before v8 can read data:

1. Full tag-to-commit verification and vendored-source/license SHA-256 manifest.
2. Implementation commit/tree hash and environment lock, including CUDA, driver, PyTorch, NumPy, and GPU identity.
3. No-pickle synthetic golden artifact and passing CPU/CUDA test report.
4. Machine-validated constants against the companion canonical charter JSON.
5. A separate pre-access protocol naming exact data paths, hashes, command, output directory, failure behavior, and one-shot authorization.

This charter deliberately contains no observed v7 or v8 metrics.
