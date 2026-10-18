# pyTEP R2024b reproduction compatibility audit

## Decision

`STOPPED_BEFORE_BANK_GENERATION`

The frozen SoftwareX pyTEP repository running under MATLAB/Simulink R2024b is
classified as `external_simulator / incompatible_with_official_H5` for this
study. Its output must not be pooled with, substituted for, or described as a
same-domain reproduction of the official Extended TEP HDF5 archive.

Only one 100-hour compatibility trajectory was generated:

`outputs/pytep_reproduction/idv3_run1_seed1105632.mat`

It used Mode 1, IDV3 magnitude 0.5 at hour 30, seed 1105632, sample period
0.05 hour, and the published structure flag 227. The corresponding official
comparison run was an already-consumed diagnostic run; no sealed Mode1 or
Mode2 partition was opened for this audit.

## Observed incompatibility

After aligning the generated time-zero row and the official 500-by-52 analytic
slice, the flattened correlation was misleadingly high because TEP variables
have very different absolute scales. The channel-wise checks rejected
compatibility:

- median channel-standardized NRMSE: approximately 1.43;
- median zero-lag channel correlation: approximately -0.005;
- channel correlations were centered near zero rather than near one.

These errors are orders of magnitude larger than a permissible cross-version
numeric drift. Repeating 20 long runs cannot repair a generator-level mismatch,
so the planned fresh bank was not generated.

## Implementation evidence

- pyTEP commit: `7caf477b049a488f819e80802ed4fef85e6a435b`.
- MATLAB R2024b and Simulink 24.2 loaded the original `TElib.mdl` and
  `MultiLoop_mode3.mdl` without saving over them.
- Plant S-function: `temexd_mod`.
- Plant parameters: `xInitial.signals(1).values,seed,227`.
- `loadSimEnvironment.m` supplies Mode 1, `Ts_base=5e-4`, `Ts_save=0.05`, and
  a default seed that must be overwritten after every reset.
- The backup model uses structure flag 243, which disables measurement noise,
  and is not an acceptable replacement.
- The upstream package targets Python 3.7 and MATLAB R2019b and reports testing
  only around R2019a/R2020 releases; R2024b is outside that declared matrix.

The governed MATLAB wrapper was corrected to place `seed`, `Ts_save`, and
`idv_init` in the base workspace before the initialization scripts run. This
fix prevents silent workspace misuse, but it does not reverse the observed
runtime/generator domain shift and does not authorize bank generation.

## Data-governance consequence

The generated MAT file was never used for training, calibration, thresholding,
model selection, or evaluation. If this simulator is studied later, it must be
treated as a separately calibrated simulator-domain transfer experiment. It is
not independent real-process validation and is not an official-H5 replicate.
