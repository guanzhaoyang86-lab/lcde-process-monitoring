# GPU MiniROCKET Log-Odds CUSUM v8: implementation and execution protocol

## Status and activation

`gpu-minirocket-logodds-cusum-v8-fixed` is the single, pre-result fallback
defined by `gpu_minirocket_logodds_v8_contingency.{md,json}` before FARM v7
opened its formal data.  It is activated only by the exact categorical v7
result `DEVELOPMENT_NO_GO`.  The v8 runner verifies and hash-binds the v7
formal result, its append-only pre-refit decision evidence, and completion
record; no v7 numerical metric is used to alter v8.

Development uses the already-consumed Extended TEP Mode2 magnitude-50 pool
only.  It remains adaptive exploratory feasibility evidence.  Mode1 and
Rieth are not authorized.  A v8 `DEVELOPMENT_NO_GO` ends method search on the
current development pool.

## Upstream provenance

The transform is vendored from the official timeseriesAI/tsai repository,
tag `v1.0.1`.  That tag is annotated: tag-object SHA
`2bca845b2905aa2cfcb03a2e1615bdb196b821e5`, peeled commit SHA
`f9ecc6b6df798946fa5fb290ac2ade9a00815289`.  The exact vendored source,
Apache-2.0 license, local modifications, and synthetic golden artifact are
part of the implementation manifest and therefore immutable after protocol
freeze.

## Exact data contract

The only authorized source is
`data/extended_tep_dtu_mode2/TEP_Mode2.h5`, SHA-256
`8462c0249761ae762a99028c7f8391620b68c69aaea4ecc961fe482799c974b4`,
size 23,897,125,720 bytes.  Exactly IDV1/3/9/15, runs 1--100, magnitude 50,
`SingleFault/SimulationCompleted` are selected.  IDV1 uses rows `[100,600)`;
faults use `[440,940)`.  Each exact selector reads only
`processdata[start:stop,0:53]`, `idv_init[0:3,0:28]`, the group seed, and
object shape/dtype/type.  The time column is checked then removed, yielding
500 by 52 finite float64 values.  Root labels, other rows, wildcard paths,
other branches, Mode1, and Rieth are forbidden.

The collision-safe five-fold map is unchanged (canonical SHA-256
`64a5aee8bc61f0865072e05be3f4cdd25f1e6f39e4061b44add2ab7d1e08eb13`).
For fold `f`, outer is fold `f`, inner healthy is `(f+1) mod 5`, and the other
three folds are fit data.  Inner fault values are never used.

## Frozen method

All 52 channels are robust-normalized from 60 fit-healthy runs only using
float64 median and `max(1.4826*MAD,1e-6)`, clipped to `[-10,10]`, then cast
to float32.  Inclusive causal windows have shape `[52,128]`.

The MiniROCKET transform uses seed 20260912, all 84 length-nine kernels,
requested 10,000 and effective 9,996 PPV features, at most 32 dilations per
kernel, at most 9 input channels, no LSAZ, strict `convolution > bias`, and
batch size 128.  Biases are fitted only from the exactly ordered 20,400
fold-training windows.  The 9,996 feature means and population standard
deviations are fitted in float64 from those same windows with a `1e-6` floor.

The four-class linear head is zero-initialized and trained in one CUDA full
batch using weighted cross entropy plus `(1e-4/2)||W||^2`, with LBFGS
`lr=1`, `max_iter=100`, `max_eval=125`, history 20, strong-Wolfe line search,
gradient tolerance `1e-7`, and change tolerance `1e-9`.  Class weights are
`H=85/109` and each fault `85/77`.  Training fails closed on non-finiteness or
if the final objective exceeds the zero-initialization objective by more than
`1e-10`.

For each run, eight prefix endpoints 128--156 step four and dense endpoints
160--499 are scored.  CPU-float64 fault-vs-healthy log odds are centered by
the run prefix median.  A robust center/scale is fitted only from all 20,400
fit-healthy dense values.  The final path is the maximum of a one-sided CUSUM
with drift 0.25 and, from endpoint 175, one-half the sum of the largest four
positive standardized values in the inclusive trailing 16 endpoints.

The inner threshold is the maximum of 20 inner-healthy run maxima and alarms
use strict `>`.  Numeric validity requires at least 18 distinct maxima and
`rank20-rank19 > 100*max(same-CUDA replay error,1e-6)`.  Complete inner
scoring/calibration/scan is replayed twice.  CPU scoring is report-only.

## Development gates and GO-only refit

All five folds must be numerically valid.  Healthy false alarms must be at
most 10/100 and at most 4/20 in each fold.  Each of F3/F9/F15 must reach at
least 0.85 overall and 0.70 per fold.  Across all 300 fault delays (miss=340),
median must be at most 140 and nearest-rank p90 at most 220.

The append-only `development_decision_before_final_refit.json` is persisted
immediately after the gates determine GO/NO_GO and before any GO-only refit.
Only GO may fit a final state from the same 100 Mode2 runs per class.  This
does not authorize Mode1 access or target-domain calibration.

## Fail-closed execution

The protocol is written exclusively as canonical JSON plus one LF, with
separate file-bytes and canonical-JSON SHA-256 domains.  Existing partial or
complete protocol bundles cannot be repaired or overwritten.  Before the
one-use project-global claim, the runner revalidates every implementation
hash, source hash, activation evidence, synthetic/golden tests, RTX 5090
CUDA, and the production backend against the persisted protocol.  A meta-path
blocker rejects even caught `h5py` import attempts during this handshake.
The test subprocess is isolated from site initialization, pytest plugins,
conftest files, inherited pytest/Python options, and pre-existing bytecode;
the protocol requires exactly 66 passed tests with no skip, deselection,
xfail, or error.  Every trusted local scoring module is compiled directly
from the same source bytes whose SHA-256 is in the implementation manifest.

The exact validated backend module object is reused after the exclusive claim
and committed access intent.  Its atomic access-state record distinguishes
pre-open, open, slice-read, training, and completion phases and must report
exactly 400 process-data plus 400 declaration slices, zero Mode1 opens, and
zero out-of-scope reads.  Changing output roots cannot create a second claim.
The fixed run directory and its `artifacts` child are exact protocol fields.
Immediately before any source/HDF5 numeric access the backend creates the
permanent, kernel-exclusive
`v8_fixed_formal/numeric_execution.claim.json`; a complete or partial token,
artifact directory, failure, completion, or result permanently forbids replay.
The formal process also verifies isolated/no-site/no-bytecode Python flags,
the exact interpreter and conda prefix, repository cwd, script/argv, and
unbuffered stdout/stderr before creating the global claim.

Third-party resolution keeps the pinned conda `site-packages` ahead of project
paths.  Torch, NumPy, and lazily imported h5py must originate there; h5py is
additionally version/API-attested after the committed intent.  Each authorized
HDF5 group/dataset must be an in-file hard link whose owning file is the exact
hash-bound Mode2 file, so external and soft links are rejected.

The protocol also binds a canonical implementation-tree identity and the
exact formal environment: Python 3.11.14, NumPy 2.4.0, h5py 3.15.1, PyTorch
2.10.0.dev20251124+cu128, CUDA 12.8, cuDNN 91002, NVIDIA driver 595.97, and
the NVIDIA GeForce RTX 5090 Laptop GPU (capability 12.0, 25,650,855,936
bytes).  Any drift fails before the data claim.

## Reproducible entry points and model loading

Run these commands separately and in order from the repository root.  The
explicit `conda run` wrapper is part of the formal execution contract because
it supplies the DLL search path required by the `torch5090` environment.
`freeze-development` is exclusive and must happen only after implementation
review; `development` consumes the one-use claim.

```powershell
conda run --no-capture-output -n torch5090 python -I -S -B -u scripts\run_gpu_minirocket_logodds_v8.py freeze-development
conda run --no-capture-output -n torch5090 python -I -S -B -u scripts\run_gpu_minirocket_logodds_v8.py preflight
conda run --no-capture-output -n torch5090 python -I -S -B -u scripts\run_gpu_minirocket_logodds_v8.py development
```

`mate_spc_minirocket_v8.backend.load_frozen_fold_pipeline(path)` strictly
loads a fold `freeze_manifest.json` and returns the complete pipeline,
calibration, and threshold.  Authoritative outer scoring itself uses that
reloaded bundle.  After GO,
`mate_spc_minirocket_v8.backend.load_final_refit_pipeline(path)` strictly
loads the final `manifest.json`.  Both loaders use `numpy.load(...,
allow_pickle=False)`, enforce exact members, shapes, dtypes, metadata, and
file hashes, and reconstruct the vendored MiniROCKET transform plus robust
normalizer, feature standardizer, and linear head.  The GO writer requires an
exact log-odds roundtrip after reload before reporting a usable artifact.
