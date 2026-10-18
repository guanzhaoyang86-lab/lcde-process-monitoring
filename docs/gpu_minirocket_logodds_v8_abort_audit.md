# GPU MiniROCKET Log-Odds v8 terminal abort audit

## Outcome

The frozen v8 formal development invocation did **not** produce a scientific
`DEVELOPMENT_GO` or `DEVELOPMENT_NO_GO` result.  It terminated with the
append-only decision
`ABORTED_DURING_NUMERIC_BACKEND_ACCESS_STATE_UNCERTAIN` on 2026-09-09.

This is an implementation/handshake abort, not evidence for or against the
MiniROCKET log-odds monitoring hypothesis.  No fold, score path, fitted model,
or `development_result.json` was created.

## Frozen identities

- protocol id: `gpu-minirocket-logodds-v8-mode2-development`
- protocol canonical JSON SHA-256:
  `6271b00b34c585ac08c3d29970b622b38c43b14cac6081d9c47bdd9418c1ed15`
- protocol file SHA-256:
  `aecbf8a73b628851e8c0230388c9b3180465ad9320e457f0da0a36a4d7d7ea93`
- runner SHA-256:
  `452172c79789b6b90bf4a5fc0683dcf57f87db546e5173bb8af2a9f8d35120df`
- backend SHA-256:
  `9d21ebb4026c3264b55dd262b3243bd194a333fb0ae2d471d6d2d8bf1e4323b5`
- global claim file SHA-256:
  `91592644b67b44aeeefd7d50953e933a678231b86839e6138031180c70ec878a`
- access-intent file SHA-256:
  `67240239701f4cc10376126a8ef98f74388e4b94ad9ab94ee13dfe3b6c12f9bb`
- failure file SHA-256:
  `ed2e653b845293d08f5cc7e239ca96b10e17461d3ea02ab4bc93aa6c3ce3afd4`

The preclaim GPU suite passed 66/66 tests, and the frozen preflight passed the
RTX 5090/CUDA/upstream golden checks.  Those checks are software evidence only;
they are not experimental performance evidence.

## Root cause

The runner-created access intent contains the correct v7 activation hashes
inside `activation_evidence`, but omits the two redundant top-level fields
required by the production backend:

- `activation_result_file_bytes_sha256`
- `activation_decision_file_bytes_sha256`

The backend therefore raised `ValueError: access intent semantic binding
mismatch` while validating the committed authorization bundle.  The unit
tests missed the defect because the backend fixture manually inserted both
fields, while the runner test used a simulated backend; no producer-consumer
test passed the runner's actual committed intent to the real backend validator.

## Numeric-access evidence

The production order is:

1. validate the committed authorization;
2. create the permanent numeric-execution claim;
3. atomically create the backend artifact directory and access-state record;
4. import/open HDF5 and decode the authorized blocks.

The exception occurred in step 1.  The numeric-execution claim, backend
artifact directory, backend access-state record, folds, score paths, and models
do not exist.  The preflight only streamed the 23.897 GB source as raw bytes to
verify its SHA-256; it did not import h5py, enumerate HDF5 structure, or decode
numeric values.  The formal invocation did not reach HDF5 access.

Mode1 magnitude-25 and magnitude-75 cells remain numerically unread and
`SEALED`/`QUARANTINED` exactly as recorded before v8.  Rieth was not accessed
by v8 and remains excluded from new tuning, but it was consumed in earlier
project work and must not be described as globally unseen.

## Governance consequence

The fully committed global claim is permanent.  The frozen contract forbids a
same-candidate rerun after any claim residue, including a repaired runner, an
alternate output root, or a cosmetic v8.1/v9 rename.  The claim, intent,
failure, and completion evidence must remain unchanged.

The recommended next study is a genuinely new hypothesis developed on a new
independent source (or freshly generated simulator seeds), with all prior
v2-v8 outcomes and this abort disclosed.  Only after that method is frozen and
passes its development gate may a new, separately preregistered protocol
consider one-time Mode1 access.

## Evidence

- `../outputs/gpu_minirocket_logodds_v8/development_access.claim.json`
- `../outputs/gpu_minirocket_logodds_v8/v8_fixed_formal/access_intent.json`
- `../outputs/gpu_minirocket_logodds_v8/v8_fixed_formal/preflight.json`
- `../outputs/gpu_minirocket_logodds_v8/v8_fixed_formal/failure.json`
- `../outputs/gpu_minirocket_logodds_v8/v8_fixed_formal/completion.json`

