# PCF-CycleRank-Scan v6 terminal handshake-abort audit

## Audit conclusion

The single frozen v6 candidate `pcf-cyclerank-scan-v6-fixed` terminated at
the protocol-validation handshake on 2026-09-09.  This is a terminal
infrastructure/governance abort, not a `DEVELOPMENT_GO` or
`DEVELOPMENT_NO_GO` scientific result.  The global claim and exact access
intent were committed, so the frozen no-rescue rule forbids another v6 run,
including a cosmetically renamed iteration.

No HDF5 container was opened through an HDF5 library, no dataset value was
decoded, no selector window was materialized, and no fitting, scoring or
training began.  The governance preflight did stream the source file as raw
bytes solely to verify its SHA-256; this is not an HDF5 parse or numeric data
access.  Mode1 magnitude 25 and 75 remain sealed and unconsumed.

This audit updates only this postmortem document and, as an explicit
postmortem exception, the historical access ledger.  It does not edit the
frozen v6 protocol, its sidecar or commit, the claim, intent, failure,
completion, or any executable v6 method/runner/test artifact.  Because the
ledger itself was hash-bound at freeze, its intentional postmortem update also
makes that old implementation bundle unverifiable for another v6 run.

## Durable event record

| UTC time | Durable event | Evidence |
|---|---|---|
| 2026-09-09 06:49:21.133532 | Frozen protocol committed | `protocols/pcf_cyclerank_v6_mode2_development.json.commit.json` |
| 2026-09-09 06:51:04.135865 | Exclusive global v6 claim committed | `outputs/pcf_cyclerank_v6/development_access.claim.json` |
| 2026-09-09 06:51:04.135865 | Governance preflight completed; 120 tests passed; RTX 5090 present; HDF5 library not imported | `outputs/pcf_cyclerank_v6/v6_fixed_formal/preflight.json` |
| 2026-09-09 06:51:04.146405 | Exact 400-selector Mode2 access intent created | `outputs/pcf_cyclerank_v6/v6_fixed_formal/access_intent.json` |
| 2026-09-09 06:51:04.154452 | Access-intent bundle committed | `outputs/pcf_cyclerank_v6/v6_fixed_formal/access_intent.json.commit.json` |
| 2026-09-09 06:51:05.789191 | Runner rejected the backend protocol-hash response | `outputs/pcf_cyclerank_v6/v6_fixed_formal/failure.json` |
| 2026-09-09 06:51:05.798374 | Terminal completion recorded | `outputs/pcf_cyclerank_v6/v6_fixed_formal/completion.json` |

The frozen protocol file has 17,127 bytes and ends in one LF byte.  Its exact
persisted-file SHA-256, bound by the sidecar and commit, is
`2b3cf4268b6f479b9944d2154aaec25d20b019a161143e78c783dd51fce98b7a`.
The SHA-256 of the same canonical JSON bytes without the terminal LF is
`9e7fcdaf9e84b6c3d062375ba3dbc09444c7dcb569c2ef5aef2d92fe8200d4ff`.

The claim binds access-plan SHA-256
`74b19927fee7d31f33c3b64d62a0c5235b58ad41e5d2d7d28b583a6ff4e43466`
and fold-map SHA-256
`64a5aee8bc61f0865072e05be3f4cdd25f1e6f39e4061b44add2ab7d1e08eb13`.
The access-intent document SHA-256 is
`52678c66be7e940d80357541e45b6ddb86829680f95e6490d8080f461fd1cd53`.

## Root cause

The runner used the name `protocol_sha256` for the exact frozen file hash,
which includes its terminal LF.  The backend's in-memory
`canonical_document_sha256` correctly hashed the canonical serialized JSON
without that file terminator.  The runner then required these two different
hash domains to be equal.  Consequently, a valid backend response containing
`9e7f...d4ff` was compared with the frozen-file value `2b3c...8b7a` and
rejected with `ValueError: numeric backend rejected the frozen v6 protocol`.

This was a hash-domain contract defect.  It was not a protocol-content
mismatch, source-identity mismatch, CUDA numerical failure, HDF5 error, model
failure, or statistical gate failure.

## Exact access disposition

The completion uses the conservative label `ABORTED_AFTER_NUMERIC_ACCESS`
because its implementation maps every failure after access-intent commitment
to that label.  The label records authorization-state timing; it does not prove
that a numeric read occurred.  Positive evidence for this incident shows:

- the access intent was committed and enumerated 400 exact Mode2 magnitude-50
  selectors with no wildcards;
- backend import occurred only after intent;
- protocol response validation failed before `backend.cuda_preflight()` and
  before `backend.run_development()`;
- there is no `backend_cuda_preflight.json`, `development_result.json`,
  artifact directory, fold result, checkpoint, training history, scan
  calibrator, or final refit in the v6 formal directory;
- HDF5 library import/open and numeric decode counts are zero;
- Mode1 access authorization is false and Mode1 HDF5 open/value-read count is
  zero.

The intent and claim are nevertheless irreversible governance artifacts.  The
same candidate cannot be retried, patched in place, or revived under another
v6 directory or iteration ID.  The postmortem ledger edit also changes a file
that the frozen v6 protocol had implementation-hash-bound; this is acceptable
only because v6 is terminal and further strengthens the technical refusal to
rerun it.

Mode2 magnitude-50 IDV1/3/9/15 runs 1--100 were already consumed by v5.1.
The failed v6 handshake added no new numeric consumption.  Those data may be
reused for genuinely new v7 exploratory development, but never as an unbiased
confirmation set.  Mode1 magnitude-25 and magnitude-75 cells remain sealed;
v6 does not authorize them, and a future method requires a separate frozen
pre-access protocol before any Mode1 action.

## Minimal reliable v7 governance

1. Assign a new protocol ID and a substantively revised candidate ID.  v7 must
   not be the identical v6 candidate under a rescue name.
2. Define two explicit hashes and never compare across domains:
   `canonical_json_sha256 = SHA256(canonical_json_bytes)` and
   `file_sha256 = SHA256(exact_persisted_bytes)`.  Record the canonicalization
   profile and whether the persisted file has a terminal LF.
3. Let the sidecar and atomic commit bind `file_sha256`.  Let in-memory runner
   and backend semantic validation exchange only `canonical_json_sha256`.
   The commit should record both values under their full names.
4. After freeze but before any global claim or access intent, run the actual
   production runner-to-backend handshake on the exact persisted protocol.
   Import and semantic validation must be explicitly read-free and must fail
   if any loader or HDF5 library is invoked.
5. Commit the one-use claim only after file verification, strict JSON parse,
   canonical-hash comparison, backend semantic validation, tests, and CUDA
   preflight all pass.  Commit the access intent immediately before the first
   HDF5 library open.
6. Instrument phase state directly: `intent_committed`,
   `hdf5_open_started`, `hdf5_open_completed`, `numeric_slices_read`, and
   `training_started`.  Use distinct completion states such as
   `ABORTED_AFTER_INTENT_BEFORE_HDF5_OPEN` and
   `ABORTED_AFTER_NUMERIC_READ`; do not infer numeric access solely from intent.
7. Keep candidate count one and prohibit rescue IDs.  Use the already-consumed
   Mode2 cell only for adaptive exploratory development.  Preserve Mode1 as
   sealed until a separate successor GO/refit and target-access protocol exist.

The simplest regression test is an exact frozen-file fixture with a terminal
LF.  It must assert that file and canonical hashes differ, that each verifies
in its own namespace, and that the full pre-claim runner/backend handshake
succeeds without importing an HDF5 library.
