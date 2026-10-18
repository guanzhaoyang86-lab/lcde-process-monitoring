# CP-ETC-Scan v5.1 development result

## Immutable outcome

- Decision: `DEVELOPMENT_NO_GO`.
- Execution device: NVIDIA GeForce RTX 5090 Laptop GPU, CUDA 12.8,
  PyTorch 2.10.0.dev20251124+cu128.
- Iteration: `v5_1_fixed_formal_20260909`.
- Result artifact SHA-256:
  `f4ff76b465fcd02a90d297dc5320abecf54bb9182ed8db6481521a07d0f9a81c`.
- Completion artifact SHA-256:
  `3687009c0a6bcdd0b77f860c5e1b46bc1cb672c70dffd52069446e94a33ea74a`.
- Access-intent artifact SHA-256:
  `57f90ba392c9f8dbec7898958958176bca7025107aed6bad0c64a94f1307ece2`.
- Source Mode 2 archive SHA-256:
  `8462c0249761ae762a99028c7f8391620b68c69aaea4ecc961fe482799c974b4`.

The result did not emit a selected candidate, a final Mode 2 refit, or a
formal model. It did not open Mode 1. The Mode 1 magnitude-25 healthy
calibration cell and magnitude-75 confirmation cells therefore remain sealed.

## Primary development metrics

| Condition | Alarmed runs | Rate | Median delay | Nearest-rank p90 delay |
|---|---:|---:|---:|---:|
| Healthy | 0 / 100 | 0.00 | not applicable | not applicable |
| Fault 3 | 0 / 100 | 0.00 | 340 | 340 |
| Fault 9 | 0 / 100 | 0.00 | 340 | 340 |
| Fault 15 | 0 / 100 | 0.00 | 340 | 340 |

The healthy winner-dominance maximum share was 0.34224, so the expert-collapse
diagnostic passed. The healthy false-alarm gate also passed. Detection,
threshold-nondegeneracy, and CPU/CUDA numeric-parity gates failed.

## Root-cause audit

Every fold selected the same inner threshold
`14.8771591785096`. For one expert, the smallest possible empirical upper-tail
probability was `3 / 5101`, hence its largest evidence was
`-log(3 / 5101) = 7.4385795892548`. The top-two fusion therefore had the exact
finite ceiling

`2 * 7.4385795892548 = 14.8771591785096`.

Because an alarm required a score strictly greater than the threshold, no run
could ever alarm. This is a structural monitor failure rather than evidence of
perfect specificity.

Removing that ceiling alone is insufficient. Post-hoc outer-fold AUROC of the
raw expert heads averaged approximately 0.519 for Fault 3, 0.418 for Fault 9,
and 0.577 for Fault 15. Scan-level AUROC averaged approximately 0.514, 0.502,
and 0.528, respectively. Thus the representation also failed to generalize.

CPU/CUDA logits differed by roughly `1.45e-5` to `3.43e-5`. Rank-based ECDF
made these small continuous differences discontinuous: the p-value difference
was exactly `1 / 5101`. Alarm decisions and run maxima nevertheless agreed.
Future protocols should require deterministic repeated inference on the target
device and report cross-device drift, not make an overly strict CPU/CUDA
floating-point equality test an outcome gate.

## Consequence for v6

Mode 2 magnitude-50 is now an explicitly consumed, reusable exploratory
development pool. It cannot be described as a fresh holdout again. v6 must use
a materially different representation, a raw unbounded sequential statistic,
and out-of-fit healthy threshold estimation. Only a fully frozen Mode 2
development GO may authorize the one-time Mode 1 magnitude-25 healthy
calibration, followed by the one-time magnitude-75 confirmation.
