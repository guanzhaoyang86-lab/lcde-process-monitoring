# Audited tsai MiniROCKET subset

This directory vendors only `MiniRocketFeaturesPlus` and its direct standard,
NumPy, and PyTorch imports from the official tsai `v1.0.1` tag. Runtime code
must not import the `tsai` package.

`v1.0.1` is an annotated tag. Its Git object ID is
`2bca845b2905aa2cfcb03a2e1615bdb196b821e5`; peeling that object yields the
actual commit `f9ecc6b6df798946fa5fb290ac2ade9a00815289`. These are deliberately
recorded as different object identities in `PROVENANCE.json`.

The public numerical interface is the upstream interface:

- construct `MiniRocketFeaturesPlus(...)`;
- call `fit(X, chunksize=...)` once to fit the bias buffers;
- call the module on `[batch, channels, length]` tensors for PPV features;
- use the inherited `state_dict()` / `load_state_dict()` for tensors.

The no-pickle synthetic golden artifact and manifest under `golden/` bind the
upstream source, configuration, inputs, buffers, and CPU PPV output. The test
suite checks the vendored CPU implementation against it and, when CUDA is
available, checks CUDA agreement and deterministic replay.

