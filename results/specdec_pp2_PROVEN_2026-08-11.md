# Speculative decoding for K3 works under pipeline parallelism (2×RTX 3090)

> **CORRECTION (2026-08-12).** The gates below are true as written, but the
> title overstates them. Greedy output at PP=2 with speculation **differs** from
> the PP=1 reference — measured the next day, together with the controls that
> make it meaningful: PP alone preserves the text and speculation alone preserves
> the text. Booting and answering is not correctness, and a matching tap
> fingerprint proves the taps arrive, not that verification consumes them
> correctly. See `specdec_pp4_FIXED_2026-08-12.md`.

Measured 2026-08-11 on 2×RTX 3090 (sm_86), vLLM main `dev638+g52be12cfa`,
4-layer synthetic K3 slice + matched miniature DSpark draft, dummy weights,
`--pipeline-parallel-size 2`, `num_speculative_tokens 1`.

## Gate 1 — PASS: the engine serves at PP=2 with speculation enabled

`GATE1 PASS`. The server boots on two GPUs with the DSpark draft attached and
answers a completion request. Verified from this run's own artifacts (not stale
files): the fingerprint file carries a real 11-token request, not the zero-filled
profiling pass.

## Gate 2 — PASS: the taps are bit-identical to PP=1

The tensor the draft consumes, fingerprinted immediately before
`combine_hidden_states`, for the same prompt:

```
PP=1:  shape=(11, 4096)  sum=6.4611821672e-09  weighted=2.2653207910e-05  absmax=7.7307049651e-11
PP=2:  shape=(11, 4096)  sum=6.4611821672e-09  weighted=2.2653207910e-05  absmax=7.7307049651e-11
```

Identical to every digit, including the **position-weighted** checksum, which
was chosen precisely so that a *reordered* tap set would not match. Width 4096 =
4 taps × 1024, and at PP=2 two of those four are computed on rank 0 — so they
crossed the stage boundary intact and in the right order.

(The other fingerprint line, `shape=(2048, 4096)`, differs between the runs: at
PP=2 it is exactly zero, at PP=1 it is tiny but non-zero. That line is the
profiling pass with dummy input, not the served request, and its input differs
between the two configurations. It is not evidence either way.)

## What had to be fixed to get here

Eighteen obstacles in series, on top of the ten patches already recorded in
`DSPARK_PP_LADDER.md`. The two that mattered most in this final stretch were
both **mine**, and both invisible until measured:

1. **The boundary tap was taken twice.** `KimiK3Model.forward` opens by tapping
   `self.start_layer` from the stage *input*. At PP=1 that never fires (aux ids
   are 1-based, rank 0's start_layer is 0). Under PP it fires on every non-first
   rank and re-taps a value the previous stage already emitted — 5 taps where the
   draft wants 4. Fixed by suppressing it on non-first ranks (patch B5).
2. **The receiver under-allocated.** The receive-buffer count used
   `L < start_layer`, one slot fewer than the sender sends, so the last rank
   blocked forever inside `execute_model` while rank 0 sailed on into sampling.
   Fixed to `L <= start_layer` (patch B4).

Milestone tracing, not stack dumps, found both: stacks were unavailable
(ptrace blocked in the container, `faulthandler` silent), so progress markers
around each phase were used instead, and the diagnosis then came from
arithmetic — 4 layers, taps on all four, two ranks, 2 sent vs 1 expected.

## Honest limits of this result

- **Warmup is skipped.** Under PP+spec the last rank hangs inside warmup's
  sampling step — *before* the drafter is invoked, since the very same
  `speculator.propose` completes during the earlier profiling pass. Warmup only
  pre-JITs kernels, so it was bypassed (`SKIP_KERNEL_WARMUP=1`, test-only hack)
  to answer the end-to-end question. **That hang is a real, unfixed defect.**
- **PP=2 has no middle rank.** The branch that receives taps *and* forwards them
  onward never executes here. At PP=47 that is 45 of 47 ranks. Proving the patch
  for the rig needs PP≥3 — see `DSPARK_PP_DESIGN.md`.
- **No speedup number.** Acceptance is meaningless with a random dummy draft;
  measuring the actual win needs real weights.
- Everything is a 4-layer slice on 2 GPUs.

## Reproduce

`tools/dspark_pp_patch.py` (12 patches, anchor-asserted, applied cleanly on four
different `main` commits), `tools/slice_eagle3_shim.py` and
`tools/skip_warmup_hack.py` (both test-only), `tools/gen_dspark_draft.py`,
`tools/dspark_pp_probe.py`, `tools/warmup_pp_probe.py`, `tools/dspark_pp_test.sh`.
