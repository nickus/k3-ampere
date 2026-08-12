# PP=4 speculative decoding: the deadlock was a broadcast width mismatch

Box: 4× RTX 3090 (sm_86), driver 595.58.03, vLLM nightly `0.26.1rc1.dev651+g1d2d83a07`
(clean reinstall, then our patch set only). Kimi-K3 4-layer slice + DSpark draft,
`--load-format dummy`, `--enforce-eager`. Warmup **enabled** (no
`SKIP_KERNEL_WARMUP`) — see below.

## The bug

`PPHandler.receive` allocates `[num_reqs, max_sample_len]` where
`max_sample_len = num_speculative_tokens + 1`. `PPHandler.broadcast` sends
whatever the sampler produced. On any step with **no draft tokens** — the
prefill step above all — `sample()` takes its non-speculative branch and returns
one token per request. Measured at the fault site, PP=2, `k=1`:

```
Worker_PP0  RECV expects=(1, 2)           max_sample_len=2
Worker_PP1  SEND sampled_token_ids=(1, 1) max_sample_len=2
```

A `torch.distributed.broadcast` whose element counts disagree does not raise —
it blocks. Every symptom followed from this one line:

- three ranks stuck in `pp_handler.receive` while the last rank had already
  moved on to the next microbatch's `execute_model → irecv_tensor_dict`;
- the NCCL watchdog firing at 600s on `BROADCAST SeqNum=1, NumelIn=1024`;
- `TimeoutError: RPC call to sample_tokens timed out`.

Fix (patch **A7**): pad to the width the receivers allocate. `-1` is the
existing no-token filler, and the valid count travels separately in
`num_sampled`, which `get_prev_sampled_outputs` passes through — so no
semantics change.

## Five wrong explanations, each killed by a measurement

Worth recording, because every one of them was plausible and would have cost a
day of "fixing" the wrong thing:

| Hypothesis | Killed by |
|---|---|
| Upstream #51065's `query_len_support=UNIFORM` flag change | reverted it, used our A6 instead — still hung |
| PP≥3 pipeline ordering | **PP=4 without speculation serves fine** |
| `num_speculative_tokens > 1` | `k=1` hung too |
| Slow lazy Triton compile (warmup was skipped) | raised the deadline 120s → 600s, still dead |
| Asymmetric guard around the collective | both sides printed their decision: rank 0 `RECV`, last rank `SEND` — they agreed |

The last one is the instructive one. The guards *were* symmetric; the
disagreement was in the tensor, not the control flow.

## Result at PP=4

```
GATE1 PASS                      # boots and serves at PP=4
  taps seen: 25
  middle ranks exercised: 2     # ranks 1 and 2 receive AND forward taps
SpecDecoding metrics: Mean acceptance length: 1.05,
  Accepted throughput: 0.09 tokens/s, Drafted throughput: 5.94 tokens/s
```

Drafts are proposed **and accepted** with two middle ranks in the pipeline. The
prefill tap the draft consumes is bit-identical to the PP=1 reference:

```
PP=1  shape=(11, 4096) sum=6.4611821672e-09 weighted=2.2653207910e-05 absmax=7.7307049651e-11
PP=4  shape=(11, 4096) sum=6.4611821672e-09 weighted=2.2653207910e-05 absmax=7.7307049651e-11
```

## What is still open — do not overstate this

- **Decode taps are not element-wise comparable across PP degrees.** The 24
  decode taps differ between PP=1 and PP=4. At PP=4 four microbatches are in
  flight and sampled outputs are consumed `pp_size` steps later, so the k-th tap
  is not the same logical step as at PP=1. Gate 2's element-wise diff is only
  meaningful for the prefill tap. This is a limitation of the gate, not
  evidence of a defect — and equally, it is not evidence of correctness.
- **Greedy text parity is BROKEN at every PP>=2, and this is now measured, not
  suspected.** Four comparisons, same prompt, `temperature=0`:

  | comparison | result |
  |---|---|
  | PP=1 no-spec vs PP=4 no-spec | **SAME** — pipeline parallelism alone preserves the text |
  | PP=1 no-spec vs PP=1 + spec  | **SAME** — speculation alone preserves the text, as it must |
  | PP=1 no-spec vs PP=2 + spec  | **DIFFER** |
  | PP=1 no-spec vs PP=3 + spec  | **DIFFER** |
  | PP=1 no-spec vs PP=4 + spec  | **DIFFER** |

  Each factor is innocent alone and the combination is not, so this is a real
  defect in the PP+speculation path — not floating-point reordering, and not a
  property of the dummy weights. It appears at **PP=2**, i.e. it does not need a
  middle rank.

  **This retracts the implication of `specdec_pp2_PROVEN_2026-08-11.md`.** That
  file's two gates are still literally true — the engine boots at PP=2 and the
  taps the draft consumes are bit-identical — but "boots and answers" was never
  correctness, and the tap fingerprint only proves the taps arrive, not that
  verification uses them correctly. Greedy parity was never checked there; the
  gate that would have caught it compared strings contaminated by a probe banner.

  A7 is **not** the cause: padding with `-1` and padding with `0` produce
  byte-identical output, so the receivers do respect `num_sampled` and never
  read the filler. The parity defect predates A7 — the hang was simply masking
  it.

- ~~Greedy text parity PP=1 vs PP=4 is not established.~~ The two outputs agree
  for most of the sequence and diverge in the tail. With random dummy draft
  weights and a degenerate repetitive completion, a chance draft acceptance
  changes the sequence; acceptance was 1.00 at PP=1 and 1.05 at PP=4. Whether
  this is benign is **not yet measured**. What *is* measured: PP=4 is
  deterministic — two requests to the same server and a request to a freshly
  started one give byte-identical text — so the difference from PP=1 is a real
  PP=1-vs-PP=4 difference, not run-to-run noise. Separating "benign FP
  reordering flips a near-tied argmax on a degenerate output" from a genuine
  parity gap needs real weights.
- No speedup number, and acceptance is meaningless with a random draft.
- ~~Warmup is still skipped~~ — **A7 fixed the warmup hang too.** It was never a
  separate defect: warmup's first sample after its prefill step is exactly the
  no-draft-tokens case, so it broadcast the same narrow tensor. With A7 and no
  `SKIP_KERNEL_WARMUP`, PP=4 reaches `STATE=served` in 50s with zero errors, and
  a full gate run with warmup enabled reproduces the numbers above (acceptance
  1.05, drafted 5.92 tok/s). `tools/skip_warmup_hack.py` is now obsolete.

## Is A7 an upstream bug?

The mechanism is method-independent: `sample()` returns the narrow
`[num_reqs, 1]` whenever `input_batch.num_draft_tokens == 0`, which is true at
the prefill step for every speculative method, and `PPHandler` is shared. On the
V2 runner, `_get_v2_model_runner_unsupported_features` blocks `ngram` and
`eagle3 + PP>1`, but **allows `mtp` with PP>1** — so an MTP model served with
pipeline parallelism should hit the same broadcast.

Not filed yet: it was measured on DSpark+PP, which upstream forbids without our
patches. Filing needs one reproduction on a supported combination (MTP + PP),
which needs a model with an MTP head — K3 ships `num_nextn_predict_layers = 0`.
