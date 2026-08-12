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
- **Greedy text parity — much narrower than first reported. Measured, then
  measured again with the controls that matter.**

  My first pass said "broken at every PP>=2". That is true as a raw string
  comparison and **misleading as a description**. Two follow-ups pinned it:

  | max_tokens (k=3, PP=2) | first divergence | distance from the end |
  |---|---|---|
  | 24 | #22 of 24 | 2 |
  | 64 | #62 of 64 | 2 |

  | k (PP=2, max_tokens=32) | first divergence | distance from the end |
  |---|---|---|
  | **1** | **none — 32 tokens identical** | — |
  | 2 | #31 of 32 | 1 |
  | 3 | #30 of 32 | 2 |

  So the divergence is anchored to the **end of generation**, not to any absolute
  position, and its size is **k - 1** tokens — it scales with the speculative
  block, not with the pipeline depth. At `num_speculative_tokens=1` the PP=2
  output is byte-identical to the PP=1 reference over the whole completion.

  That makes this a **tail accounting defect at the stop boundary**: the final
  speculative block produces up to k+1 tokens at once, generation is cut by
  `max_tokens` inside that block, and the last k-1 of them are committed
  differently under PP than at PP=1. It is not corruption of the generated
  content, and it is not "speculation under PP is wrong".

  **Root cause found (2026-08-12, later).** Not the commit path and not the stop
  boundary. With a working probe (the first one printed its own template — see
  the retraction below), the target's verification logits differ per position:

  ```
  PP=1:  target=[100889, 100889, 100889, 100889]
  PP=2:  target=[100889,  16925,  16925,  16925]
  ```

  Position 0 — the token that always commits — is **correct** under PP. Positions
  1..k, the ones the drafts are verified against, all carry the **same wrong
  token**, which is what a set of rows reading one identical bad hidden-state row
  looks like. While every draft is rejected (`rejected=[3]`) the corruption is
  invisible; on the one step where a draft was accepted (`sampled=[2]
  rejected=[2]`) that wrong token, 16925, entered the output — it is the `做什么`
  in the diverging tail.

  So the defect is in the last rank's `sample_hidden_states =
  hidden_states[input_batch.logits_indices]`, where `hidden_states` arrived over
  `irecv_tensor_dict`: the speculative rows are taken from the wrong offsets.
  That also explains why the visible damage looked tail-shaped and rare — it only
  surfaces through an accepted draft.

  Retraction: the earlier "verification traces are identical, so the defect is
  downstream of sampling" was an artifact. That probe emitted doubled braces into
  an f-string, so every line printed as the same literal template and diffed
  clean. The "zero rejections" reading came from the same broken output.

  **Confirmed one level deeper: only the first row of hidden_states crosses the
  pipeline.** Fingerprinting the rows the last rank selects:

  ```
  PP=1  hs=(4,1024) idx=[0,1,2,3] rows=[0.014293,  0.014293,  0.014293,  0.014293]
  PP=2  hs=(4,1024) idx=[0,1,2,3] rows=[0.014293, -0.020827, -0.020827, -0.020827]
  ```

  Row 0 matches the PP=1 reference exactly. Rows 1..k carry a constant,
  `-0.020827`, which is the same value the probe printed during the profiling
  pass — i.e. they are **stale receive-buffer contents that the transfer never
  overwrote**. The buffer is the right shape; only one row of it arrives.

  The whole chain then follows: the target's logits for the speculative
  positions are computed from stale rows, so verification of drafts is
  meaningless and acceptance is effectively random; on the rare step where a
  draft is accepted, a token derived from that garbage enters the output. That is
  the "tail" divergence, and it is why plain PP is clean — without speculation a
  request has one query row and there is nothing to go stale.

  Same class of defect as A7, one transfer over: a count mismatch in the
  pipeline hand-off, silent because the buffer is large enough.

  **Narrowed to the transfer itself.** Fingerprinting on the receiving rank at
  the point of arrival — before it runs a single layer of its own:

  ```
  arrived: [2.075268, 2.035151, 2.035151, 2.035151]
           [2.046728, 2.035151, 2.035151, 2.035151]
           [2.057472, 2.035151, 2.035151, 2.035151]
  ```

  Row 0 changes from step to step (live data); rows 1..k hold **one constant
  across every step**. The corruption is already there on arrival, so it is not
  the receiving rank's forward — the inter-stage hand-off moves **only the first
  row** and leaves the rest as previous buffer contents. A probe on the sending
  side shows it holding the full 4 rows, so the data exists before the transfer.

  Sequence of the whole defect, all measured:
  1. inter-stage transfer delivers row 0 only; rows 1..k stay stale;
  2. the last rank computes speculative logits from those stale rows, giving the
     same wrong token at every speculative position;
  3. drafts are therefore verified against garbage and acceptance is random;
  4. on a step where a draft is accepted, that wrong token enters the output.

  Still open: the exact line that sizes the transfer (upstream's
  hidden_states/residual path — our B1/B2/B4 only add aux tap entries to the
  dict, they do not resize hidden_states), and whether it fires on EOS-terminated
  generation as well as token-limit truncation.

- ~~Greedy text parity is BROKEN at every PP>=2~~ (superseded by the rows above) Four comparisons, same prompt, `temperature=0`:

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

  A7 is **not** the cause, checked three independent ways: padding with `-1` and
  with `0` produces byte-identical output; a deliberately conspicuous filler
  (`12345`) never appears in the generated ids; and the verification traces are
  identical with the padding in place. The receivers do respect `num_sampled`
  and never commit the filler. The parity defect predates A7 — the hang was simply masking
  it.

  **Localised.** 22 of the 24 generated tokens are identical; the first
  divergence is at generated token #22:

  ```
  ref (PP=1)  ... 决赛中 有那么多 决赛中 有那么多 决赛中 | 有那么多
  PP=2 + spec ... 决赛中 有那么多 决赛中 有那么多 决赛中 | 决赛中
  ```

  The completion is a strict two-token cycle, and PP=2 breaks it by repeating.
  That lines up with the acceptance counters: PP=1 reports mean acceptance
  length **1.00** (nothing accepted) while every PP>=2 run reports **1.05**. So
  the PP path accepts a draft that PP=1 rejects, and the accepted token is not
  the one the target would have produced. The next step is to compare the
  verification logits the rejection sampler sees at PP=1 and PP=2 for that
  step — not to guess at the bookkeeping.

  **Done, and it moves the defect downstream of sampling.** A probe on the last
  rank printed, for every `sample()` call, `num_draft_tokens`, the target's
  argmax per verified position, the draft's argmax, `num_sampled` and
  `num_rejected`. Diffing a PP=1 run against a PP=2 run: **24 lines each, zero
  differences.** In the same instrumented build the texts still differ
  (`PP=1 spec vs PP=2 spec: DIFFER`).

  So the sampler reaches identical decisions at both PP degrees, and the
  sequences still come out different. The defect is therefore *not* in drafting
  and *not* in verification — it is in what the PP path does with the sampled
  tokens afterwards: `postprocess_sampled` / `update_requests` / the
  `PPHandler` FIFO that hands "previous sampled outputs" back `pp_size` steps
  later. The symptom fits: PP=2 repeats the previous token instead of advancing
  the two-token cycle, i.e. one token too many is committed.

  Where to look next, from reading the commit path rather than guessing: the
  last rank calls `postprocess_sampled` with `query_start_loc`, while the
  non-last ranks get it from `PPHandler.get_prev_sampled_outputs`, whose dict
  carries only `sampled_tokens`, `num_sampled`, `num_rejected` and
  `idx_mapping` — so `query_start_loc` is `None` there and `post_update` runs
  with different inputs on the two sides. That asymmetry is the first thing to
  instrument: log `num_computed_tokens` and `total_len` per rank per step at
  PP=1 and PP=2 and find the first step where they disagree.

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
