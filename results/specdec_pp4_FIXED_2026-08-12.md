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

  **Correction: the transfer is not short.** Probing the copy that fills the
  receive buffer shows the full width arriving:

  ```
  [NCOPY] n=4 num_tokens=4 recv_shape=(4, 1024) buf_shape=(2048, 1024)
  ```

  Four rows are received and four are copied, yet the fingerprint taken
  immediately after that copy already shows rows 1..k as a constant. So the
  corruption is present in **what the first rank sent**, i.e. in its own forward
  — not in the hand-off and not in the receiving rank.

  In this slice the first rank holds the **KDA linear-attention layers**, which
  carry recurrent state per sequence. Under pipeline parallelism that rank
  advances its state on a different schedule than at PP=1, which is the standing
  KDA-resume risk from the offload work. That makes the working hypothesis:
  **the KDA recurrent state is advanced wrongly for the speculative positions**,
  so rows 1..k of the stage output are stale while row 0 is right.

  Next measurement, and it is a direct one: fingerprint the output of the
  boundary layer at PP=1 and at PP=2 for the same prompt. If PP=1 gives four
  varying rows there and PP=2 gives one varying plus a constant, the defect is
  confirmed inside the first stage's layers rather than anywhere in the PP
  plumbing.

  Also still open: whether any of this fires on EOS-terminated generation rather
  than token-limit truncation.

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

## The first rank: root cause found, fix incomplete

**Measured, and it ends the search for where.** The token ids each rank feeds to
the model, decode step, PP=2, k=3:

```
PP=1              ids=[86072, 86072, 86072, 86072]
PP=2  first rank  ids=[86072,     0,     0,     0]
PP=2  last rank   ids=[86072, 86072, 86072, 86072]
```

The first rank embeds **token id 0** for every speculative position. The constant
`2.03515` seen in every earlier fingerprint is simply `embedding(0)`. Layer 0's
output already differs, so nothing before the embedding is involved.

Mechanism, read from the source:

- the draft token **values** live in `req_states.draft_tokens`
  (`gpu/states.py:73`, created as zeros);
- the only writer is `self.req_states.draft_tokens[...] = draft_tokens`
  immediately after `speculator.propose(...)` (`gpu/model_runner.py:1650`), and
  the speculator exists only on the last PP rank;
- the runner reads `scheduler_output.scheduled_spec_decode_tokens` for **counts**
  only (`model_runner.py:997`) and takes the **values** from that local buffer,
  which every other rank leaves at zero;
- `DraftTokensHandler` (`spec_decode/utils.py`) does not distribute anything
  across ranks — it copies the drafts to numpy for the engine's own output.

At PP=1 this is invisible: one process both writes and reads the buffer.

**Patch A8** fills `req_states.draft_tokens` from the scheduler dict on ranks
that do not run the drafter. It is mechanically effective — the first rank's ids
change from `0` to `-1` — but that is **not a fix**: `-1` is the "no token"
filler, so the dict the engine broadcasts does not carry real draft values
either. Greedy parity is still broken at k=1,2,3.

So the remaining question is one hop further back: what the last rank actually
returns to the engine as `DraftTokenIds`, and why the engine's next
`scheduled_spec_decode_tokens` is all `-1`. A8 stays in the patch set because it
is necessary but demonstrably not sufficient.

### Why A8 cannot work: the scheduler dict itself is empty of values

Measured on the non-last rank:

```
[DICT] last=False spec_tokens=[('cmpl-92abed90471fc5fe-0-ba20653e', [-1, -1, -1])]
```

`scheduler_output.scheduled_spec_decode_tokens` carries **`-1`**, the no-token
filler — while the last rank's own `req_states.draft_tokens` holds real ids
(its input ids were `[86072, 86072, 86072, 86072]`, not `-1`).

So there are **two** independent breaks, not one:

1. the device buffer `req_states.draft_tokens` has no producer on non-last ranks
   (A8 addresses this, and is necessary);
2. the engine publishes `[-1, -1, -1]` as the next step's draft tokens, so there
   is nothing for A8 to copy. The last rank verifies drafts the scheduler does
   not know about — it allocated k slots but filled them with the filler.

A8 is therefore correct and useless on its own, and is recorded as such. The next
question is narrow: what `DraftTokensHandler.get_draft_tokens()` returns, and
from which rank's `ModelRunnerOutput` the engine takes `draft_token_ids` under
pipeline parallelism.

A parallel code-reading pass (7 agents) reached the same mechanism for break 1
independently, and added two things this investigation did not have:

- row 0 survives because it is fed from `last_sampled_tokens`, which *is*
  maintained on non-last ranks via `update_pp_decode_requests()`, called before
  `prepare_inputs`;
- this is a V2-runner regression: the V1 runner spliced draft ids from
  `scheduled_spec_decode_tokens` on the CPU side, which is rank-symmetric; V2
  replaced that with a device-buffer read to avoid an H2D copy, and the buffer
  has no PP producer. Upstream's `ValueError` for `dspark` + PP
  (`model_runner.py:225-232`) is the acknowledgement that this path is unwired.

### The `-1` is by design, and that changes the verdict

`DraftTokensHandler` (`vllm/v1/worker/gpu/spec_decode/utils.py:22-52`):

```python
    def set_draft_tokens(self, input_batch, draft_tokens) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        if not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self.draft_tokens_np = None
            return
        ...
    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is not None:
            ...
        else:
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
```

Draft **values** are shipped to the scheduler only when structured outputs need
grammar validation. Otherwise the handler deliberately returns `-1` placeholders,
and the scheduler pads `scheduled_spec_decode_tokens` with `[-1] * k`
(`core/sched/scheduler.py:1071`). The scheduler only ever needed the *count*; the
values stay on the GPU and are consumed there. The engine reads them from the
last PP stage (`multiproc_executor.py:_get_output_rank` — "first TP worker of the
last PP stage"), so the rank is right; there is simply nothing to read.

**So A8 is wrong at the root**, not merely insufficient: the scheduler dict is
not a source of draft values by design. It stays in the tree only as a marker of
a dead end and should be removed rather than fixed.

**And the real obstacle is a data dependency, not a missing wire.** The drafts
for step T+1 are produced at the END of step T, on the last rank. The first rank
needs them at the START of T+1, to embed them as input tokens. Under pipelining
the first rank is already working on T+1 while the last rank is still on T, so
the value has to travel *backwards* in the pipeline against the direction of
flow. That is very likely what upstream's blanket `ValueError` for `dspark` + PP
(`gpu/model_runner.py:225-232`) is protecting.

Options, none of them a one-line patch, and none yet measured:

1. **Stall the pipeline** for the draft hand-off: correct, and it gives back part
   of the speedup speculation was meant to buy.
2. **Host the drafter on the first rank.** It is where the tokens are needed, but
   the draft consumes the target's final hidden states and lm_head, which live on
   the last rank — that is why upstream hosts it there.
3. **Carry the drafts to the first rank one step late** and verify with a lag.
   Changes semantics; needs proof it still matches greedy output.
4. **Give up speculation under PP** and take the throughput from elsewhere.

This is the first finding in this campaign that is a design question rather than
a defect, so it is recorded as one rather than patched over.

### A9: the naive broadcast deadlocks — the dependency is real, measured

A9 replaced A8 with the obvious fix: broadcast `req_states.draft_tokens` from the
last PP rank at the top of the step, before `combine_sampled_and_draft_tokens`
reads it. The last rank's buffer does hold the right values at that moment, so
the data is there.

Result: the server boots, and the first request **never returns** — empty
response, no forward executed after it, no error in the log (NCCL's watchdog did
not reach its 600s deadline inside the window). No stack was captured for this
run, so "deadlock" is the reading most consistent with the evidence rather than a
dumped call stack.

It is the predicted shape: rank 0 blocks in the broadcast waiting for the last
rank, while the last rank is still finishing the previous microbatch and waiting
on rank 0's hidden states. The draft values have to move **against** the
pipeline's direction, and a collective placed in the flow cannot do that.

So the backward dependency is no longer an argument — it is measured. Options 1-4
above stand, and option 1 ("stall the pipeline") specifically means an explicit
sequencing change, not a broadcast dropped into the existing step.

Both A8 and A9 are dead ends and are recorded as such. Neither is kept.

## This is a known, open upstream problem — not our bug

Searching the vLLM tracker (not deriving) turns up an active cluster of work on
exactly this combination:

| Issue/PR | What it is |
|---|---|
| [#44697](https://github.com/vllm-project/vllm/issues/44697) | **RFC**: MTP speculative decoding under pipeline parallelism (PP>1) |
| [#49355](https://github.com/vllm-project/vllm/issues/49355) | **Bug**: MTP spec decode broken with PP>1 — *three distinct failures* |
| [#44698](https://github.com/vllm-project/vllm/pull/44698), [#46994](https://github.com/vllm-project/vllm/pull/46994) | PRs adding MTP + PP support |
| [#45985](https://github.com/vllm-project/vllm/issues/45985) | Eagle3 speculative decoding with PP |
| [#49442](https://github.com/vllm-project/vllm/pull/49442) | Fix drafter access on non-final PP ranks (a crash, not our correctness bug) |
| [#50514](https://github.com/vllm-project/vllm/pull/50514) | Feat/spec decode under pipeline parallel |
| [#42109](https://github.com/vllm-project/vllm/issues/42109) | RFC: disaggregated spec decode with a standalone draft model |

**#49355 is our bug, reported independently on completely different hardware** —
3 nodes × 4×H100, Nemotron-3-Ultra 550B NVFP4, TP=4 PP=3, Ray. Their "Failure 3"
quotes the same two things this investigation measured:

```
scheduled_spec_decode_tokens={<req>: [-1, -1, -1, -1, -1]}
...
  File "vllm/v1/worker/gpu_model_runner.py", line 4399, in execute_model
    sample_hidden_states = hidden_states[logits_indices]
```

Same `-1` dict, same line. Theirs is the V1 runner, ours the V2 — the same shape
in both. They also note the model's own official launch recipe uses MTP at
`TP=8, PP=1`, so the vendor never exercised this path either.

**One practical knob from that report, tested here.** Their Failure 2: vLLM
*auto-enables* async scheduling for MTP-class methods with no `PP > 1` check, and
it breaks the PP broadcast; workaround `--no-async-scheduling`. Tested on our
setup: it changes behaviour — the A9 broadcast that previously deadlocked now
completes — but greedy parity is still broken at k=1,2,3. So it is a necessary
setting for anyone doing this, and not sufficient here.

## The relay works — defect 1 is fixed, defect 2 is next

`tools/specdec_pp_relay.py` (R1-R8) carries draft tokens on the deferred slot the
PP handler already uses. Measured effect on the ids the first rank embeds:

```
before:  first=True  ids=[86072, 0, 0, 0]
after:   first=True  ids=[86072, 86072, 86072, 86072]
         first=False ids=[86072, 86072, 86072, 86072]   <- ranks agree
         first=True  ids=[100889, 100889, 100889, 100889]
         first=False ids=[100889, 100889, 100889, 100889] <- agree
         first=True  ids=[100889, ...]
         first=False ids=[124005, ...]                    <- diverge
```

So the values arrive, the ranks agree for the first decode steps, and then the
first rank's ids **freeze** while the last rank moves on. Greedy parity is still
DIFFER at k=1 and k=3.

That freeze is upstream's **second** defect, quoted from
[#50514](https://github.com/vllm-project/vllm/pull/50514):

> `compute_need_sampled_mask` ended the broadcast early. The scheduler advances
> `num_computed_tokens` by the full scheduled width up front and rolls the
> rejected part back in `update_from_output`, which under PP lands *after* the
> next batch is scheduled. Reading the inflated count marked requests as
> finishing up to `num_speculative_tokens` early, after which the last rank
> stopped broadcasting and the other ranks' `last_sampled_tokens` froze.

A frozen `last_sampled_tokens` on the non-last rank is exactly what the trace
shows. So the remaining work is that second fix, not more searching.

**Where we are relative to upstream.** #50514 (yongqinwang-cmd, 2026-07-31, open,
21 comments, unmerged) implements both fixes and is validated on 2x8 B200 with
Kimi-K3 MXFP4 + DSpark: accept length 5.42, needle 3/3 at 1,029,433 tokens. Its
diff does **not** apply to current nightly (3/6 and 2/8 hunks fail), which is why
we carry our own equivalent of fix 1 written against today's tree.

We are not first. What is genuinely missing from that PR, and what this rig can
supply, is **validation on Ampere / consumer hardware** — every published run is
B200, H200 or GB10. One published datapoint even suggests our forced backend
helps: a practitioner reports acceptance *dropping* when moving the draft off
`TRITON_MLA`, and `TRITON_MLA` is the only MLA decode backend sm_86 has.

## Our patch set covers all three upstream defects — and the parity gate is the wrong instrument

Reading the rest of [#50514](https://github.com/vllm-project/vllm/pull/50514):

| Their defect | Ours |
|---|---|
| 1. draft tokens never reach non-last ranks (placeholder `-1` embedded) | measured independently; fixed by the V-series relay |
| 2. `compute_need_sampled_mask` ends the broadcast early, `last_sampled_tokens` freezes | measured (the ids froze); fixed by V6 |
| 3. **sampled-token broadcast width mismatch, "fixed by padding the send side"** | **this is our A7**, derived independently before we found the PR |

So the three defects are the same three, found twice, from opposite ends.

Two sentences from that PR change what we should be measuring:

> **Scope: capped at `pipeline_parallel_size <= 2`.** … pp>2 is the first topology
> with a **middle** stage, which must both adopt upstream taps and contribute its
> own to the same payload, and **no such run has happened on hardware**.

> this feature does not fail loudly: a drafter fed mis-ordered or missing taps
> still emits syntactically valid proposals that simply get rejected more often,
> so **the only symptom is a depressed acceptance rate**. Lifting the cap should
> require an **acceptance-rate comparison at pp>2**, not just a successful boot.

Consequences for this work:

1. **The greedy-parity gate is the wrong instrument on this stand.** With random
   dummy draft weights the proposals are noise and acceptance is ~0, so text
   parity can fail for reasons that have nothing to do with the plumbing. Every
   "DIFFER" recorded above is consistent with that and is not evidence against
   the relay. The right criterion, and theirs, is acceptance rate on real weights.
2. **The gap we can uniquely fill is exactly the one they name**: pp>2 with a real
   middle stage, on Ampere. We have already booted and served PP=4 with two
   middle ranks; nobody in any published run has gone past PP=2, and every
   published run is B200/H200/GB10.
3. **Statistically meaningful speedup numbers require real weights.** Acceptance
   rate and tokens/s on a dummy 4-layer slice are meaningless by construction.

## PROVEN on the right instrument: the corrupted quantity is now correct

Comparing the target's verification logits — the exact thing the missing draft
tokens corrupted — between PP=1 and PP=2, with the relay applied:

```
PP=1:  ndraft=3 target=[100889, 100889, 100889, 100889] sampled=[1] rejected=[3]
PP=2:  ndraft=3 target=[100889, 100889, 100889, 100889] sampled=[1] rejected=[3]
       cmp: IDENTICAL   (12 lines each, byte for byte)
```

Before the fix the same comparison gave `PP=2 target=[100889, 16925, 16925,
16925]` — position 0 right, the speculative positions all carrying one wrong
token derived from `embedding(0)`.

This is weight-independent. It does not care that the draft is random, because it
compares the *target's* verification against itself across PP degrees. Greedy
text parity, which I used for hours, cannot do that on a dummy-weight stand — by
the PR author's own description the failure mode is silent and shows up only as a
depressed acceptance rate.

Checked the probe itself before believing the result: the traces contain real
token ids, not the boilerplate that fooled an earlier run.

**Status: the plumbing is fixed and demonstrated at PP=2 on sm_86.** What remains
is the measurement that matters — acceptance rate and throughput on real weights,
at PP=1 / 2 / 4, which is both the number we owe and the evidence #50514 says it
needs to lift its own `pp <= 2` cap.

---

# A second, larger defect: MTP cannot run under PP **at all**, and it fails before any weight loads

Moving to real weights immediately produced a failure that had nothing to do with
the relay. GLM-4.5-Air AWQ, `--speculative-config '{"method":"mtp",...}'`, PP=4:

```
NotImplementedError: Pipeline parallelism is not supported for this model.
Supported models implement the `SupportsPP` interface.
  vllm/config/speculative.py:_verify_args
    -> draft_model_config.verify_with_parallel_config(_dpc)
  vllm/config/model.py:verify_with_parallel_config
```

This is a **config-time** rejection inside `create_engine_config`. Nothing is
loaded, no GPU is touched, no relay is reached. The whole `spec=yes` half of the
first benchmark matrix wrote zero rows because of it, which is how it was found:
`bench.jsonl` jumped from `pp4_specno_r5` straight to `pp8_specno_warmup`.

## Why the check is wrong for a draft model

`create_draft_parallel_config` copies the target's `pipeline_parallel_size` into
the draft's parallel config. Verification then demands the draft implement
`SupportsPP`. But in the V2 runner the drafter is built here:

```python
if self.speculative_config is not None:
    if self.is_last_pp_rank:
        self.speculator = init_speculator(self.vllm_config, self.device)
```

`is_last_pp_rank` — the draft is instantiated on exactly one rank and is never
split across stages. So the question "can this model be pipeline-sharded?" is one
nobody needs answered about a draft. Asking it anyway rejects a configuration
that would have worked.

## Scope: this is not a Kimi problem

Read the class definitions: `Glm4MoeMTP` and `DeepSeekMTP` are
`(nn.Module, <...>MixtureOfExperts)` — neither inherits `SupportsPP`. Neither do
the other entries of `MTPModelTypes`, which currently lists ~20 heads
(`deepseek_mtp`, `glm4_moe_mtp`, `ernie_mtp`, `qwen3_next_mtp`, `minimax_m3_mtp`,
`kimi_k3_mtp`, …). **Every one of them is blocked under PP>1 by this check.**

Searched the vLLM tracker before claiming novelty (`gh search issues`, four
phrasings): the only related open item is #50098, DSpark-specific. The general
MTP case appears unreported.

## The fix, and the self-correction it forced

Our own earlier patch had already relaxed this — but gated on
`method == "dspark"`, which turned a general defect into a local workaround and
hid it from us for a week. The gate is now unconditional, with only the
verification relaxed and the config restored in `finally`, so rank and
world-size bookkeeping downstream is untouched:

```python
if _dpc.pipeline_parallel_size > 1:
    _pp = _dpc.pipeline_parallel_size
    object.__setattr__(_dpc, "pipeline_parallel_size", 1)
    try:
        self.draft_model_config.verify_with_parallel_config(_dpc)
    finally:
        object.__setattr__(_dpc, "pipeline_parallel_size", _pp)
```

Lesson, recorded because it will recur: **a special case that makes our own
configuration work is a reason to ask what general defect it is hiding.**
Nothing here was Kimi-specific; we just never tried a second method.


---

# Real weights, PP=8: the PP=2 result did NOT generalize. Retracting the implied claim.

The relay was proven at PP=2 on a 4-layer dummy-weight stand, by byte-identical
verification traces. I took that as the mechanism being fixed. On real weights at
PP=8 it is not.

## The stand

A Kimi-K3 built from 24 of REAP-448's 448 experts: 141.3 GB, all 93 layers, KDA
linear attention, MLA, MXFP4 intact, real tiktoken tokenizer, and the real
`RedHatAI/Kimi-K3-speculator.dspark` draft. Fits 8x3090 with a hand-balanced
pipeline (`VLLM_PP_LAYER_PARTITION=11,13,13,13,13,13,12,5` - the last stage
carries the draft, measured at ~9.7 GB resident, so it gets 5 layers).

Kept 24 rather than 16 deliberately: at 16, top-16 routing selects every expert
and the MoE degenerates into a dense sum, so nothing is being routed.

## Baseline: it serves, and it is coherent-shaped garbage

```
k3-slice PP=8 spec=off:  TPOT 243.0 ms  =  4.12 tok/s/req,  TTFT 0.35 s
```

Generation, greedy:

```
' The Python function is a Python function that merges two sorted lists.
  The Python function is a Python function that merges two sorted lists. ...'
```

Degenerate, exactly as predicted for 24-of-448 experts - and printed BEFORE any
rate, because on a slice like this a high acceptance number is the least
trustworthy outcome available.

## Speculative decode on: the output is corrupted

```
' The \n Python+ PythonPython PythonPython Python PythonPython Python ...'
' Explain! same a time race! condition a!! is!!!!!!!!!!!!!!!!!!!!!!!!!!!'
```

vLLM reported `mean acceptance length = 1.44`, i.e. drafts were accepted.

The tokenizer settles what those are:

```python
>>> t.decode([0]) -> '!'
```

Token id 0 is `!`. A run of `!` in the output is `embedding(0)` - the signature
of draft tokens arriving as zeros, which is precisely the defect the relay was
written to fix and precisely what it fixed at PP=2. Greedy output differs between
spec-on and spec-off, so speculative decoding is not lossless here: it is
accepting garbage, and the acceptance figure of 1.44 is counting that garbage.

**So: fixed at PP=2 on a dummy stand, still broken at PP=8 on real weights.** I
am not able to claim from this session that speculative decoding works under
pipeline parallelism on real weights. The honest status is that the PP=2 proof
was necessary and not sufficient, and that the deferred-slot lag scaling with
pp_size is the obvious suspect - at PP=8 the payload is consumed eight steps
later, not two.

## The other spec path fails too, differently

GLM-4.5-Air + `method=mtp` at PP=4 never even reaches the relay: it dies in
`create_engine_config` on a `SupportsPP` check that no MTP head satisfies
(fixed here, generally), then in an unmasked Triton store in
`_prepare_prefill_inputs_kernel` (guard written, deliberately not applied), then
in `sample()` on `hidden_states[input_batch.logits_indices]` with

```
Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out
of bounds"` failed
```

K3+DSpark at PP=8 hits that SAME assertion at k=3 before the corruption above
appears at k=1. Two unrelated proposers, two model families, one class of
failure: indices derived from the scheduled step layout are out of range on the
drafting rank. That is now the thing to chase, and it is upstream of our patches
- verified by reproducing the MTP case with `relay_drafts=False`.

## Measured, and not in dispute

```
GLM-4.5-Air AWQ, PP=4, spec off, 8x3090, enforce-eager
  concurrency 1:  TPOT 94.60 ms   10.57 tok/s/req    10.6 tok/s total
  concurrency 4:  TPOT 92.68 ms   10.79 tok/s/req    43.2 tok/s total
  concurrency 8:  TPOT 93.99 ms   10.64 tok/s/req    85.1 tok/s total
  repeat stability: 94.60 vs 94.70 ms  (0.1%)
```

Per-request decode is flat while aggregate scales ~8x, which says the pipeline at
batch 1 is mostly idle. That is the number that makes a batch-1 speculative
speedup claim indefensible on its own, and it is why the harness now sweeps
concurrency.

TTFT above is a prefix-cache hit, NOT prefill. There is no honest prefill number
in this session; it needs unique prompts or caching disabled.
