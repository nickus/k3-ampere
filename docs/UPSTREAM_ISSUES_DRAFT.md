# Two upstream issues, drafted. NOT filed.

Both reproduce in about a minute on 2 GPUs. Neither needs Kimi-K3 or our patches:
issue 1 is a config-time rejection, issue 2 reproduces on any PP-capable model.

---

## Issue 1 — `SupportsPP` is required of draft models, which are never pipelined

**Title:** [Bug]: MTP speculative decoding cannot start under pipeline
parallelism — `SupportsPP` demanded of the draft model

**Repro** (any model with an MTP head; GLM-4.5-Air used here):

```
vllm serve <model> --pipeline-parallel-size 4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

```
NotImplementedError: Pipeline parallelism is not supported for this model.
Supported models implement the `SupportsPP` interface.
  vllm/config/speculative.py  _verify_args
    -> draft_model_config.verify_with_parallel_config(_dpc)
  vllm/config/model.py        verify_with_parallel_config
```

This is raised inside `create_engine_config`, before any weight is loaded.

**Why it is wrong.** `create_draft_parallel_config` copies the target's
`pipeline_parallel_size` into the draft's parallel config, so verification then
demands `SupportsPP` from the draft. But in the V2 runner the drafter is built
under

```python
if self.speculative_config is not None:
    if self.is_last_pp_rank:
        self.speculator = init_speculator(self.vllm_config, self.device)
```

`is_last_pp_rank` — a draft is instantiated on exactly one rank and is never
split across stages, so whether it *could* be split is not a question that needs
answering about it.

**Scope.** `Glm4MoeMTP` and `DeepSeekMTP` are `(nn.Module, ...MixtureOfExperts)`
— neither inherits `SupportsPP`, and neither do the other ~20 entries of
`MTPModelTypes`. Every MTP head is blocked under PP>1 by this check.

**Suggested fix.** Verify the draft as the single-stage model it is, restoring
the config afterwards so rank and world-size bookkeeping is untouched:

```python
if _dpc.pipeline_parallel_size > 1:
    _pp = _dpc.pipeline_parallel_size
    object.__setattr__(_dpc, "pipeline_parallel_size", 1)
    try:
        self.draft_model_config.verify_with_parallel_config(_dpc)
    finally:
        object.__setattr__(_dpc, "pipeline_parallel_size", _pp)
else:
    self.draft_model_config.verify_with_parallel_config(_dpc)
```

---

## Issue 2 — spec decode + PP silently corrupts output without async scheduling

**Title:** [Bug]: speculative decoding under pipeline parallelism produces wrong
output with `--no-async-scheduling`

**The mechanism.** `next_decode_eligible_step` is assigned in exactly one place:

```
v1/core/sched/async_scheduler.py:49
    request.next_decode_eligible_step = self.current_step + self.pp_size
```

and read in exactly one place:

```
v1/core/sched/scheduler.py:509
    if self.current_step < request.next_decode_eligible_step:
        # V2+PP+async: enforce `pp_size` steps between same-req decodes
        # to match worker-side sampled-tokens broadcast slot ring cadence.
```

The base `Scheduler` — selected by `--no-async-scheduling` — reads that field and
never sets it. It stays 0, the guard never fires, and nothing keeps a request's
decodes `pp_size` steps apart, so the worker-side sampled-token broadcast ring is
read out of phase.

**What that looks like, measured.** A probe where the step's input ids are built:

```
[ANCHOR] last_sampled=[16925]  drafts=[16925, 16925, 16925]   healthy
[ANCHOR] last_sampled=[0]      drafts=[0, 0, 0]               ~25% of spec steps
```

Token id 0 gets embedded as the anchor. On Kimi-K3, token 0 is `!`, and the
output fills with `!`. There is a second face of the same skew:

```
healthy:  hs=(4,1024)  idx=[0, 1, 2, 3]   qlen=[4]  num_logits=[4]  ndraft=3
failing:  hs=(3,1024)  idx=[-1, 0, 1, 2]  qlen=[3]  num_logits=[4]  ndraft=3
```

`logits_start = query_end - num_logits = 3 - 4 = -1`, and
`hidden_states[input_batch.logits_indices]` then either wraps to the last row
(torch) or trips

```
Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out
of bounds"` failed
```

The same skew also reaches `_prepare_prefill_inputs_kernel`, where
`query_len -= num_rejected` can reach 0 and the unmasked
`tl.store(draft_input_ids_ptr + query_start - 1, ...)` writes out of bounds.

**Impact.** Silent wrong output, or a CUDA fault, for any spec method under PP
when async scheduling is off. Reproduced with `mtp` on GLM-4.5-Air and with
`dspark` on Kimi-K3, at PP=2, 4 and 8.

**Suggested fix.** Either set the cadence in the base scheduler too, or reject
`speculative_config` + `pipeline_parallel_size > 1` + `--no-async-scheduling` at
config time with a message that says why. Silently producing `!` is the worst of
the three options.

---

## Verified on a pristine tree

Both issues above were re-checked in a separate venv holding an unpatched
`vllm==0.26.1rc1.dev693+g7f7a32cfe` (the latest nightly at the time), serving
stock GLM-4.5-Air AWQ with its own MTP head. Nothing of ours is in that venv
except where noted.

MTP under pipeline parallelism turns out to fail in **three** places in a row,
each hiding the next:

1. **Config time.** `SupportsPP` demanded of the draft model — issue 1 above.
   Nothing loads. Fixed by the one-line verification relaxation.
2. **Warmup.** With only that line applied, the last rank dies inside
   `compile_or_warm_up_model`:
   ```
   vllm/v1/worker/gpu/warmup.py:410  warmup_kernels
   vllm/v1/worker/gpu/warmup.py:379  _run_decode_step
   vllm/v1/worker/gpu_worker.py:1073 execute_model
       get_pp_group().irecv_tensor_dict(...)
   vllm/distributed/parallel_state.py:1148 irecv_tensor_dict
   ```
   The other ranks then report `Connection closed by peer`, which is the
   downstream symptom, not the cause. This is a send/receive width mismatch on
   the sampled-token broadcast: the sender ships `sampled_token_ids` narrower
   than the width receivers allocate when speculation is on.
3. **Runtime.** With warmup fixed, output is wrong unless async scheduling is on
   — issue 2 above.

The controls, same tree, same day:

```
spec off, async on  vs  spec off, async off : IDENTICAL   (scheduler mode alone
                                                           changes nothing)
Qwen3-0.6B, PP=1 vs PP=2, no speculation    : IDENTICAL   (the pipeline path
                                                           itself is greedy-stable)
```

So neither finding is "PP is broken" and neither is "our patches did it". Both
are specific to speculative decoding under pipeline parallelism, and both
reproduce on stock code with a stock model.
