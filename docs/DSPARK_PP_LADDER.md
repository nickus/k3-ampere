# DSpark under pipeline parallelism: the full ladder (2026-08-11)

What looked like one `NotImplementedError` is twelve obstacles stacked in
series. Each one is only visible after the previous is removed, which is why
[vllm#50098](https://github.com/vllm-project/vllm/issues/50098) has been open
since July with its author blocked for lack of multi-GPU hardware.

Found on 2×RTX 3090, vLLM main (`6c95a641e`), 4-layer synthetic K3 slice with a
matched miniature DSpark draft, PP=2.

## The ladder

| # | Where | Kind | What happens |
|---|---|---|---|
| 1 | `config/model.py:1381` | upstream fence | `verify_with_parallel_config` demands `SupportsPP` **of the draft**, because `create_draft_parallel_config` copies the target's PP size into the draft's config. The draft is not pipelined and should never need it. |
| 2 | `v1/worker/gpu/model_runner.py:212` | upstream fence — **the root** | `raise ValueError(f"{method} with pipeline parallel is not supported")` for eagle3/dflash/dspark. Its own comment names the reason: *"Drafting may require auxiliary hidden states from target model outputs"*. |
| 3 | `v1/worker/gpu/spec_decode/dspark/utils.py:49` | upstream fence | `NotImplementedError("DSpark does not support pipeline parallelism.")` |
| 4 | `models/kimi_k3/nvidia/model.py` | **upstream defect** | A non-last rank returns `IntermediateTensors({"hidden_states", "residual"})` and **silently drops `aux_hidden_states`**. The draft can only ever see taps produced by the last stage. |
| 5 | `models/kimi_k3/nvidia/dspark_mla.py:419` | **upstream defect (PP-only)** | The draft offsets its layer names by `get_num_layers(parallel_config)` — the count **on this rank**. Under PP it starts at 2 instead of 93 and collides: `ValueError: Duplicate layer name: model.layers.2.self_attn`. Invisible at PP=1, where the two numbers coincide. |
| 6 | `SupportsEagle3` protocol | our slice | The text-only `KimiLinearForCausalLM` does not declare the interface (the multimodal `KimiK3ForConditionalGeneration` does), although the inner model already implements the machinery. Slice-only; see `tools/slice_eagle3_shim.py`. |
| 7 | protocol members | our slice | The protocol is runtime-checkable, so `isinstance` tests member **presence**: `has_own_embed_tokens` / `has_own_lm_head` were missing. Asked `SupportsEagle3.__protocol_attrs__` directly instead of guessing. |
| 8 | our draft config | ours | Speculator refuses to start without `mask_token_id` (or one of three alternatives). |
| 9 | `vllm/usage/usage_lib.py:216` | **upstream robustness** | Optional usage telemetry calls `py-cpuinfo`, which raises `JSONDecodeError` inside a forked worker and **takes the whole engine down**. Telemetry should never be fatal. Workaround: `VLLM_NO_USAGE_STATS=1`. |
| 10 | our patch | ours | A symmetric-looking collective that cannot be symmetric: `init_speculator` runs under `if self.is_last_pp_rank`, so `load_dspark_model` **never executes on rank 0**. Broadcasting the embedding there hangs the group — rank 1 waits on `ncclUniqueId` while rank 0 finishes loading and idles. |
| 11 | our draft config | ours | vLLM adds 1 to `target_layer_ids` (DFlash id semantics), so `[1..n]` became `(2..n+1)`; the out-of-range last id silently produced one tap fewer than `context_proj` expects. |
| 12 | hybrid launch flags | ours | Missing `--enable-prefix-caching --block-size`, which we had already proven mandatory for hybrid K3 earlier the same day. |
| 13 | our miniature draft geometry | ours | `Triton Error [CUDA]: an illegal memory access` while loading the draft's MLA kernels. We kept production MLA dims (`q_lora_rank 1536`) against a 1024-wide slice — a ratio that does not occur in the real draft (1536 against 7168). The mini-draft geometry needs designing, not copying. |

Six are upstream (three fences, two real defects, one robustness bug); seven
are ours — five in the test harness, two in the patch itself. That ratio is
itself the finding: most of the cost of this work is not the upstream fix, it
is standing up an environment faithful enough to exercise it.

## The evidence that the core fix works

Obstacle 11 produced the most valuable error of the day:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (2048x3072 and 4096x1024)
```

`context_proj` was built for **4** taps (4 × 1024 = 4096) and received **3**
(3 × 1024 = 3072). Count where those three came from: with taps at layers 2, 3
and 4 under PP=2, layer 2 lives on **rank 0** and layers 3–4 on rank 1. So one
of the three arrived **from the other GPU**.

Before the patch that tap was discarded at the stage boundary and never reached
the draft at all. The tensor dimension in a failure message is a stronger
demonstration than a green check would have been — it cannot be produced by
accident.

## Status

- Obstacles 1–3, 5, 9 removed by `tools/dspark_pp_patch.py` (8 patches, all
  anchor-asserted, applied cleanly on two different main commits).
- Obstacle 4 — the real defect — fixed by the same patch (B1–B4) and
  demonstrated working, as above.
- Obstacles 6–8, 11–12 are test-harness issues, fixed in `tools/`.
- Obstacle 10 is **worked around, not solved**: the draft rank now takes the
  embedding from a tied `lm_head` when weights are tied, and otherwise builds an
  **uninitialised placeholder with a loud warning**. That is enough to exercise
  the tap plumbing and *not* enough for production — draft token ids would be
  meaningless. The production fix must move the weight where every rank is
  present (e.g. in the model runner right after `load_model`) or have the last
  rank read it from the target checkpoint.

## What is still unproven

Gate 2 — the tap fingerprint at PP=2 matching PP=1 exactly — has not run yet.
Until it does, "the taps cross the boundary" is established but "they arrive
intact and in the right order" is not. Gate 3 (acceptance rate, output parity)
is meaningless until obstacle 10 has a real fix, because a placeholder embedding
cannot produce sensible drafts.
