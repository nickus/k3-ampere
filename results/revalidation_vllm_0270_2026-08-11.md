# Revalidation on vLLM 0.27.0 (current release) — 2026-08-11

Follow-up to `kv_offload_PROVEN_2026-08-11.md`, whose results were produced on a
build pinned 11 days behind main. This re-runs the same stack on the newest
release. Box: 2×RTX 3090 (sm_86), Quebec, `vllm/vllm-openai:latest` image, fresh
venv, `pip install vllm --pre` → resolved to **0.27.0** (published 2026-08-10).

## Headline

**The offload stack is blocked on 0.27.0 by two upstream bugs, both already
reported.** Neither is ours to fix, both have workarounds we already carry, and
with the workarounds the stack behaves as documented.

| Test | Config | Result on 0.27.0 |
|---|---|---|
| T1 | CPU tier, explicit `--block-size 512` | boots; **first request kills the engine** (`index_fill_` int64) |
| T2 | CPU tier, no explicit block size | **boot fails**: `tokens_per_block=16 not divisible by tokens_per_hash=512` |
| T3 | CPU tier, `"blocks_per_chunk": 1` instead of block size | **boot fails**, identical assertion |
| T4 | NVMe (`fs`) tier, explicit `--block-size 512` | boots; same first-request kill |
| T5 | fp8_ds_mla + NVMe tier | our 4 fp8 patches **apply cleanly**, all anchors matched |

## What each result means

**T2/T3 confirm [vllm#51752](https://github.com/vllm-project/vllm/issues/51752)
on the current release**, not just on our old pin — the PP rank owning no
attention layer never runs `_align_hybrid_block_size`. Prefix caching *was*
enabled in every run, which is exactly what makes the assertion message
misleading. Posted as a comment on the issue.

**T3 is a new negative result**: `blocks_per_chunk` — which the offloading docs
describe as the alternative to `block_size` "for models whose KV cache groups
have different block sizes", i.e. apparently aimed at hybrids — **does not work
around it**. The divergence happens before the connector is consulted. Only
pre-fork configuration (`--block-size`, which sets `user_specified_block_size`
in the parent) helps.

**T1/T4 hit the `index_fill_` int64 crash**, which is
[vllm#50947](https://github.com/vllm-project/vllm/issues/50947), open since
2026-08-04. Important correction to what this repo said earlier today: the fix
exists **only on main** (both branches replaced by Triton kernels
`_scatter_num_accepted_kernel` / `_fill_num_accepted_kernel`); **0.27.0 still
ships the broken call**. Our `.long()` patch therefore remains required for any
release-pinned deployment. Commented on the issue asking about a 0.27.x
backport.

**T5 is the good news**: the fp8_ds_mla port survived the 0.26→0.27 jump with
zero edits. All four patches (`P3a` dtype list, `P3b` 656-byte cache shape,
`P3c` SM89 gate carve-out, `P5` forward_mqa dispatch) matched their anchors —
and those patches assert on drift, so this is a real check, not a silent no-op.

## Slice-config drift (cost us most of the time, worth recording)

Upstream `moonshotai/Kimi-K3` config is now multimodal, and 0.27.0's registry
knows `KimiK3ForConditionalGeneration` (our old build did not, so it silently
fell back to the text-only path). Consequences, in the order they bit:

1. `OSError: Can't load image processor` → needs `preprocessor_config.json`.
2. Then a chain of remote-code files: `kimi_k3_vision_processing.py`,
   `modeling_kimi_k3.py`, `kimi_k3_processor.py`, `media_utils.py`.
3. With those present, the process was **OOM-killed** — dummy weights for a
   full-size vision tower against our 1024-hidden text slice.
4. Setting `architectures: ["KimiLinearForCausalLM"]` alone then failed with
   `'KimiK3Config' object has no attribute 'linear_attn_config'`: the text path
   wants a **flat** config, not one nested under `text_config`.

Fix: write the slice config as the flattened `text_config` with
`architectures: ["KimiLinearForCausalLM"]`. `tools/gen_slice_hf.py` should be
updated to emit that directly instead of the real repo's outer config.

## Cost / hygiene

Two rented boxes were discarded before this one (first accepted no SSH key,
second could not pull an image); both destroyed within minutes. Total spend for
this session's revalidation is under $2 of the $5.28 balance.
