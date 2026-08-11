# Upstream issue drafts — hybrid + KV-offload + PP (found 2026-08-10/11)

Three independent findings from bringing up `OffloadingConnector` on a hybrid
model (Kimi-K3: KDA linear-attention + MLA) with `--pipeline-parallel-size 2`.
All reproduced on real hardware (2×RTX 3090, sm_86), vLLM 38a267cdd, and all
three still present in main @1a1727330a (2026-08-10).

Filing order: #1 is a crash with a one-line fix (best first impression);
#2 is the subtle one that costs people days; #3 is a docs/UX papercut.

---

## ISSUE 1 — `index_fill_(): Expected dtype int64 for index` in the mamba-hybrid chunked-prefill path

**Title:** `[Bug] Hybrid (mamba/linear-attention) models crash with
index_fill_ dtype error under prefix caching + chunked prefill`

**Severity:** hard crash (`EngineDeadError`) on the first request. Affects any
hybrid model with `--enable-prefix-caching`, independent of KV offload.

**Where:** `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`, in
`postprocess_state`:

```python
else:
    # Fill with single value.
    self.num_accepted_tokens_gpu.index_fill_(
        0, idx_mapping, max(num_sampled, 1)
    )
```

`idx_mapping` is int32 (the sibling branch feeds it to a Triton kernel that
accepts int32), but `Tensor.index_fill_` requires an int64 index. The `else`
branch is taken when `num_sampled` is an int, i.e. **chunked prefill** — which
`mamba_cache_mode="align"` explicitly requires
(`vllm/model_executor/models/config.py`, "Chunked prefill is required for
mamba cache mode 'align'"). So the crash is on the mandatory path.

**Repro:** any hybrid model, `--enable-prefix-caching`, send one request.
Ours: Kimi-K3 (69 KDA + 24 MLA), PP=2, one completion request →
`RuntimeError: Worker failed with error 'index_fill_(): Expected dtype int64
for index.'`

**Fix (verified on hardware):**
```python
self.num_accepted_tokens_gpu.index_fill_(0, idx_mapping.long(), max(num_sampled, 1))
```
`.long()` is a no-op when the tensor is already int64. With this applied, the
full hybrid + prefix-caching + offload path runs and restores bit-exactly.
Happy to open the PR.

---

## ISSUE 2 — hybrid block-size alignment is skipped on PP ranks that own no attention layer

**Title:** `[Bug] Hybrid block-size alignment diverges across pipeline-parallel
ranks (rank with no attention layers keeps the unaligned block size)`

**Severity:** startup failure with a misleading message; silently
configuration-dependent (works or fails depending on where the PP split lands).

**Mechanism:**
`Platform.update_block_size_for_backend` (`vllm/platforms/interface.py`) does
```python
backend_cls = cls._find_non_ssm_backend(vllm_config)
if backend_cls is None:
    return
```
and `_find_non_ssm_backend` inspects only the layers **this process
instantiated** (`static_forward_context`). Under PP each rank builds only its
slice; the rest are `PPMissingLayer`. A rank whose slice contains **no
attention layer** therefore returns `None` and skips `_align_hybrid_block_size`
entirely, keeping `cache_config.block_size` / `mamba_block_size` at their
pre-alignment values, while the rank that owns an attention layer raises both
to the aligned value. Nothing synchronises these fields across ranks.

Specs are later merged by layer **name** (PP stages have disjoint names, so the
"specs differ across workers" assert never fires), producing KV-cache groups
with mixed block sizes. `resolve_kv_cache_block_sizes` then backs off to the
LCM, and the offloading connector's divisibility assert fires:

```
AssertionError: tokens_per_block=16 not divisible by tokens_per_hash=512.
Hybrid models (e.g. Mamba+Attention) need --enable-prefix-caching to align block sizes.
```

The message misleads: prefix caching was already enabled; the real problem is
cross-rank divergence, and the group stuck at 16 is a **mamba** group on the
attention-less rank.

**Repro:** hybrid model whose PP split leaves one rank without attention
layers. Ours: 4-layer K3 slice, KDA on layers 1–3, MLA on layer 4, PP=2 →
rank 0 = pure KDA.

**Workaround (works, worth documenting either way):** pass `--block-size N`
explicitly. It sets `user_specified_block_size` before the fork, so every rank
starts aligned and the mutation never happens.

**Suggested fix:** compute the alignment from the *model config* rather than
from locally-instantiated layers, or broadcast the aligned
`(block_size, mamba_block_size)` from a rank that has an attention layer.

---

## ISSUE 3 — offloading `kv_connector_extra_config` key mismatch / discoverability

**Title:** `[Bug/Docs] OffloadingConnector: num_cpu_blocks silently unsupported;
cpu_bytes_to_use is required but undocumented in the connector docs`

Passing the natural-looking `{"num_cpu_blocks": N}` yields
`RuntimeError: cpu_bytes_to_use must be specified in kv_connector_extra_config`
(`vllm/v1/kv_offload/cpu/spec.py`) — the unknown key is ignored rather than
rejected. The tiering/NVMe schema (`spec_name: "TieringOffloadingSpec"`,
`secondary_tiers: [{"type": "fs", "root_dir": ...}]`) is documented only in the
module docstring of `vllm/v1/kv_offload/tiering/spec.py` and the factory
registry.

**Ask:** validate/reject unknown keys in `kv_connector_extra_config`, and add a
short "KV offloading to CPU/NVMe" section with a copy-pasteable config to the
connector docs.

---

## Companion data we can offer upstream

With all three addressed, hybrid K3 + PP=2 + CPU tier + NVMe (`fs`) tier +
`fp8_ds_mla` KV all work, and restore is **bit-exact** (max logprob delta
0.00000000 between a fresh run and one restored after forced GPU-cache
eviction). Details and measurements:
`results/kv_offload_PROVEN_2026-08-11.md`. A hybrid+PP offload repro is
something the tracker currently lacks (cf. #50821, #46453, #43508, #50235).
