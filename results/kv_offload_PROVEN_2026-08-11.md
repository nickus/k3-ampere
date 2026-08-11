# KV offload for Kimi-K3 works — hybrid + PP + NVMe + fp8, bit-exact (2026-08-11)

This supersedes `kv_offload_validation_2026-08-10.md`, whose verdict
("offload NOT ready for K3-hybrid") was **wrong**: those runs died on a
chain of three config requirements plus one genuine upstream bug, not on an
architectural gap. All four are identified below, and with them applied the
full stack works.

## Measured on hardware (2×3090, PP=2, 3 KDA + 1 MLA slice, dummy weights)

| Configuration | Result |
|---|---|
| Hybrid K3 + `OffloadingConnector` + PP=2, bf16 KV | boots; cold 1.17 s → warm 0.21 s (**5.7×**); **max logprob delta 0.00000000** |
| Same, forced eviction (`--num-gpu-blocks-override 64`) so the restore MUST come from the CPU tier | cold 0.25 s → 0.13 s (1.9×); **delta 0.00000000**; 9 offload log events |
| **NVMe tier** (`TieringOffloadingSpec` + `secondary_tiers:[{type:"fs"}]`), bf16 | **9 files / 10 MB physically written to disk**; restore 2.04×; **delta 0.00000000** |
| **NVMe tier + `fp8_ds_mla`** (our fp8-KV port), block 1024 | boots; restore 1.98×; **delta 0.00000000**; files on disk |

**Bit-exactness is the headline.** Zero logprob divergence between a fresh
run and a run restored from CPU/NVMe means the KDA recurrent state and the
MLA latent KV both come back byte-identical. LMCache's "generation is not
bit-exact after restore" caveat does **not** apply to this path (that caveat
is about batch-composition nondeterminism in their connector, not state
fidelity). The recurrent state is the complete Markovian summary of the
sequence, stored as raw fp32 bytes — copy out, copy in, no reconstruction.

## The four things that were actually blocking it

1. **`--enable-prefix-caching` is mandatory.** Without it
   `mamba_cache_mode="none"` → `mamba_block_size = max_model_len` (4096),
   which can never divide the attention block size → the "Hybrid models …
   need --enable-prefix-caching" assert. The error message says this; we
   didn't read it.
2. **`--block-size N` must be passed EXPLICITLY** — and this works around a
   real upstream PP bug. `Platform.update_block_size_for_backend` →
   `_find_non_ssm_backend` only inspects layers *this rank* instantiated.
   A PP stage owning **no attention layer** (ours: rank 0 = pure KDA)
   returns None and **skips `_align_hybrid_block_size` entirely**, keeping
   `block_size=16` while the other rank computes 512 → groups disagree →
   assert. Passing `--block-size` sets it pre-fork, so all ranks agree by
   construction. Worth filing upstream.
3. **`cpu_bytes_to_use`, not `num_cpu_blocks`** in
   `kv_connector_extra_config` (`v1/kv_offload/cpu/spec.py:81-84`).
4. **Upstream bug, one-line fix:**
   `v1/worker/gpu/model_states/mamba_hybrid.py:306` calls
   `index_fill_(0, idx_mapping, …)` with an **int32** index →
   `RuntimeError: index_fill_(): Expected dtype int64 for index`. The
   sibling branch uses a custom Triton kernel that accepts int32. This path
   is chunked prefill, which align mode *requires*, so **any** hybrid model
   with prefix caching hits it. Fix: `idx_mapping.long()`.

## Working launch recipe

```bash
vllm serve <k3> --trust-remote-code \
  --pipeline-parallel-size N --tensor-parallel-size 1 \
  --kv-cache-dtype fp8_ds_mla \        # or auto
  --enable-prefix-caching \
  --block-size 1024 \                  # >= align size; see arithmetic below
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec",
      "cpu_bytes_to_use":<RAM bytes>,
      "secondary_tiers":[{"type":"fs","root_dir":"/nvme/kv"}]}}'
```

Block-size arithmetic (`platforms/interface.py:901-907`):
`attn_block = 128 · ceil(mamba_page / (128 · attn_page_per_token))`.
Slice: mamba page 542,720 B; attn 1152 B (bf16) → **512**; attn 656 B
(fp8_ds_mla) → **896** (we used 1024). For real K3 (96 KDA heads):
mamba page ≈ 6.5 MB → bf16 **≈ 5,760**, fp8 **≈ 10,112** tokens/block.
Large but workable for 1M-token sessions (~100–200 blocks/seq); wasteful
for short prompts — a real consideration for mixed workloads.

## Projection to 100 agents × 1M context on ~50 cards (model, not measurement)

Per session, real K3: MLA KV = 656 B × 24 layers × 1M = **15.7 GB**
(fp8; bf16 would be 27.6 GB). KDA state = 96 heads × ~65.5 KB × 69 layers
≈ **448 MB**, constant in context length.

- 100 sleeping sessions ≈ 1.57 TB (MLA) + 45 GB (KDA) ≈ **1.62 TB** → fits a
  4 TB NVMe tier, or per-node NVMe split across hosts.
- VRAM budget: W2 weights ~883 GB / 50 cards ≈ 17.7 GB/card, leaving
  ~4.8 GB/card → ~240 GB of resident KV → **~15 concurrently *decoding*
  1M-context sessions**. For a 100-agent swarm where most agents are waiting
  on a tool/user, that is the right shape.
- Wake latency: 15.7 GB spread over 24 MLA-owning cards = ~650 MB/card;
  at PCIe 4.0 ×4 (~7 GB/s) ≈ 0.1 s/card in parallel, bounded by NVMe read
  (~2 s on one 7 GB/s drive, ~0.4 s if striped per node) — versus
  **15+ minutes** to re-prefill 1M tokens. Two to three orders of magnitude.

## Honest gaps before this is a production claim

1. **Weights, not KV, are the binding constraint.** Full K3 MXFP4 is
   ~1601 GB and does not fit 50 cards; this projection assumes a ~883 GB
   W2 checkpoint that does not exist yet (quality gate M0 + GSQ production
   still open). The offload result is independent of that and stands.
2. Measurements are on a 4-layer slice on 2 cards. At 93 layers / 50 cards /
   multi-node, three known upstream risks remain: PP per-rank bytes-per-block
   accounting (#50821, fix PR #50653 still open), multi-node NVMe locality,
   and the `_find_non_ssm_backend` PP bug above (mitigated by explicit
   `--block-size`).
3. Per-agent throughput at 100-way concurrency is unmeasured; K3's
   896-expert top-16 routing amortizes poorly with batch (see
   `llamacpp_q2_validation`), so expect expert-read-bound decode.
