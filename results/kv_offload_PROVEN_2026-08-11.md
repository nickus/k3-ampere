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
   `kv_connector_extra_config` (`v1/kv_offload/cpu/spec.py:81-84`). Unknown
   keys are silently ignored, so `num_cpu_blocks` looks accepted and then
   fails on the missing required key.
4. **A crash we patched locally, already fixed upstream.**
   Our build (38a267cdd) had `v1/worker/gpu/model_states/mamba_hybrid.py:306`
   calling `index_fill_(0, idx_mapping, …)` with an **int32** index →
   `RuntimeError: index_fill_(): Expected dtype int64 for index`, on the
   chunked-prefill path that align mode *requires*. We patched it with
   `idx_mapping.long()` and it works.

## Verified against fresh main (1a1727330a, 2026-08-10) before filing upstream

Re-checking each finding against current main before opening issues killed two
of the three. Recording that, because the earlier version of this file implied
we had found three upstream bugs:

| Finding | Status in main |
|---|---|
| index_fill_ int32 crash (#4 above) | **already fixed.** Both branches now go through Triton kernels — `_scatter_num_accepted_kernel` / `_fill_num_accepted_kernel` — which take int32 and handle `-1` sentinels, with an explicit comment about PP. Not our bug to report. |
| `cpu_bytes_to_use` undocumented (#3) | **mostly documented.** `docs/features/kv_offloading_usage.md` (328 lines) now carries a full `kv_connector_extra_config` table, the tiering schema, the `fs` tier and its on-disk layout. Only "unknown keys are silently ignored" survives, which is too thin to file. That doc also documents `blocks_per_chunk` — "alternative to `block_size` for models whose KV cache groups have different block sizes" — i.e. a knob aimed squarely at hybrids that we did not know about and have not tried. |
| PP alignment skipped on attention-less ranks (#2) | **still present**, `platforms/interface.py:624-626`. Filed as [vllm#51752](https://github.com/vllm-project/vllm/issues/51752). |

The call site matters for #2 and was re-confirmed: `update_block_size_for_backend`
runs **inside each worker process** after `load_model()`
(`v1/executor/multiproc_executor.py:677`, `ray_executor.py:366`,
`uniproc_executor.py:75`), so every rank aligns against only its own layers.
It is a distinct bug from #50821/#50653 (which is about `num_cpu_blocks`
diverging across ranks, one level later).

## Long-context restore sweep (2×3090, PP=2, bf16 KV, 256 MB CPU tier + NVMe fs tier)

GPU KV capped at 160 blocks × 512 = 81,920 tokens. Each row: cold prefill →
5 same-size filler prompts (evicts it from GPU) → `posix_fadvise(DONTNEED)` on
every tier file (drops page cache without root; `drop_caches` is unavailable in
the container) → restore call → one more call to read the GPU-resident upper
bound. `diskMB` is the delta of the server processes' `read_bytes` in
`/proc/<pid>/io`, i.e. bytes that actually came off the drive.

| words | prompt tok | cold s | restore s | GPU-hit s | cold tok/s | restore tok/s | speedup | disk MB | MB/s | max Δlogprob |
|---|---|---|---|---|---|---|---|---|---|---|
| 1400 | 4,063 | 0.08 | 0.04 | 0.04 | 52,327 | 97,654 | 1.87× | 0.0 | – | **0.000000** |
| 2800 | 8,107 | 0.08 | 0.06 | 0.06 | 100,476 | 144,730 | 1.44× | 0.0 | – | **0.000000** |
| 5600 | 16,247 | 0.12 | 0.10 | 0.07 | 138,274 | 167,004 | 1.21× | 0.0 | – | **0.000000** |
| 9800 | 28,417 | 0.21 | 0.16 | 0.12 | 132,937 | 173,206 | 1.30× | 79.0 | 482 | **0.000000** |

Tier total after the sweep: 939 MB in 797 files on NVMe.

**What this establishes.** Bit-exactness holds all the way to 28,417 tokens —
`max Δlogprob = 0.00000000` at every size, including the row that provably came
off the disk. The KDA recurrent state survives a full evict/restore round trip
at long context, which was the load-bearing correctness question.

**What this does NOT establish — read this before quoting the speedups.** The
1.2–1.9× numbers are a *floor artifact of the test model*, not the expected
win. This is a 4-layer dummy-weight slice whose prefill runs at ~130k tok/s;
there is almost nothing to save by skipping it. Real K3 is 93 layers with 104B
active parameters, so prefill costs orders of magnitude more per token, while
restore cost scales only with KV bytes. The speedup on the real model is
`prefill_time / restore_time` with a numerator that grows enormously and a
denominator that grows only with layer count — this slice cannot measure it.

**Tiering behaved correctly.** Only the largest prompt read from disk; the
smaller ones were served by the 256 MB CPU tier. That is the intended
RAM-first / NVMe-overflow cascade, confirmed rather than assumed.

**On the 482 MB/s.** That is a shared vast.ai virtual disk under a live
inference process, so treat it as a floor on plumbing throughput, not as NVMe
bandwidth. A dedicated Gen4 ×4 drive is ~7 GB/s. Extrapolating a 1M-token K3
session (fp8: 656 B × 24 MLA layers × 1M ≈ 15.7 GB, plus ~448 MB KDA state):
≈ 33 s at the measured 482 MB/s, ≈ 2.3 s on one dedicated 7 GB/s drive, and
well under a second when the read is striped across per-node drives and the 24
MLA-owning ranks pull their shares in parallel. Against re-prefilling 1M tokens
this remains two to three orders of magnitude cheaper — but the per-node NVMe
layout is now a **design requirement**, not a detail.

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

0. **These results were produced on a vLLM pinned 11 days behind main**
   (38a267cdd, 2026-07-30; main was 1a1727330a, 2026-08-10 — at least 50
   commits). The pin was deliberate: the from-source build on that box was
   expensive to get working, and the fp8-KV port and patches were developed
   against it. The cost showed up immediately — two of three "upstream bugs"
   we thought we had found were already fixed in main. Nothing here is
   *invalidated* by the age (the bit-exact restore, the fp8-KV port, and the
   PP alignment bug all hold on main), but **the offload stack must be
   re-validated on a current main before the rig depends on it**, and the
   local `index_fill_ .long()` patch must be dropped when we move.

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
