# KV-offload + PP=2 + K3-hybrid validation — 2026-08-10 (2×3090)

Question (the load-bearing check before betting the RAM+NVMe KV-offload
serving plan): does a KV connector evict/restore KV **per-PP-stage** on a
**hybrid** K3 (KDA linear-attention state + MLA latent KV), and is a resumed
session correct?

## Result 1 — fp8-KV port under PP=2: PASS (independent of offload)

`--kv-cache-dtype fp8_ds_mla --pipeline-parallel-size 2` boots and generates;
greedy output is **token-identical to PP=1** (same slice, same prompt). The
fp8 MLA-KV path is correct under pipeline parallelism and per-stage memory
splitting. **This does not depend on any offload connector.**

## Result 2 — KV offload on K3-hybrid: NOT READY on either connector today

Both connectors fail on the **hybrid** nature of K3 (they cannot reconcile
the KDA linear-attention state with the MLA KV in one offload scheme):

| Connector | Config | Outcome |
|---|---|---|
| native `OffloadingConnector` | PP=2, bf16 KV | **AssertionError**: `tokens_per_block=512 not divisible by tokens_per_hash=4096. Hybrid models…` — the block/hash geometry assumption breaks on KDA+MLA. |
| native `OffloadingConnector` | PP=2, fp8_ds_mla | Same hybrid assertion (never reaches the 656B-page question). |
| `LMCacheConnectorV1` | PP=2, bf16 | **ValueError**: `Failed to promote local KV cache specs to one unified type` (`unify_hybrid_kv_cache_specs`) — can't unify KDA+MLA specs. |

Also: both connectors reject `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
unless `enable_cumem_allocator` — a launch-config gotcha for the runbook.

## Result 3 — the KDA-state resume risk is REAL (confirmed by LMCache's own docs)

The subtle bug flagged at the start of this task — a resumed session keeps
MLA-KV but loses KDA state → silently corrupted context — is corroborated by
LMCache's official hybrid-models guide, which **explicitly warns**:
> "Generation is **not bit-exact** between a cached and a fresh run …
> recommend score-level comparison, not token-level validation."

For a linear-attention layer that carries recurrent state, "not bit-exact
after restore" means the state restore is lossy. So even LMCache's supported
hybrid path (which needs `LMCacheMPConnector` + a separate `lmcache server`
+ `--mamba-cache-mode align` + `--separate-object-groups`, and does **not**
document PP support) trades correctness for cache reuse on hybrids.

## Verdict for the NVMe/offload serving plan

- **Do NOT depend on KV offload for the K3 serving plan as of 2026-08-10.**
  It is barely-alpha for hybrid models: native connector doesn't support
  hybrid at all; LMCache's hybrid path is single-node-shaped (no PP),
  needs a separate server process, and is not correctness-preserving.
- The upstream signal agrees: RFC #33689 lists "extend cross layers to
  hybrid models" as an *unchecked upcoming* item; issues #50821 (PP-rank
  block inconsistency), #46453/#43508 (hybrid + connector crashes),
  #50235 (K3 + OffloadingConnector actively being debugged) are all open.
- **The fp8-KV port stands on its own** and delivers the capacity win
  (656 vs 1152 B/token) with no offload dependency — that is the reliable
  lever for fitting long-context agents on ~50 cards.
- If single-node RAM offload is ever wanted, the one lead worth pursuing is
  LMCache `LMCacheMPConnector` at PP=1 with the documented mamba flags,
  accepting non-bit-exact resume — untested here (stopped on cost; PP
  unsupported makes it a poor fit for the multi-host rig anyway).

Track upstream (RFC #33689 + the issues above); revisit when "hybrid cross
layers" ships and a connector documents PP.
