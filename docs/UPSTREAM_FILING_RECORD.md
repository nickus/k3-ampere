# Upstream filing record — hybrid + KV-offload + PP findings (2026-08-11)

Three candidate findings came out of bringing up `OffloadingConnector` on a
hybrid model (Kimi-K3: KDA linear attention + MLA) with
`--pipeline-parallel-size 2` on 2×RTX 3090, vLLM 38a267cdd.

**Each was re-verified against fresh main (1a1727330a) before filing. Two did
not survive.** One issue filed, not three.

| # | Finding | Verdict | Action |
|---|---|---|---|
| 1 | `index_fill_(): Expected dtype int64` in `mamba_hybrid.py` chunked-prefill path | **Already fixed upstream** | Not filed |
| 2 | Hybrid block-size alignment skipped on PP ranks with no attention layer | **Still present** | Filed: [vllm#51752](https://github.com/vllm-project/vllm/issues/51752) |
| 3 | `cpu_bytes_to_use` vs `num_cpu_blocks`, tiering schema undiscoverable | **Now documented upstream** | Not filed |

## CORRECTION 2026-08-11 (later the same day): #1 is fixed on main but NOT in the newest release

vLLM **0.27.0 was released 2026-08-10** — one day before this work. Installing
it on a clean box and inspecting the tree shows it **still contains
`index_fill_(0, idx_mapping, …)`** and has **none** of the Triton kernels that
replaced it. So the fix exists only on unreleased main; every user on the
current stable release still hits the crash.

"Already fixed upstream, not our bug to report" was therefore too strong. The
accurate statement: fixed on main, present in 0.27.0. For us this is
load-bearing — if we pin a release rather than main, our `.long()` patch is
still required.

## Why #1 was dropped

Our build crashed in `postprocess_state`:

```python
self.num_accepted_tokens_gpu.index_fill_(0, idx_mapping, max(num_sampled, 1))
```

with an int32 `idx_mapping`. Main no longer has this call at all. Both branches
now dispatch to Triton kernels — `_scatter_num_accepted_kernel` and
`_fill_num_accepted_kernel` — which read the index with `tl.load`, so int32 is
fine, and which skip `-1` rows with an explicit comment: *"idx_mapping may
contain -1 sentinels (filtered rows) under PP"*. Upstream fixed this while
solving the adjacent PP-filtering problem. Our local `.long()` patch stays on
the pinned build; it becomes unnecessary when we move to main.

## Why #3 was dropped

`docs/features/kv_offloading_usage.md` (328 lines) now documents the full
`kv_connector_extra_config` table including `cpu_bytes_to_use`, the
`TieringOffloadingSpec` schema, `secondary_tiers`, the `fs` tier with its
on-disk layout and cross-process sharing, plus tuning tips. The only residue is
that unknown keys (`num_cpu_blocks`) are silently ignored instead of rejected —
too thin to file on its own.

**Worth acting on for us:** that doc describes `blocks_per_chunk` as the
"alternative to `block_size` for models whose KV cache groups have different
block sizes". That is precisely the hybrid case, and it may be a better lever
than our explicit-`--block-size` workaround, which forces very large blocks
(896–10,112 tokens at K3 scale) and wastes cache on short prompts. Untested.

## What was filed (#51752)

`Platform.update_block_size_for_backend` runs **per worker process** after
`load_model()` (`v1/executor/multiproc_executor.py:677`, `ray_executor.py:366`,
`uniproc_executor.py:75`) and early-returns when `_find_non_ssm_backend`
finds no attention layer among the layers *this rank* instantiated
(`platforms/interface.py:624-626`). A PP stage that is all linear-attention
therefore never runs `_align_hybrid_block_size` and keeps the unaligned
`block_size` / `mamba_block_size`, while stages owning an attention layer raise
both. Specs merge by layer name — disjoint across PP stages — so the
"specs differ across workers" check never fires, and the failure surfaces as a
misleading divisibility assert telling you to enable prefix caching you already
enabled.

Distinct from #50821/#50653, which is `num_cpu_blocks` diverging one level
later. Workaround (`--block-size N` explicitly, set pre-fork) is in the issue.

## Lesson

Two of three "upstream bugs" were stale within days — vLLM main moves fast
enough that findings decay. Verify against a fresh checkout immediately before
filing, every time, and cite the exact SHA in the issue.
