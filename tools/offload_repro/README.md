# KV-offload reproduction scripts

Scripts behind `results/kv_offload_PROVEN_2026-08-11.md`, salvaged into the
repo after box 47416893 (2×3090) was destroyed on 2026-08-11.

| Script | What it does |
|---|---|
| `longctx4.sh` | The long-context restore sweep that produced the results table: unique random-token stream per size, 5 same-size filler prompts to evict from GPU, `posix_fadvise(DONTNEED)` on every tier file, restore call, then a GPU-resident call as upper bound. Reads real disk I/O from `/proc/<pid>/io`. |
| `longctx3.sh` | Earlier version. Kept because it contains the server launch block (`srv5.sh`) that `longctx4.sh` reuses. Its sweep loop has a methodology bug — sizes shared a `random.seed`, so longer prompts got a prefix hit from shorter ones and their "cold" times are invalid. |
| `box7_offload_gate.sh`, `box7_offload_test.sh` | Earlier offload gates (connector boot, CPU tier, forced eviction). |

## Honest gap in the evidence chain

Only `results/logs/longctx_sweep_2026-08-11.log` was pulled before the box was
destroyed. The raw server logs for the **earlier** offload runs — the CPU-tier,
NVMe-tier and fp8 rows in the PROVEN table, each with
`max_logprob_delta 0.00000000` — went with the box. Those results were read
off the console at the time and are reported accurately, but they cannot be
re-derived from a stored artifact; re-running these scripts is the only way to
reproduce them.

The pinned build they ran on (vLLM `38a267cdd`, 2026-07-30, plus our fp8 and
`.long()` patches) is also gone. Since the follow-up task is to revalidate on
current main anyway, that build's loss costs a rebuild-at-SHA (hours) only if
someone needs the exact original environment.

**Rule going forward:** pull the whole evidence bundle — every `srv*.log` and
result file — not just the last output, before destroying a box.

## Rebuilding a box to run these

`tools/gen_slice_hf.py` (K3 slice), `tools/hand_patches.py` and
`fp8kv_k3_port/fp8kv_k3/apply_vllm_patches.py` bring a fresh vast box to a
serving state in roughly 30 minutes. Both patch scripts print DONE even when
they apply zero patches on a drifted tree — always verify the patch count they
report against the tree you are on.
