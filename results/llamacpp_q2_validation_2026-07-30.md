# llama.cpp Q2 route — validation on 2×3090 (2026-07-30)

Target: unsloth UD-Q2_K_XL (861 GB) on the fleet. K3 support = llama.cpp
**PR #26185** (not in master; unsloth points at their fork of the same work).

## What was validated on hardware

| Step | Result |
|---|---|
| Build PR #26185 @cf67f0d, CUDA sm_86 | ✅ (one import fix for new transformers: `bytes_to_unicode` moved to `convert_slow_tokenizer`) |
| Synthetic slice in REAL checkpoint layout (nested config, `language_model.*`, shape formulas verified 45/45 vs real shard headers) | ✅ `tools`-style generator on box (`gen_gguf_slice.py`) |
| `convert_hf_to_gguf.py` accepts the slice | ✅ 107 tensors, 767 MB f16 GGUF |
| `llama-quantize` → Q2_K | ✅ 222 MB ≈ 2.3 bpw |
| Graph executes on **CPU** | ✅ but anomalously slow (pp64 10.0 / tg32 6.7 t/s on 64-core EPYC; PR thread reports ~55 — worth an upstream look, irrelevant for GPU serving) |
| Graph executes on **GPU sm_86** | ✅ pp512 **62,061** t/s, tg64 **962** t/s (slice-scale) |
| 2-GPU `--split-mode layer` | ✅ pp 61,246 / tg 981 — no penalty at slice scale |

## Extrapolation to full K3-Q2 on ~50×3090 (model, not measurement)

Active bytes/token/layer at UD-Q2-ish ≈ 210 MB (16 routed experts ≈165 MB +
shared + attn at higher bpw) → memory floor ~0.26 ms/layer at ~800 GB/s;
plus MoE kernel-launch overhead (llama.cpp `mul_mat_id`) and 50+ PCIe hops
(~14 KB each, latency-bound). **CORRECTED after source-verified deep-check (2026-07-30 late):** the
"no vLLM-class continuous batching" sentence was WRONG — llama-server has
iteration-level continuous batching on by default, ID-aware MMQ MoE
batching on Ampere, GPipe-style prefill pipelining, and a modern cross-slot
prompt cache. What limits aggregate is K3 itself: 896-expert top-16 routing
saturates the expert-read union slowly (distinct experts ≈
896·(1−(1−16/896)^B): 16@B1, 224@B16, 830@B128), and layer-split
serializes the fleet to ONE card's bandwidth (pipelining is hard-disabled
when any RPC device is present: `ggml-rpc.cpp:1874` async=false).
Honest model: single-stream **10–20 tok/s** (downgraded from 20–33; the
8×B200=16.6 datapoint is 25× off its BW floor — overhead-bound, and 50
stages + RPC hops make that worse), aggregate ~45–70 tok/s at 16–32 slots
≈ **2–5 tok/s per agent**. Plus a correctness landmine for our exact
topology: Issue #20052 (dual 3090 no-P2P layer-split → incoherent output
>2048 ctx, open) — see M0_RUNBOOK pre-flight.

## Verdict

Mechanically **feasible** (arch verified on sm_86, conversion pipeline
verified, layer-split works); strategically **weak**: expected ~20–30 tok/s
single-stream and quality ≤ GLM-5.2 INT4 (see `W23_KERNEL_VERDICT.md`).
Use it for exactly one thing: **gate M0** — download UD-Q2_K_XL onto the
volume, spread across available hardware, A/B against a strong INT4 baseline on representative evals.
