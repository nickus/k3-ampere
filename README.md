# k3-ampere — Kimi K3 on RTX 3090 (sm_86), pipeline-parallel only

Port/validation effort for serving [Kimi K3](https://arxiv.org/abs/2607.24653)
(2.8T MoE, 69 KDA + 24 MLA layers, MXFP4 experts) with vLLM on Ampere consumer
GPUs. Upstream targets Hopper/Blackwell; this repo tracks exactly what sm_86
needs, what already falls back cleanly, and what a 3090 rig can and cannot do.

## → Start with [`JOURNAL.md`](JOURNAL.md)

The consolidated record: everything proven, everything disproven (including two
of our own retracted verdicts), every experiment's actual measurement, the
operational lessons, and the ranked open questions. This README is the
quickstart; the journal is the memory.

## Status (2026-08-11)

| Claim | Status | Evidence |
|---|---|---|
| KDA Triton kernels correct on sm_86 | ✅ verified on HW | `tests/models/kimi_k3/test_kda.py`: 33 passed / 6 skipped (skips = sm90+ FlashKDA paths) on 2×3090 |
| MXFP4 MoE backend selectable on sm_86 (Marlin) | ✅ verified on HW | `test_mxfp4_kernel_selection.py`: 5 passed on 2×3090 |
| vLLM boots K3 code path on sm_86 | ✅ verified on HW (PP=2) | Phase A bf16 + Phase B MXFP4-Marlin + CUDA graphs; 3 python patches (`patches/`); `results/2x3090_validation_2026-07-30.md` |
| Still works on current vLLM | ✅ verified on HW | **main = 0 patches, release 0.27.0 = 1 line**; `results/revalidation_vllm_0270_2026-08-11.md` |
| fp8 KV cache for K3 MLA on sm_86 | ✅ **we wrote it** | 656 B/token/layer vs 1152 → 1.75× capacity. Kernel 12/12, cosine 0.9999967, greedy parity exact, exact under PP=2. `fp8kv_k3_port/RESULTS.md` |
| KV offload to NVMe on hybrid + PP | ✅ verified on HW, **bit-exact** | max logprob delta 0.00000000 restoring from CPU and disk tiers, up to 28,417-token prompts. `results/kv_offload_PROVEN_2026-08-11.md` |
| Full K3 (MXFP4, 896 experts) fits 50×3090 | ❌ does not fit | ≈1601 GB vs 1160 usable. 1 MoE layer = 15.72 GB packed: fits a 3090, **two OOM** → ~93 cards. `docs/CAPACITY_50x3090.md` |
| **A K3 that does fit 50×3090** | ⚠️ exists, unvalidated | `runrunway/Kimi-K3-REAP-448experts` — **837 GB, same MXFP4 format we already serve**. Fits at 16.7 GB/card. Quality on K3 unproven. `docs/REAP_ROUTE.md` |
| Spec decode | ❌ **none available under PP** | K3 ships no MTP head (`num_nextn_predict_layers=0`); DSpark is the only option and hard-raises under PP. Root-caused + fix sketched: `docs/DSPARK_PP_BLOCKER.md` |
| Quality vs GLM-5.2 | ❓ **never measured** | The gate that decides everything. `docs/M0_RUNBOOK.md` |

## Layout

- `tools/make_slice_config.py` — builds tiny text-only slice configs
  (`KimiLinearForCausalLM`, 3 KDA + 1 MLA layer) that keep every
  layout-critical dimension at production value. Phases: **a** bf16 mechanics,
  **b** + `mxfp4-pack-quantized` (Marlin), **c** + 1 MTP layer.
  Weights via `--load-format dummy` — no checkpoint needed.
- `tools/k3_real_config.json` — reference copy of the real HF config.
- `configs/` — generated slice configs.
- `docs/` — gap analysis, capacity math, runbook.
- `tests/` — config-generator invariants (`pytest -q`).

## Quickstart (2×3090, PP=2)

```bash
mkdir k3-slice && cp configs/slice_a.json k3-slice/config.json
# tokenizer: reuse the real one (tokenizer.json etc. from the HF repo)
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve ./k3-slice \
  --trust-remote-code --load-format dummy \
  --pipeline-parallel-size 2 --tensor-parallel-size 1 \
  --kv-cache-dtype auto --max-model-len 4096 \
  --gpu-memory-utilization 0.9 --enforce-eager
```

Expected backend log lines on sm_86: `Using TRITON_MLA`,
`Using FLASH_ATTN MLA prefill backend`, no FlashKDA line.

## Known traps (do not)

- `--kv-cache-dtype fp8` → ValueError at init (SM89+ required). Our
  `fp8_ds_mla` is a *different* dtype and is carved out of that gate by `P3c`.
- `additional_config.kda_prefill_backend=flashkda` → RuntimeError (sm90+).
- DSpark spec decode + PP → `NotImplementedError` by design; see
  `docs/DSPARK_PP_BLOCKER.md` for the root cause and the fix sketch.
- MTP + PP without `VLLM_USE_V2_MODEL_RUNNER=1` → rank-0 `AttributeError: drafter`.
- **Hybrid + offload without an explicit `--block-size`** → misleading
  "need --enable-prefix-caching" assert even when it *is* enabled. At PP=50 this
  hits 26 of 50 ranks; always pass `--block-size`. Upstream
  [#51752](https://github.com/vllm-project/vllm/issues/51752).
- **vLLM 0.27.0 + hybrid + prefix caching** → engine dies on the first request
  (`index_fill_(): Expected dtype int64`). Fixed on main only; patch
  `idx_mapping.long()` if you pin the release. Upstream
  [#50947](https://github.com/vllm-project/vllm/issues/50947).
- Generating a slice from the upstream K3 config verbatim → vLLM ≥0.27.0 takes
  the multimodal path and the process is OOM-killed on a dummy vision tower.
  `tools/gen_slice_hf.py` now emits a flat, text-only config.
