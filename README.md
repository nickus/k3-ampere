# k3-ampere — Kimi K3 on RTX 3090 (sm_86), pipeline-parallel only

Port/validation effort for serving [Kimi K3](https://arxiv.org/abs/2607.24653)
(2.8T MoE, 69 KDA + 24 MLA layers, MXFP4 experts) with vLLM on Ampere consumer
GPUs. Upstream targets Hopper/Blackwell; this repo tracks exactly what sm_86
needs, what already falls back cleanly, and what a 3090 rig can and cannot do.

## Status (2026-07-30)

| Claim | Status | Evidence |
|---|---|---|
| KDA Triton kernels correct on sm_86 | ✅ verified on HW | `tests/models/kimi_k3/test_kda.py`: 33 passed / 6 skipped (skips = sm90+ FlashKDA paths) on 2×3090 |
| MXFP4 MoE backend selectable on sm_86 (Marlin) | ✅ verified on HW | `test_mxfp4_kernel_selection.py`: 5 passed on 2×3090 |
| vLLM boots K3 code path w/o kernel patches | ◑ static analysis (15-agent, adversarially verified); slice boot pending | every capability gate has an in-tree fallback: KDA→Triton, MoE→Marlin W4A16, MLA decode→TRITON_MLA, MLA prefill→FA2 |
| fp8 KV cache for K3 MLA on sm_86 | ❌ hard-blocked upstream | `TritonMLAImpl.__init__` raises below SM89; K3 fp8 path asserts require capability family 100. bf16 KV only: 1152 B/token/MLA-layer |
| Full K3 fits 50×3090 | ❌ does not fit | weights ≈ 1601 GB (1446 mxfp4 experts + 155 bf16 rest) vs 1160 GB usable. Floor ≈ 70 cards weights-only, ~74 usable. See `docs/CAPACITY_50x3090.md` |
| Spec decode under PP | MTP only, `VLLM_USE_V2_MODEL_RUNNER=1` | eagle3+PP and DSpark+PP raise in both runners; base checkpoint ships `num_nextn_predict_layers=0` (no MTP head!) |

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

- `--kv-cache-dtype fp8` → ValueError at init (SM89+ required).
- `additional_config.kda_prefill_backend=flashkda` → RuntimeError (sm90+).
- eagle3/dspark spec decode + PP → raises / drafter crash.
- MTP + PP without `VLLM_USE_V2_MODEL_RUNNER=1` → rank-0 `AttributeError: drafter`.
