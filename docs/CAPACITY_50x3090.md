# Can 50× RTX 3090 serve full Kimi K3? — No. The arithmetic.

All numbers below derive from the **real** `config.json`
(`tools/k3_real_config.json`, pulled from the HF repo 2026-07-30), not from
press releases. Re-run the arithmetic before any hardware decision.

## Model constants (text_config)

| field | value |
|---|---|
| layers | 93 = 69 KDA + 24 gated MLA (`full_attn_layers`: every 4th, plus 93) |
| hidden | 7168, heads 96 |
| MLA | kv_lora 512, q_lora 1536, nope 128, rope 64, v 128 → cache head 576 |
| KDA | 96 heads × 128 dim, conv 4, full-rank gate |
| MoE | 896 routed experts (top-16) **in a 3584 latent** (LatentMoE), expert FFN 3584→3072, +2 shared experts (bf16), first layer dense |
| quant | compressed-tensors `mxfp4-pack-quantized`, group 32 — **routed experts only**; ignore-list keeps self_attn, shared_experts, dense mlp, lm_head, vision in bf16 |
| vocab | 163840; `num_nextn_predict_layers` **= 0** (no MTP head ships) |

## Weights

- Routed experts: 896 × 3 × 3584 × 3072 = **29.6 B params/layer** × 92 MoE
  layers = **2.723 T params**. At MXFP4 (4 bit + E8M0 scale per 32 = 4.25
  bit = 0.53125 B/param): **1446 GB**.
- Everything else (≈77 B params: attention 34 B, shared experts 12 B,
  embeddings+head 2.3 B, dense layer, norms — all on the quant ignore-list):
  bf16 → **≈155 GB**.
- **Total ≈ 1601 GB.**

## Rig

50 × 24 GiB = 1288 GB raw; at `--gpu-memory-utilization 0.9` → **1160 GB
usable**. Deficit: **441 GB before the first KV byte**. Per-rank at PP=50:
1601/93 × 1.86 layers ≈ 32 GB/rank vs 23.2 usable — every rank is ~38% over.
PP topology cannot change aggregate; TP is excluded on this rig (PCIe).

## KV / state (what a working rig must also hold, bf16-only — fp8 KV is
hard-blocked on sm_86)

- MLA KV: 576 × 2 B = 1152 B/token/layer × 24 layers = 27.6 KB/token
  → 0.9 GB per 32K-token sequence.
- KDA recurrent state (fp32-forced): 96×128×128×4 ≈ 6.3 MB/seq/layer × 69
  layers ≈ **0.43 GB per sequence** regardless of length (linear attention:
  constant in T — the one thing that *helps* this rig).
- ~1.3 GB per concurrent 32K seq; 16 seqs ≈ 21 GB.

## Floors

- Weights-only: 1601 / 23.19 → **70 cards**.
- With a modest KV/state/runtime budget: **~74 cards**.
- Max feasible expert quantization on 50 cards: (1160 − 155 − ~90) ≈ 915 GB
  for 2.723 T expert params → **≈2.7 bit/param** — no such path exists in
  vLLM (Marlin: int4/int8 + MXFP4 only).

## Options, in order of realism

1. **Grow to ~72–74 cards** — everything in this repo then applies as-is.
2. **unsloth UD-Q2_K_XL GGUF (861 GB) via llama.cpp** — fits 50 cards, but:
   no vLLM (KDA GGUF support in llama.cpp is immature), PP across 50 GPUs
   with llama.cpp's scheduler, expect low throughput. Unvalidated.
3. **Sub-3-bit expert kernel for vLLM (Marlin-style W2/W3)** — multi-kLOC
   CUDA project. Out of scope until 1–2 are exhausted.
4. Serve **Kimi-Linear-48B** (same code path, fits trivially) for stack
   validation and as an interim production model.

## Sanity anchors

- vLLM day-0 blog: minimum deployment = 8× B300 (8×288 GB = 2304 GB) or
  16× B200 (16×180 = 2880 GB) — both comfortably above 1601 GB. Consistent.
- unsloth Q2 at 861 GB ≈ 2.46 bit/param over 2.8 T — consistent with a
  2-bit-dominant mix.
