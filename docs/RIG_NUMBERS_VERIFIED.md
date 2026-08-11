# Rig arithmetic, verified against the real K3 config (2026-08-11)

Everything here is computed from `moonshotai/Kimi-K3/config.json` as fetched on
2026-08-11, not from memory or estimates. Constraint: 50 × RTX 3090 = 1200 GB
raw, ~1160 GB usable, sm_86, **pipeline-parallel only**.

## Model shape (fetched, not assumed)

| Field | Value |
|---|---|
| `num_hidden_layers` | 93 |
| `full_attn_layers` | 24 (every 4th: 4, 8, 12, … 92) |
| KDA layers | 69 |
| `num_experts` / per token | 896 / 16 |
| `hidden_size` | 7168 |
| KDA `num_heads` / `head_dim` | 96 / 128 |
| `max_position_embeddings` | 1,048,576 |
| `num_nextn_predict_layers` | **0 — no MTP head ships** |

## Per-session memory at 1M context (now exact, was an estimate)

KDA recurrent state is `96 heads × 128 × 128 × 4 B` = **6.29 MB per layer**,
× 69 layers = **434 MB per session**, and it is **constant in context length** —
that is the whole point of linear attention.

| Component | @1M tokens |
|---|---|
| MLA KV, `fp8_ds_mla` (656 B × 24 layers) | **15.74 GB** |
| MLA KV, bf16 (1152 B × 24 layers) | 27.65 GB |
| KDA state (context-independent) | 0.43 GB |
| **Total per sleeping 1M session (fp8)** | **≈ 16.2 GB** |

100 such sessions ≈ **1.62 TB** → fits a 4 TB NVMe tier. The fp8-KV port buys
1.76× here, and it is the difference between 1.62 TB and 2.81 TB.

## The PP degree makes our own upstream bug the common case

vLLM splits layers with `get_pp_indices` (remainder spread over all but the
last partition). Counting stages that own **no** attention layer — the ones
that silently skip `_align_hybrid_block_size`, i.e. [vllm#51752](https://github.com/vllm-project/vllm/issues/51752):

| PP size | stages with no attention layer |
|---|---|
| 25 | 2 of 25 (8%) |
| **47** (W2 route, 2 layers/card) | **23 of 47 (49%)** |
| **50** | **26 of 50 (52%)** |
| 93 (full MXFP4, 1 layer/card) | 69 of 93 (74%) |

At PP=2 this reads like a corner case; at rig scale **more than half the ranks
take the early return**. Consequence: `--block-size N` is not an optional
workaround for us, it is mandatory in every launch. Posted upstream as severity
data.

## Capacity, restated

- Full K3 MXFP4 ≈ 1601 GB — **does not fit** 1160 GB. Needs ~93 cards at
  1 MoE layer (15.72 GB packed) per card.
- W2 (~883 GB) → 2 layers/card → ~47 cards. This is the route, and the
  checkpoint does not exist yet (task #31).
- New external candidates found 2026-08-11 (sizes from the HF API):
  QuantTrio/Kimi-K3-Cubic-2.5Bit **965 GB safetensors**,
  vessl/Kimi-K3-W4AFP8 1484 GB, RedHatAI/Kimi-K3 1561 GB,
  vellum-ai/Kimi-K3-W3A16-g64 1291 GB. Only the Cubic one is under 1160 GB;
  whether vLLM can serve that format is the open question.

## Speculative decoding: none, currently

K3 ships no MTP head, so the GLM-5.2 lever does not transfer. DSpark is the
only option, a draft exists as of 2026-08-11, vLLM supports it — **but it
raises `NotImplementedError` under pipeline parallelism**. See
`DSPARK_PP_BLOCKER.md`. Until that is solved, any throughput projection for the
100-agent swarm that assumes speculation is unfounded.
