# A K3 checkpoint that fits 50 cards already exists — in the format we already serve

Found 2026-08-11. This likely **supersedes the "produce a W2 checkpoint
ourselves" plan** (task #31), which was the single blocker on fitting K3 into
the rig.

## The artifacts (sizes and formats verified directly against the HF API)

| Repo | Size | Format | Experts | Layers |
|---|---|---|---|---|
| `runrunway/Kimi-K3-REAP-448experts` | **837.1 GB** | `compressed-tensors` / `mxfp4-pack-quantized` | 448 (top-16) | 93 |
| `runrunway/Kimi-K3-REAP-384experts` | **733.7 GB** | same | 384 (top-16) | 93 |

Architecture string, layer count, top-k and quantization format are **identical
to stock `moonshotai/Kimi-K3`**. Only `num_experts` differs. That is the whole
point: this is not a new format to port, it is the format
`k3-ampere` already proved runs on sm_86 (Phase B — MXFP4 → Marlin W4A16,
PIECEWISE CUDA graphs 51/51, PP=2 on 2×3090).

## Why this is better than the W2 route we were planning

We were going to push K3 down to **2 bits** to reach ~883 GB. REAP reaches
**837 GB while staying at 4 bits**, by removing experts instead of precision.

REAP ("Router-weighted Expert Activation Pruning", Cerebras,
[arXiv:2510.13999](https://arxiv.org/abs/2510.13999),
[code](https://github.com/CerebrasResearch/reap)) prunes experts by saliency and
keeps the router's independent control — the paper's argument is that pruning
beats expert *merging*, which collapses the router's functional subspace.
Published retention at **50% pruning**, on other models:

| Model | Retention at 50% pruned |
|---|---|
| Qwen3-30B-A3B | 95.9% coding average |
| GLM-4.5-Air | 94.1% LiveCodeBench |
| Qwen3-Coder-480B-FP8 | 97.6% non-agentic, 96.7% SWE-bench |
| Kimi-K2-Instruct-W4A16 (1T) | "near-lossless" on code generation, *combined with* quantization |

**Our workload is coding agents.** These are exactly the benchmarks that hold up
under REAP, and the K2 row is the closest analogue to what we would run:
pruning stacked on top of 4-bit quantization, reported near-lossless on code.

Compare with the 2-bit route, where quality is unmeasured and the only known
A/B (a homemade K3-Q2 vs GLM-5.2 Q5) went against us.

## Capacity arithmetic

REAP-448 halves the MoE layer: 15.72 GB packed at 896 experts → **≈7.9 GB at
448**, so **2 MoE layers per 24 GB card** — the same "2 layers/card" shape the
W2 route was chasing, at double the bit width.

| Config | Weights | Per card | Free for KV/activations |
|---|---|---|---|
| REAP-448 on 47 cards | 837.1 GB | 17.8 GB | ~5 GB → ~235 GB total |
| REAP-448 on 50 cards | 837.1 GB | 16.7 GB | ~6.4 GB → **~320 GB total** |
| REAP-384 on 50 cards | 733.7 GB | 14.7 GB | ~8.5 GB → ~425 GB total |

At `fp8_ds_mla` (15.7 KB per token across the 24 MLA layers), 320 GB of resident
KV ≈ 20M tokens live on the GPUs — with the NVMe tier holding the sleeping
sessions. REAP-384 buys a further 105 GB of resident KV for 8% more pruning.

## What is NOT established

1. **The REAP paper does not test K3.** It predates the model; every "REAP'd K3"
   on HF is a third party applying the published formula. `runrunway`'s specific
   artifact has no published evaluation.
2. **Quality on Russian.** The retention numbers above are coding benchmarks.
   REAP calibration is language-sensitive — one vendor publishes an explicitly
   Japanese-calibrated variant whose kept-expert set overlaps the English/code
   set by only ~473 of 640. If non-English matters, calibration language is a
   real variable, not a detail.
3. **Boot on our stack, with real weights.** Unverified for this artifact. But
   note our own slice has always run with a *reduced* expert count (8 vs 896)
   through the stock `kimi_k3` code — so a smaller `num_experts` is structurally
   nothing new; vLLM reads it from config.

## Naming trap (bit us while reading these repos)

`REAP640` / `REAP576` / `REAP-320` name the **kept** expert count.
`REAP50` names the **pruned percentage** (so REAP50 = 448 kept). Do not compare
vendors by the number in the name.

## Recommended next step

Download-free check first: our existing slice machinery already exercises the
reduced-expert path. The real gate is **M0 quality on hermes-agent scenarios**,
and the right contender to put in it is now **REAP-448 MXFP4**, not a homemade
2-bit quant — it is bigger-bit, already exists, and has published evidence at
this exact pruning ratio on coding workloads.
