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

## It also removes our dependency on a stalled upstream PR

The W2 route needed 2-bit MoE serving, which upstream gates behind
[PR #48918](https://github.com/vllm-project/vllm/pull/48918)
("[CT] Support Humming for WNA16 MoE"). As of 2026-08-11 that PR is **open,
last touched 2026-07-22, flagged `mergeable_state: dirty` for merge conflicts,
with seven requested reviewers and zero approvals**. Our whole W2 plan sat on
top of it.

REAP-448 is **MXFP4 → Marlin W4A16**, a path already in released vLLM and
already exercised on our own hardware. No dependency on #48918 at all.

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
2. **Which calibration set the artifact was pruned with.** REAP decides *which*
   experts to keep from a calibration corpus, and that choice is corpus-
   sensitive: one vendor publishes a Japanese-calibrated variant whose kept-set
   overlaps the English/code one by only ~473 of 640 experts. Our workload is
   English + mainstream programming languages, so this is not a risk to manage —
   it is a **selection criterion**: take an English/code-calibrated cut, and the
   published retention numbers (which are themselves English coding benchmarks)
   apply directly. Confirm `runrunway`'s calibration corpus before committing.
3. **Boot on our stack, with real weights.** Unverified for this artifact. But
   note our own slice has always run with a *reduced* expert count (8 vs 896)
   through the stock `kimi_k3` code — so a smaller `num_experts` is structurally
   nothing new; vLLM reads it from config.

## The published evidence stops at 50% pruning — this decides 448 vs 320

Checked against both the paper (arXiv:2510.13999) and the Cerebras blog: **all
published REAP numbers are at 25% and 50% compression. There is nothing above
50%.** No cliff data, no graceful-degradation curve, nothing.

That maps directly onto the two candidates:

| Candidate | Experts kept | Pruned | Position vs published evidence |
|---|---|---|---|
| REAP-448 | 448 / 896 | **50%** | **exactly the validated operating point** (94–97% coding retention) |
| REAP-320 | 320 / 896 | **64%** | **off the published curve entirely** — extrapolation |

So the honest difference between them is not a percentage. It is that one sits
where the method is measured and the other is past the last data point, on a
compression axis where nobody has published whether degradation is smooth or a
cliff.

Two further confounds make a direct comparison impossible without running it:
`runrunway`'s 448 and Blackfrost's 320 come from **different vendors with
different calibration corpora**, and the 320 is additionally **abliterated**
(refusal directions removed at the weight level — an unrelated modification the
vendor calls "the contract") and **self-declared EXPERIMENTAL, with observed
decode, coherence and under-load serving bugs**.

Against that, 320 has one real argument in its favour: it was pruned against
**measured coding routing load**, and its surviving expert tensors are bit-exact
copies of the parent MXFP4 packs. Calibrating on the target domain is exactly
what should buy back headroom at a higher ratio — but that is a hypothesis, not
a measurement.

**Consequence for M0**: this is not a documentation question, it is the
experiment. Run both.

## Naming trap (bit us while reading these repos)

`REAP640` / `REAP576` / `REAP-320` name the **kept** expert count.
`REAP50` names the **pruned percentage** (so REAP50 = 448 kept). Do not compare
vendors by the number in the name.

## What M0 would cost (researched 2026-08-11, live marketplace prices)

For a ~6 hour quality run:

| Route | Instance | $/hr | ~total |
|---|---|---|---|
| CPU + GGUF, tight RAM | vast.ai, 1032 GB RAM | $3.91 | **≈$28** |
| CPU + GGUF, comfortable RAM | vast.ai, 1156 GB RAM | $13.28 | ≈$90 |
| GPU + safetensors (the real artifact) | vast.ai 4×B300, 1100 GB VRAM | $30.25 | ≈$200 |
| GPU + safetensors | 8×H200 (vast/RunPod) | $31.58–35.12 | ≈$226–232 |

vast.ai bills per second and has an interruptible/bid tier — an opportunistic
1032 GB-RAM host was seen at $0.80/hr during the survey (~$8 for the whole job),
but that is a snapshot, not a plan.

The honest trade: **$28–90 tests a GGUF proxy** (e.g. `mmnga-o/Kimi-K3-REAP50-UD-gguf`,
which is the same 448-expert cut but *further* squeezed to Q2 — two lossy steps
stacked), while **≈$200 tests the artifact we would actually serve**
(REAP-448 at MXFP4). Given that the entire point of M0 is deciding whether to
commit to a large hardware purchase, testing the real artifact is the better
$200.

## Recommended next step

Download-free check first: our existing slice machinery already exercises the
reduced-expert path. The real gate is **M0 quality on real agentic coding sessions**,
and the right contender to put in it is now **REAP-448 MXFP4**, not a homemade
2-bit quant — it is bigger-bit, already exists, and has published evidence at
this exact pruning ratio on coding workloads.
