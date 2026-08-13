# k3-ampere

**Running Kimi-K3 on RTX 3090s.** Kimi-K3 is a 2.8-trillion-parameter model.
vLLM supports it on Hopper and Blackwell GPUs. This repo is about making it work
on Ampere consumer cards (sm_86) instead — cheap, second-hand, and everywhere.

Everything here was tested on real GPUs. When something was only read from
source code and not executed, it says so.

---

## The short version

Three questions matter when you try this:

1. **Does the model even run on these cards?** Yes. No new CUDA kernels needed —
   vLLM falls back to Triton and Marlin paths that already work on Ampere. It
   takes three small Python patches.
2. **Does it fit?** Not the full model: it needs about 1601 GB and 50 cards give
   you about 1160 GB. But a pruned version that fits **already exists**, in the
   same format we already serve.
3. **Can you keep 100 agents alive at once?** Yes — their KV caches can sleep on
   NVMe and come back **bit-for-bit identical**. This was the main open risk and
   it is now closed.

**Speculative decoding now works on this setup** — measured 1.577× decode on
real weights under 8-stage pipeline parallelism — but it took 13 patches and two
upstream bugs (both filed, one PR up). What is still open: making CUDA graphs
help K3 (they do not today), and quality after pruning. See below.

---

## What works today

| | Status | Evidence |
|---|---|---|
| Kimi-K3 runs on sm_86 (RTX 3090) | ✅ tested on GPUs | Bf16 and MXFP4 both serve at PP=2 (CUDA graph capture verified at PP=1), 3 Python patches → [`results/2x3090_validation_2026-07-30.md`](results/2x3090_validation_2026-07-30.md) |
| Still works on today's vLLM | ✅ tested on GPUs | **0 patches on `main`, 1 line on release 0.27.0** → [`results/revalidation_vllm_0270_2026-08-11.md`](results/revalidation_vllm_0270_2026-08-11.md) |
| fp8 KV cache for K3 on Ampere | ✅ we wrote it | 656 bytes/token/layer instead of 1152 → **1.75× more context per card**. Cosine 0.9999967, outputs identical to bf16 → [`fp8kv_k3_port/RESULTS.md`](fp8kv_k3_port/RESULTS.md) |
| KV cache offload to RAM and NVMe | ✅ tested on GPUs | Hybrid model + pipeline parallel + disk tier, **restored bit-for-bit** (logprob difference 0.00000000) → [`results/kv_offload_PROVEN_2026-08-11.md`](results/kv_offload_PROVEN_2026-08-11.md) |
| **Speculative decoding under pipeline parallelism** | ✅ tested on GPUs | Real DSpark draft on a real-weight 24-expert K3 slice (141 GB, all 93 layers), PP=8 on 8×3090: **172 → 109 ms/token, 1.577×**, greedy parity 7/8 (the 8th flips from batch shape alone, speculation off) → [`results/specdec_pp4_FIXED_2026-08-12.md`](results/specdec_pp4_FIXED_2026-08-12.md) |
| 2-bit and 3-bit weights on a 3090 | ✅ tested on GPUs | 2-bit costs no extra time vs 4-bit; a real 3-bit checkpoint serves after repacking, cosine 1.000000 → [`results/w3_validation_2026-08-10.md`](results/w3_validation_2026-08-10.md) |

## What does not work yet

| | Problem |
|---|---|
| Full Kimi-K3 on 50 cards | Does not fit. ~1601 GB against ~1160 GB usable. One MoE layer is 15.72 GB, so one layer per card → you would need ~93 cards. |
| CUDA graphs helping K3 | On GLM-4.5-Air graphs cut decode **3.41×** — on K3 they buy **~1%**, because PIECEWISE capture leaves attention outside the graph and 69 of K3's 93 layers are KDA linear attention. Forcing the KDA path into the graph is the biggest untried lever. |
| Speculative decoding out of the box | Two upstream bugs block it: a config-time `SupportsPP` check on the draft ([#52069](https://github.com/vllm-project/vllm/issues/52069), our PR [#52117](https://github.com/vllm-project/vllm/pull/52117)) and silent output corruption without async scheduling ([#52071](https://github.com/vllm-project/vllm/issues/52071), fix branch shared). With those patched it works — see the table above. |
| Quality after shrinking the model | **Never measured.** This is the question that decides whether any of this is worth doing. |

---

## The checkpoint that changes things

You do not have to squeeze K3 down to 2 bits to fit it. Someone already pruned
half the experts and kept 4-bit weights:

| Checkpoint | Size | Fits 50×3090? |
|---|---|---|
| `moonshotai/Kimi-K3` (original, MXFP4) | ~1601 GB | ❌ |
| **`runrunway/Kimi-K3-REAP-448experts`** | **837 GB** | ✅ 16.7 GB per card, ~320 GB left for KV cache |
| `runrunway/Kimi-K3-REAP-384experts` | 734 GB | ✅ with more room to spare |

These use the **exact same format** we already serve (`compressed-tensors`,
`mxfp4-pack-quantized`) — same architecture, same 93 layers, only fewer experts.
Nothing new to port.

Pruning experts (REAP, [arXiv:2510.13999](https://arxiv.org/abs/2510.13999))
keeps coding quality much better than crushing precision: published numbers at
50% pruning are 95.9% retention on coding for Qwen3-30B and 96.7% on SWE-bench
for Qwen3-Coder-480B.

**Two things to check before you trust it.** That paper never tested K3, and
this specific checkpoint has no published evaluation. Also check what corpus it
was pruned with — REAP picks which experts to keep from a calibration set, and
an English/code cut and a Japanese cut can differ by hundreds of experts. Treat
it as promising, not proven. Details:
[`docs/REAP_ROUTE.md`](docs/REAP_ROUTE.md).

---

## Numbers worth knowing

Taken from the live model config, not estimated.

- **93 layers**: 69 use linear attention (KDA), 24 use MLA (every 4th layer).
- **896 experts**, 16 active per token.
- **Linear-attention state: 434 MB per session — and it never grows**, no matter
  how long the conversation gets. That is why long-context offload is cheap.
- **MLA KV cache at 1 million tokens: 15.74 GB** with our fp8 format
  (27.65 GB without it).
- So one sleeping 1M-token session costs about **16.2 GB**. A hundred of them
  fit in **1.6 TB** of NVMe.
- Waking a 1M-token session means reading 15.7 GB back. On a proper Gen4 NVMe
  drive that is a couple of seconds; recomputing it would take 15+ minutes.

---

## Quick start (2 × RTX 3090)

```bash
# build a tiny synthetic slice of K3 — no checkpoint download needed
python tools/gen_slice_hf.py

VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /workspace/k3/k3-slice-hf \
  --trust-remote-code --load-format dummy \
  --pipeline-parallel-size 2 --tensor-parallel-size 1 \
  --enable-prefix-caching --block-size 512 \
  --max-model-len 32768 --enforce-eager
```

On sm_86 you should see `Using TRITON_MLA` and
`Using FLASH_ATTN MLA prefill backend`, and no FlashKDA line. That is correct —
those are the Ampere fallbacks.

To add KV offload to disk:

```bash
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec",
      "cpu_bytes_to_use":2147483648,
      "secondary_tiers":[{"type":"fs","root_dir":"/nvme/kv"}]}}'
```

---

## Traps we hit, so you do not have to

- **Always pass `--block-size`** with a hybrid model plus offload. Without it you
  get an error telling you to enable prefix caching — even when prefix caching
  *is* enabled. The real cause is that pipeline stages disagree about block size.
  At 50 stages this affects **26 of them**. Reported as
  [vllm#51752](https://github.com/vllm-project/vllm/issues/51752).
- **vLLM 0.27.0 crashes** on the first request with a hybrid model plus prefix
  caching (`index_fill_(): Expected dtype int64`). Fixed on `main`, not in the
  release. Patch is one word: `idx_mapping.long()`.
  ([vllm#50947](https://github.com/vllm-project/vllm/issues/50947))
- **`pip install vllm --pre` gives you the release, not `main`.** Install from
  `https://wheels.vllm.ai/nightly/cu130` and then actually check
  `vllm.__version__` — a wrong wheel URL fails silently.
- **`--kv-cache-dtype fp8` is rejected below SM89.** Our `fp8_ds_mla` is a
  different dtype and is carved out of that check by patch `P3c`.
- **Do not build a slice from the upstream K3 config as-is.** It is multimodal
  now, so vLLM ≥0.27.0 tries to load a vision tower and gets OOM-killed.
  `tools/gen_slice_hf.py` writes a flat, text-only config instead.
- **Never pass `--no-async-scheduling` with pipeline parallelism.** Only the
  async scheduler arms the cadence that keeps a request's decodes `pp_size`
  steps apart; without it speculative decoding silently corrupts output (token
  id 0 embedded as the anchor), and even plain decode ran **1.94× slower** in
  our measurements. ([vllm#52071](https://github.com/vllm-project/vllm/issues/52071))
- **`--enforce-eager` is for debugging, not for serving.** We carried it from a
  debugging session into a week of benchmarks; on GLM-4.5-Air it hid a 3.41×
  difference. (On K3 graphs currently buy ~1% — see above — but measure, don't
  inherit flags.)

---

## How to read this repo

**Start with [`JOURNAL.md`](JOURNAL.md).** It is the full record: what was
proven, what was disproven (including verdicts we published and later had to
take back), what every experiment actually measured, and what is still unknown.

| Path | What is in it |
|---|---|
| [`JOURNAL.md`](JOURNAL.md) | The complete story. Read this first. |
| `docs/` | Deep dives: the REAP route, the DSpark blocker, capacity math, the Ampere gap analysis |
| `results/` | One file per experiment, with the raw numbers |
| `fp8kv_k3_port/` | Our fp8 KV cache implementation, with tests |
| `tools/` | Slice generator, patch scripts, checkpoint converter |

---

## Honest limitations

- The most recent numbers come from a **real-weight 24-expert slice (141 GB,
  all 93 layers) on 8×3090** — real architecture, real draft, real MXFP4. Its
  *acceptance rate* still is not K3's (the draft was trained against the full
  448-expert target), and its degenerate text makes it unusable for any test
  that depends on exact token equality.
- **No full-scale run has ever happened.** Multi-node behaviour, throughput with
  100 concurrent agents, and quality after pruning are all unknown.
- The speedup numbers in the offload results (1.2–1.9×) are an artifact of the
  tiny test model, which is very cheap to prefill. Do not quote them as the real
  benefit.

## Upstream

Bugs found here and reported to vLLM:
[#52069](https://github.com/vllm-project/vllm/issues/52069) + PR
[#52117](https://github.com/vllm-project/vllm/pull/52117) (SupportsPP demanded
of draft models under PP),
[#52071](https://github.com/vllm-project/vllm/issues/52071) (spec + PP corrupts
output without async scheduling; fix branch shared),
[#51752](https://github.com/vllm-project/vllm/issues/51752),
[#50947](https://github.com/vllm-project/vllm/issues/50947),
[#50098](https://github.com/vllm-project/vllm/issues/50098).
Details and current status in
[`docs/UPSTREAM_FILING_RECORD.md`](docs/UPSTREAM_FILING_RECORD.md).
