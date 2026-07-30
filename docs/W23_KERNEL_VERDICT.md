# Sub-3-bit expert kernel for K3 — scoping verdict (2026-07-30)

Method: 4 research agents (kernel landscape / quality evidence / vLLM
integration surface / alternatives) + synthesis. Full trail in the session
workflow `w23-kernel-scoping`.

## Verdict: do NOT write a custom W2/W3 Marlin-style kernel

1. **The kernel gap has closed upstream.** vLLM main ships
   compressed-tensors **WNA16 for 2–8 bits** plus the **Humming** JIT GEMM
   library (SM75+ → sm_86 qualifies) with fused grouped-MoE modes
   (`HummingIndexedExperts` / `HummingGroupedExperts` / batched), and **GSQ**
   as the quantizer that emits servable CT checkpoints. The remaining W2/W3
   MoE blocker is python plumbing — `num_bits in {4,8}` asserts in
   `CompressedTensorsWNA16MarlinMoEMethod` and `oracle/int_wna16.py` — which
   open **PR #48918** (CT-WNA16 MoE via Humming) addresses. Adopting that is
   ~200–500 LOC of glue; a bespoke kernel is 1.5–3k LOC of CUDA duplicating a
   funded effort (and 3-bit breaks every power-of-2 packing assumption).
2. **Quality, not kernels, is the binding constraint — and it is UNMEASURED.**
   Structural prior: K3's experts are MXFP4-QAT native (**Q4 IS its full
   precision**), so 2.7 bpw halves from native precision with no headroom —
   but the PR author explicitly labels this "pure hypothesizing".
   The oft-cited "K3-Q2 lost to GLM-5.2 Q5" is, verified at source
   (PR #26185 thread, user csabakecskemeti, 2026-07-28/29): **one tester,
   one task** (zero-shot HTML Mario clone), a **homemade Q2_K** (NOT
   unsloth's UD-Q2_K_XL), self-flagged "may be an unfair comparison or an
   issue with my quant". The control is real — full-precision K3 GGUF
   completed the same task "immaculate" — so his Q2 did lose a capability,
   but this is an anecdote, not an A/B. (The "DeepSeek V3.1 −9.4 aider at
   2-bit" datapoint from the research sweep is likewise not verified at
   source by us.) Net: K3@≤2.7bpw quality is **unknown**; the anecdote +
   prior lean negative, which is exactly why gate M0 exists.
3. **Fleet correction:** 50×3090 + 6×4090 + 1×5090 ≈ **1376 GB** — unsloth
   UD-Q2_K_XL (861 GB) **fits today** via llama.cpp. The best-evidenced
   2-bit K3 quant is testable without any kernel work.

## Decision gates (cheap, run before ANY integration work)

- **M0 (2–3 days):** llama.cpp UD-Q2_K_XL A/B vs GLM-5.2 INT4 on Nick's own
  coding-agent evals. If K3-Q2 does not beat GLM-5.2 INT4 — the K3-on-50-cards
  effort ends here (serve GLM-5.2, revisit when a better low-bit K3 exists).
- **M1 (1 day):** Humming W2A16 grouped-MoE microbench on one 3090 — is it
  near bandwidth-bound on sm_86? (No published consumer-Ampere numbers; on
  A800/sm_80 W2 reaches only ~28% of peak.)

## If both gates pass → route (b): Humming + GSQ into vLLM

Track PR #48918; glue CT-WNA16 W2 MoE for K3's LatentMoE (latent 3584,
FFN 3584→3072 — shapes friendly); GSQ-quantize experts only (mirror the
mxfp4 ignore-list). Open risk: GSQ at 2.8T scale is unpriced; Humming MoE
path is weeks-old.

## Route ranking (full reasoning in session log)

1. (f) keep serving GLM-5.2 INT4 — zero-cost incumbent, likely winner
2. (e) llama.cpp UD-Q2 — as the M0 gate + fallback serving path
3. (b) Humming+GSQ W2 in vLLM — only route to real vLLM K3 on ~50 cards
4. (d) buy +43 cards → 93-card native MXFP4 — lossless but $30–45k + PP93 uncharted
5. (a) custom W2/W3 CUDA — only as targeted Humming/sm86 tuning, never a fork
6. (c) ktransformers/CPU-offload — no K3 support, fat-RAM capex, slowest

Caveat: the research agents' claim "K3 sm_86 fallbacks unverified" predates
our own validation (see `results/2x3090_validation_2026-07-30.md`) — the
sm_86 path IS verified; that only strengthens routes (b)/(d).
