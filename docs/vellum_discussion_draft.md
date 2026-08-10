# POSTED 2026-08-10 as https://huggingface.co/vellum-ai/Kimi-K3-W3A16-g64/discussions/1
# Draft: HF discussion for vellum-ai/Kimi-K3-W3A16-g64

Post at: https://huggingface.co/vellum-ai/Kimi-K3-W3A16-g64/discussions/new
(or I post it after `hf auth login` on the dev box)

---

**Title:** Toolchain details? And any plans for a W2A16 sibling?

**Body:**

Thanks for publishing this — as far as we can tell it's the first sub-4-bit
compressed-tensors K3 artifact, and your README's framing of the kernel gap
is exactly right.

Three questions / one heads-up:

1. **Toolchain:** what produced this — llmcompressor GPTQ, a custom
   pipeline? In particular, how did you feed the mxfp4-pack-quantized
   source experts (on-the-fly per-layer dequant, or a bf16 staging pass),
   and roughly what hardware/wall-clock did the full 2.8T run take? We're
   scoping a similar run and any producer-side landmines you hit would be
   gold (we documented ours from slice-scale runs in
   https://github.com/nickus/k3-ampere — e.g. ignore-lists needing
   vLLM-side *fused* module names like `gate_up_proj`/`in_proj_qkvgfab`,
   not just HF names).

2. **W2A16 plans?** Our target fleet is 24 GB Ampere cards (RTX 3090),
   where the capacity math is unforgiving: W3-g64 ≈ 13.7 GB/layer → one
   layer per card → ~93 cards for full K3; W2-g128 ≈ 883 GB total → two
   layers per card → ~47 cards. If a W2 sibling is on your roadmap, we'd
   rather test yours than produce our own.

3. **Heads-up on the kernel gap:** vLLM PR #48918 (CT-WNA16 MoE via the
   Humming JIT library, bits 2–8, SM75+) plus a handful of small patches
   already loads and serves CT pack-quantized MoE on Ampere — we validated
   the W2 path end-to-end on a synthetic K3 slice (quantize →
   `Using 'HUMMING' WNA16 MoE backend` → generation) on RTX 3090s, patches
   and writeups in the repo above. So your artifact may be servable
   *without* the W3→W4 recode sooner than the README suggests; we're about
   to run the same slice validation at num_bits=3 and will report back.

---
(after posting: link the discussion in README "Status" tracking?)
