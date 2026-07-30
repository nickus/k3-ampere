# sm_86 gap analysis — vLLM @ 38a267cdd (2026-07-30)

Method: 8 subsystem readers + adversarial verification of every hard-blocker
claim (call-path traced from `vllm serve`), synthesized 2026-07-30. Hardware
confirmations on 2×3090 noted inline.

## Fallback map (what runs on sm_86, with zero code changes)

| Subsystem | Blackwell path | sm_86 fallback | Gate |
|---|---|---|---|
| KDA prefill | FlashKDA (CUTLASS, 9.0a/10.0f/12.0f only) | vendored Triton `chunk_kda` | `is_flashkda_supported()` → False (capability major ∈ {9,10,12}); `flashkda` absent from sm_86 builds by cmake arch intersection. **HW-verified: 33 tests pass** |
| KDA decode | `fused_kda_decode` CUDA op (SM90/10x/12x + head whitelist) | Triton `fused_recurrent_kda` (+ separate conv update + norm) | `is_fused_kda_decode_supported()` → False via `hasattr` + capability |
| MoE MXFP4 | DeepGEMM "mega moe" (SM100), TRTLLM, CuTe DSL tail | **Marlin W4A16 MXFP4** (same as gpt-oss-on-Ampere) | oracle `fused_moe/oracle/mxfp4.py`. **HW-verified: selection tests pass.** ⚠️ Marlin is the ONLY candidate for K3 on sm_86 — no TRITON/EMULATION tail; if `MarlinExperts.is_supported_config` rejects, boot dies (5-LOC insurance: whitelist SITU in `TritonExperts._supports_activation`) |
| MLA decode | FlashMLA / TokenSpeed (SM90+/SM100) | TRITON_MLA (bf16 KV only) | backend priority list, cuda.py major==8 branch |
| MLA prefill | SM100 paths | **FA2 varlen** (V padded 128→192) — the ONLY candidate below SM100; a build without vllm-flash-attn FA2 = "No valid MLA prefill backend found" at init | `selector.py` priority |
| MLA cache insert | same op, PDL enabled | same CUDA op, PDL runtime-gated off | builds for all arches — but **absent from wheels older than ~Jul 29** (bindings added post-dev78): source build or shim required |
| Router (GateLinear) | fp32-out fused | Tier-6 bf16 `F.linear` | numerics: validate top-k agreement vs fp32 reference |
| Attn residuals (AttnRes) | CUDA kernel sm90+ | Triton `attn_res.py` (GB300-tuned configs) | capability gate in model.py |
| SiTU activation | `situ_and_mul` CUDA op | CustomOp `forward_native` (or the op itself — it's generic CUDA, builds for 8.6) | needs post-Jul-29 build |

## Hard blockers (adversarially confirmed, no fallback)

1. **fp8 KV cache — all variants.** `TritonMLAImpl.__init__` raises below
   SM89 (`triton_mla.py:206`); even on SM89+, K3's fused fp8 decode asserts
   `supports_quant_query_input` (False for TritonMLA) and fp8 prefill-query
   quant is gated to capability family 100. So plain fp8 KV for K3 is
   effectively **Blackwell-datacenter-only** upstream. sm_86 = bf16 KV,
   1152 B/token/MLA-layer. (Port sketch — the only real memory lever:
   teach TritonMLA in-kernel fp8 dequant with bf16 query + a non-quant-query
   branch in `kimi_k3/nvidia/mla.py`; ~400–800 LOC. Direct continuation of
   our fp8-KV RFC #48374 work.)
2. **Capacity on 50 cards** — see `CAPACITY_50x3090.md`. Not a kernel gap.
3. **Spec decode under PP:** eagle3 raises in V2 and crashes V1
   (`drafter` only on last rank); DSpark raises at V2 init and its V1
   escape hatch is non-functional. **MTP is the only PP-compatible method**
   and requires `VLLM_USE_V2_MODEL_RUNNER=1` (same mechanics we confirmed on
   GLM-5.2). Caveat: base K3 ships **no MTP head** (`num_nextn_predict_layers=0`)
   — production spec decode on a PP rig needs a trained MTP head from
   elsewhere; slice validates mechanics only.

## Negative test matrix (each must fail exactly so)

| Action | Expected failure |
|---|---|
| `--kv-cache-dtype fp8` | ValueError, TritonMLAImpl init |
| `--kv-cache-dtype fp8_ds_mla` | "No valid attention backend found" |
| `additional_config {"kda_prefill_backend":"flashkda"}` | RuntimeError kda.py |
| spec `{"method":"eagle3"}` + PP | V2 ValueError |
| MTP + PP without V2 flag | rank-0 AttributeError 'drafter' |

## Perf backlog (after boot, in expected-value order)

1. Retune GB300-tuned Triton configs for sm_86 99KB smem:
   `fused_recurrent.py` BV/num_stages buckets, `attn_res.py` block_l/warps.
2. Port `fused_kda_decode` CUDA kernel to 8.6 (~30 LOC cmake+gates; kernel
   is cp.async-based, Ampere-capable) — collapses 3 decode launches → 1
   per KDA layer per step.
3. FLA_USE_CUDA_GRAPH=1 + persist Triton autotune cache (70 KDA layers ×
   first-run autotune is minutes of warmup).
4. fp8 KV port (above) — 2× MLA KV capacity.
5. Accept FA2 V-padding overhead in MLA prefill (inherent to FA2).
