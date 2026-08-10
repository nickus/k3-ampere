# fp8-KV port to Kimi-K3 dense TritonMLA (sm_86) — results

Goal: read the MLA KV cache as packed fp8_ds_mla (656 B/token) instead of bf16
(1152 B/token) on RTX 3090, so 10×1M-context agents fit ~50 cards instead of
~60. Query stays bf16; fp8 is storage-only with in-kernel software dequant.

## Design (validated by the Aug-2026 research)

The port is **one decode kernel + 5 glue lines**. Everything else already
exists in vLLM main and is arch-portable:
- cache **writer** (`fused_kimi_k3_mla_*_ds_mla_insert`) — ungated in CMake,
  PDL blocks are no-ops below sm_90;
- chunked-prefill **context** dequant (`cp_gather_and_upconvert_fp8_kv_cache`)
  — ungated, `_compute_prefill_context` already dtype-dispatches on
  `fp8_ds_mla`;
- **pool sizing** — `MLAAttentionSpec.real_page_size_bytes` already returns
  656 for K3, and K3's `get_kv_cache_spec` already passes `kv_quant_mode`
  (the GLM-port engine-bug #2 is already fixed upstream for K3);
- the Blackwell-only asserts we feared are on the **plain-fp8** branch;
  fp8_ds_mla's contract (fp8 cache + bf16 query) is what TritonMLA already
  declares (`supports_quant_query_input=False`).

The only hole: TritonMLA decode reads a contiguous 576-wide row, not a packed
656-byte one, and `fp8_ds_mla` isn't in its dtype list.

## What's built & GPU-verified (this package)

`fp8kv_k3/` — overlay package (no fork):
- `dequant.py` — fp8e4m3fn→fp16 bit-math, transplanted from the GLM port.
- `layout.py` — 656 B row pack/unpack, pinned to the CUDA writer.
- `kernel.py` — grouped split-KV decode over packed rows; stage-1 mirrors
  vLLM's `_fwd_grouped_kernel_stage1` control flow exactly (same paging walk,
  int64 page offset, online softmax, MLA `v=trans(k)` reuse); stage-2 vendored
  verbatim.
- `apply_vllm_patches.py` — P3a dtype list, P3b 656 cache shape, P3c SM89-gate
  carve-out for fp8_ds_mla only, P5 forward_mqa dispatch.

**On a real RTX 3090 (12/12 tests):**
- fp8 dequant bit-exact on all 254 finite values; both NaN encodings decode to
  a valid NaN (torch's own cast is device-inconsistent here — 0x7F80 CPU vs
  0x7FFF CUDA — so exact NaN payload is not contractual; NaN never occurs in
  real KV).
- 656 B layout round-trips (cosine > 0.999 NoPE, RoPE bit-exact) with a
  negative test pinning the scale byte offsets.
- **decode kernel vs fp32 naive reference: cosine 0.9999967** (bf16 query),
  split-KV invariant across {1,2,4} splits (the GLM scratch-kernel merge-bug
  guard).

## Perf note (microbench, not the headline)

Single-request decode, 96 heads: BLOCK_N=16/stages=2/warps=4 wins; effective
14–17 GB/s over the 656 B rows. This is latency-bound tiny work — the port's
value is **capacity** (656 vs 1152 = 1.75× context per card / fewer cards per
replica), consistent with the ~10–15 % decode cost measured on the GLM sparse
path, not decode throughput. Real serving amortizes launch across a
continuous batch; a bf16-vs-fp8 A/B on the e2e slice is the honest speed
number (pending).

## GPU gate C — e2e boot: PASSED (2026-08-10)

Booted the K3 slice with `--kv-cache-dtype fp8_ds_mla` on a real 3090
(wheel rc5, all patches applied). Confirmed from the engine log:
- `Using fp8_ds_mla data type to store kv cache` — the dtype threaded through
  cache config;
- `Using TRITON_MLA attention backend` + `Using FLASH_ATTN MLA prefill
  backend` — our P3a/b/c gates let fp8_ds_mla reach TritonMLA on sm_86;
- `Application startup complete`.
- **Generation runs** on both short and long (400-token) prompts — the full
  mixed loop executes: fp8 cache insert (CUDA writer) → chunked-prefill
  dequant (`cp_gather_and_upconvert_fp8_kv_cache`) → our decode kernel via
  `forward_mqa`. (Text is gibberish by design — `--load-format dummy`.)

The 5 integration patches (`apply_vllm_patches.py`) are proven end-to-end,
not just in isolation.

**Final numeric confirmation: greedy parity EXACT.** Same deterministic
slice served twice — `--kv-cache-dtype auto` (bf16) vs `fp8_ds_mla` — same
greedy prompt, 32 tokens: **token-for-token identical output**, equal TPOT
(8 ms at slice scale). The fp8 storage path introduces no decision-level
divergence on this workload.

Boot-path notes for the runbook: top-level `architectures` must be forced
to `KimiLinearForCausalLM` (text-only) or the loader pulls an image
processor; the fp8kv_k3 module must be importable by the worker (drop into
site-packages — an editable install needs a pyproject). rc5 wheel kept
with a zip-integrity check (the rc4 copy was a silently-truncated scp).
