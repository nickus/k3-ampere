# W3-g64 (vellum geometry) serves on sm_86 via Humming — VALIDATED (2026-08-10)

Claim proven on 1×3090 (wheel rc4): a compressed-tensors
**int3 / group 64** K3-slice checkpoint — the exact geometry of
`vellum-ai/Kimi-K3-W3A16-g64` — loads and generates in vLLM:
`Using 'HUMMING' WNA16 MoE backend` → startup complete → generation.

**One extra step vs W2 is REQUIRED: a weight repack.** compressed-tensors
packs 3-bit word-wise (pack_factor 10, 2 dead bits/word → 52 int32 per
K=512 row); Humming expects bit-continuous (K·bits/32 = 48 words). Layouts
differ, so raw vellum shards will NOT load — this is exactly why the GSQ
toolchain ships `convert_to_humming.py`. Our standalone converter:
`tools/convert_and_verify.py` — unpacks CT (its own layout rules), repacks
via `humming.ops.pack_weight` (canonical for the kernel), and **numerically
proves itself**: dequant-reference GEMM vs Humming forward, signed→unsigned
offset +4, **cosine = 1.000000**; then repacks all expert tensors.

## Three NEW upstream findings (all hit only at num_bits ∉ {4,8})

1. **`get_weight_shape` floor-divides by pack_factor** (`hidden_size //
   self.packed_factor`, 6 sites): for 3-bit gives 51 words vs checkpoint's
   ceil-packed 52 → loader shape mismatch. Correct universal formula is
   bit-exact `K * num_bits // 32` (identical for 4/8-bit, which is why
   tests never caught it). → PR #48918 feedback.
2. **`QuantKey.__str__` crashes on custom ScalarTypes**
   (`fx.graph.dtype_abbrs[self.dtype]` KeyError on uint3b4) — the PR adds
   `_dtype_abbr` helper; until merged any 3-bit error path dies on
   *printing* the error. Fixed with `.get(dtype, str(dtype))`.
3. **CT word-wise vs Humming bit-continuous packing mismatch** (above) —
   needs either a load-time repack in vLLM or an offline converter;
   documenting the +4 offset and layout pair saves the next person a day.

Also re-learned (process): after reinstalling a wheel, the PR *diff* must
be re-applied too, not just our hand patches — `_supports_quant_scheme`
silently returning False for everything was the tell.

## Infra notes for the runbook

- rented GPU instances can be evicted after prolonged idle time, and storage on a
  stopped instance is not durable; artifacts pulled same-day now (rc4 wheel in
  `/home/dev/wheels/`).
- vastai re-published the `vllm:v0.25.1-cuda-13.0` image tag with a
  different torch ABI build → rc2 wheel became unloadable (undefined
  symbol in FA2). Wheel rebuilt as rc4 on the new box (192 cores, ~25 min).
  Do not trust image tags for ABI stability; pin by digest or rebuild.
- Host egress was flaky (TLS resets to github/HF): cutlass shipped from
  the dev box via `VLLM_CUTLASS_SRC_DIR`; HF file fetches wrapped in
  retries.

## Meaning

- The **only known way to serve vellum's W3 artifact on any shipped GPU
  stack** now exists (patched vLLM + repack). Posted as follow-up in their
  discussion #1.
- For OUR fleet the W3 capacity math is unchanged (1 layer/card → ~93
  cards); W2 remains the target. But the whole low-bit path (scheme gates,
  factories, kernel, converter) is now exercised at 2 AND 3 bits — Track B
  de-risked further.
