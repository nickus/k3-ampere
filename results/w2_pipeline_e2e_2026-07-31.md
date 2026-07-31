# W2 producer→consumer pipeline: END-TO-END GREEN on sm_86 (2026-07-31)

Step 1 of the GSQ plan (`docs/GSQ_W2_PLAN.md`): the full pipeline
**quantizer → CT checkpoint → vLLM+Humming serving** validated on the
synthetic K3 slice, 1×3090.

## The chain that now works

1. `llmcompressor` data-free W2 (`QuantizationModifier`, int, group 128,
   symmetric, experts-only ignore) over the HF-loadable K3 slice
   → CT `pack-quantized` checkpoint (`weight_packed` I32 + `weight_scale`
   BF16 + `weight_shape`), byte-for-byte the schema PR #48918's tests
   assert.
2. `vllm serve` (wheel rc3 + patches): **`Using 'HUMMING' WNA16 MoE
   backend`** + `HummingLinearKernel for CompressedTensorsWNA16` +
   startup complete + generation.

## Landmines found & fixed (feed these into GSQ Track B and upstream)

1. **CT ignore-lists must include vLLM-side FUSED module names.** HF names
   (`q_proj`, `gate_proj`…) match, but vLLM's fused modules
   (`gate_up_proj`, `fused_qkv_a_proj`, `in_proj_qkvgfab`, `conv1d`) fail
   `should_ignore_layer` even when every constituent HF name is ignored
   (probe: `q_proj→True, in_proj_qkvgfab→False`). Symptom: fused module
   built quantized → `AttributeError: no attribute 'weight'` (KDA merged)
   or loader `KeyError` (dense/shared/MLA). Fix: extend checkpoint ignore
   with the fused names. **GSQ's `_build_ignore_list` for K3 must do the
   same; also a candidate vLLM bug report (matcher asymmetry).**
2. **`HummingExpertsBase._supports_activation` lacks SITU** — same bug as
   TritonExperts; the docstring says any `apply_moe_activation` activation
   works. 1-line whitelist → backend selectable for K3. **Feedback for
   PR #48918.**
3. **PR #48918 no longer applies cleanly to main@38a267c** — rejects in
   `compressed_tensors_moe_wna16_marlin.py` (subsumed by `is_transposed`
   on our tree) and `oracle/int_wna16.py` (the QuantizationArgs schema
   branch — hand-applied). **Feedback for the PR author.**
4. **llmcompressor drags torch → 2.12**, breaking the ABI of anything
   compiled against 2.11 (cost us a wheel rebuild: rc3). Pin torch when
   installing, or use a separate venv for the quantizer.
5. transformers 5.10 vs K3 remote code: `OutputRecorder` moved
   (shim from `transformers.utils.output_capturing`), `_tied_weights_keys`
   list→dict, `A_log` is per-head `[num_heads]` on the HF side (the real
   checkpoint's `[128]` is head_dim-shaped — converter tolerated it,
   HF modeling does not; keep both layouts in the generator).

## What this de-risks

The "nobody has ever quantized 2.8T with this toolchain" unknown is now
narrowed to: (a) GSQ-fork work items (K3 wrapper, mxfp4 reader, wbits bug),
(b) calibration quality at scale, (c) resources. The FORMAT and the
SERVING side are proven. Track A (full-scale RTN smoke, ~$100) is the
next de-risk; it can reuse today's ignore-list verbatim.
