# Negative matrix — observed on 2×3090, vllm 0.26.1rc2+sm86 (2026-07-30)

| Test | Expected | Observed |
|---|---|---|
| `--kv-cache-dtype fp8` | ValueError TritonMLAImpl | ✅ `ValueError: FP8 KV cache is not supported by the Triton MLA backend on NVIDIA GeForce RTX 3090 (compute capability 8.6); native FP8 (fp8e4nv) requires SM89+` |
| `additional_config {"kda_prefill_backend":"flashkda"}` | RuntimeError | ✅ RuntimeError at EngineCore init |
| eagle3 + PP=2 | V2 ValueError | pending |
| MTP + PP=2 without VLLM_USE_V2_MODEL_RUNNER | rank-0 AttributeError drafter | pending |
