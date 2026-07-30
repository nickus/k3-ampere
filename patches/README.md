# sm_86 patches (vs vLLM 38a267cdd), each found by a real boot failure on 2×3090

| # | File | Gap it closes | Upstream-worthy |
|---|---|---|---|
| 01 | `01-triton-experts-situ.patch` | Phase A boot: `NotImplementedError: No Unquantized MoE backend...` — TritonExperts rejects SiTU although the shared `apply_moe_activation` dispatcher already implements it via `torch.ops._C.situ_and_mul` (compiled for all arches). Whitelist SITU in both Triton expert classes; `is_gated`/`adjust_N_for_activation` are generic and need no change. | yes |
| 02 | `02-kimi-linear-mtp-remap.patch` | MTP on a text-only K3 config: the `kimi_k3_mtp` draft remap fires only for `model_type == "kimi_k3"` (the multimodal wrapper); `kimi_linear` with `num_nextn_predict_layers > 0` falls through to `NotImplementedError: Unsupported speculative method`. | yes (guarded by the MTP-layer count) |
| 03 | `03-kimi-k3-mtp-supports-pp.patch` | MTP + PP: the draft ModelConfig is verified against the *target* parallel config (PP>1), so `KimiK3MTPModel` hits "implement the SupportsPP interface" although the drafter runs on the last rank only. Veneer: `supports_pp = True` + a stub `make_empty_intermediate_tensors`. (Arguably the real upstream fix is verifying the draft against the draft parallel config.) | discuss — veneer vs config-check fix |

Applied on the box to `site-packages` (python-only); the wheel itself
(`vllm-0.26.1rc2+sm86`) is stock 38a267cdd built with
`TORCH_CUDA_ARCH_LIST=8.6`.
