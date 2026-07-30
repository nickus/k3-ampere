# SPDX-License-Identifier: Apache-2.0
"""Generate small synthetic Kimi K3 slice configs for 2x RTX 3090 (PP=2).

The slice is text-only (``KimiLinearForCausalLM`` — the exact architecture
named in the real checkpoint's ``text_config``) so it exercises the same
vLLM module path as full K3 minus the vision tower. Weights come from
``--load-format dummy``; only ``config.json`` + the real tokenizer are needed.

Shrunk: layer count, hidden size, head count, expert count/dims, vocab-side
embeddings stay real-sized (real tokenizer reuse).
Kept at production values (layout-critical): MLA head layout
(512/128/64/128 -> head 576 for the fused cache-insert + FA2 dims whitelist),
KDA head_dim=128 + conv=4 (upstream-tested Triton shapes), SiTU betas,
sigmoid+noaux_tc routing, the MXFP4 quant format with its exact ignore list.

Phases: a = bf16 mechanics, b = + compressed-tensors mxfp4-pack-quantized
(Marlin on sm_86), c = + one MTP layer (spec decode under PP, V2 runner).
"""

import argparse
import copy
import json
from pathlib import Path

REAL_CONFIG_PATH = Path(__file__).with_name("k3_real_config.json")

# Everything the slice inherits verbatim from the real text_config.
_INHERIT = [
    "activation_situ_beta",
    "activation_situ_linear_beta",
    "attn_res_block_size",
    "bos_token_id",
    "eos_token_id",
    "first_k_dense_replace",
    "hidden_act",
    "kv_lora_rank",
    "latent_moe_use_norm",
    "mla_use_nope",
    "mla_use_output_gate",
    "moe_layer_freq",
    "moe_renormalize",
    "moe_router_activation_func",
    "norm_topk_prob",
    "num_shared_experts",
    "q_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "rms_norm_eps",
    "rope_theta",
    "routed_scaling_factor",
    "scoring_func",
    "tie_word_embeddings",
    "topk_group",
    "topk_method",
    "v_head_dim",
    "vocab_size",
]

_SLICE = {
    "architectures": ["KimiLinearForCausalLM"],
    "model_type": "kimi_linear",
    "torch_dtype": "bfloat16",
    "hidden_size": 1024,
    "intermediate_size": 2048,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "routed_expert_hidden_size": 512,
    "moe_intermediate_size": 256,
    "max_position_embeddings": 32768,
    "num_nextn_predict_layers": 0,
}


def build_slice_config(phase: str = "a") -> dict:
    if phase not in ("a", "b", "c"):
        raise ValueError(f"unknown phase {phase!r}")
    real = json.loads(REAL_CONFIG_PATH.read_text())["text_config"]

    cfg = dict(_SLICE)
    for key in _INHERIT:
        if key in real and real[key] is not None:
            cfg[key] = real[key]

    la = copy.deepcopy(real["linear_attn_config"])
    la["kda_layers"] = [1, 2, 3]
    la["full_attn_layers"] = [4]
    la["num_heads"] = cfg["num_attention_heads"]
    cfg["linear_attn_config"] = la

    if phase == "b":
        cfg["quantization_config"] = copy.deepcopy(real["quantization_config"])
    if phase == "c":
        cfg["num_nextn_predict_layers"] = 1
    return {k: cfg[k] for k in sorted(cfg)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).parents[1] / "configs")
    ap.add_argument("--phases", default="abc")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for phase in args.phases:
        path = args.out / f"slice_{phase}.json"
        path.write_text(json.dumps(build_slice_config(phase), indent=2) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
