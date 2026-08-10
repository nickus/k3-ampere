"""Rebuild the HF-loadable K3 slice on box2 (lost with the old volume).
Reuses the shape-verified generator logic from k3-ampere; emits model.* names
(no language_model. prefix) + per-head A_log [num_heads] as HF modeling wants."""
import json
import urllib.request

import torch
from safetensors.torch import save_file

OUT = "/workspace/k3/k3-slice-hf"
import os

os.makedirs(OUT, exist_ok=True)

# --- tensors (same formulas as k3-ampere/tools/gen_gguf_slice.py, slice cfg) ---
C = dict(h=1024, H=8, V=163840, E=8, lat=512, mi=256, n_sh=2,
         ql=1536, kvl=512, di=2048, n_layers=4, dense0=True, kda={0, 1, 2})
d, rope, fr = 128, 64, 128
g = torch.Generator().manual_seed(42)
T = {}


def t(name, shape, dtype=torch.bfloat16, ones=False):
    x = torch.ones(shape) if ones else torch.randn(shape, generator=g) * 0.02
    T[name] = x.to(dtype).contiguous()


h, H, E, lat, mi = C["h"], C["H"], C["E"], C["lat"], C["mi"]
t("model.embed_tokens.weight", [C["V"], h])
t("lm_head.weight", [C["V"], h])
t("model.norm.weight", [h], ones=True)
t("model.output_attn_res_norm.weight", [h], ones=True)
t("model.output_attn_res_proj.weight", [1, h])
for L in range(C["n_layers"]):
    p = f"model.layers.{L}."
    t(p + "input_layernorm.weight", [h], ones=True)
    t(p + "post_attention_layernorm.weight", [h], ones=True)
    for r in ("self_attention_res", "mlp_res"):
        t(p + r + "_norm.weight", [h], ones=True)
        t(p + r + "_proj.weight", [1, h])
    a = p + "self_attn."
    if L in C["kda"]:
        T[a + "A_log"] = (torch.rand(H).abs() + 0.5).float()  # per-head for HF
        t(a + "o_norm.weight", [d], dtype=torch.float32, ones=True)
        t(a + "dt_bias", [H * d], dtype=torch.float32)
        for q in ("q", "k", "v"):
            t(a + q + "_proj.weight", [H * d, h])
            T[a + q + "_conv1d.weight"] = (torch.randn(H * d, 1, 4, generator=g) * 0.01).float()
        t(a + "b_proj.weight", [H, h])
        t(a + "f_a_proj.weight", [fr, h])
        t(a + "f_b_proj.weight", [H * d, fr])
        t(a + "g_proj.weight", [H * d, h])
        t(a + "o_proj.weight", [h, H * d])
    else:
        t(a + "q_a_proj.weight", [C["ql"], h])
        t(a + "q_a_layernorm.weight", [C["ql"]], ones=True)
        t(a + "q_b_proj.weight", [H * (d + rope), C["ql"]])
        t(a + "kv_a_proj_with_mqa.weight", [C["kvl"] + rope, h])
        t(a + "kv_a_layernorm.weight", [C["kvl"]], ones=True)
        t(a + "kv_b_proj.weight", [H * (d + d), C["kvl"]])
        t(a + "g_proj.weight", [H * d, h])
        t(a + "o_proj.weight", [h, H * d])
    if L == 0 and C["dense0"]:
        m = p + "mlp."
        t(m + "gate_proj.weight", [C["di"], h])
        t(m + "up_proj.weight", [C["di"], h])
        t(m + "down_proj.weight", [h, C["di"]])
    else:
        m = p + "block_sparse_moe."
        t(m + "gate.weight", [E, h])
        t(m + "gate.e_score_correction_bias", [E], dtype=torch.float32)
        t(m + "routed_expert_up_proj.weight", [h, lat])
        t(m + "routed_expert_down_proj.weight", [lat, h])
        t(m + "routed_expert_norm.weight", [lat], ones=True)
        sh = C["n_sh"] * mi
        t(m + "shared_experts.gate_proj.weight", [sh, h])
        t(m + "shared_experts.up_proj.weight", [sh, h])
        t(m + "shared_experts.down_proj.weight", [h, sh])
        for e in range(E):
            t(m + f"experts.{e}.w1.weight", [mi, lat])
            t(m + f"experts.{e}.w3.weight", [mi, lat])
            t(m + f"experts.{e}.w2.weight", [lat, mi])

save_file(T, f"{OUT}/model.safetensors", metadata={"format": "pt"})
print("tensors:", len(T))

# --- config + tokenizer + remote code from the real repo ---
BASE = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main/"
for f in ("config.json", "configuration_kimi_k3.py", "modeling_kimi_linear.py",
          "tiktoken.model", "tokenization_kimi.py", "encoding_k3.py",
          "tokenizer_config.json", "generation_config.json"):
    urllib.request.urlretrieve(BASE + f, f"{OUT}/{'k3_real_config.json' if f == 'config.json' else f}")
c = json.load(open(f"{OUT}/k3_real_config.json"))
tc = c["text_config"]
tc.update({"hidden_size": 1024, "intermediate_size": 2048, "num_hidden_layers": 4,
           "num_attention_heads": 8, "num_key_value_heads": 8, "num_experts": 8,
           "num_experts_per_token": 2, "routed_expert_hidden_size": 512,
           "moe_intermediate_size": 256, "first_k_dense_replace": 1,
           "max_position_embeddings": 32768, "num_nextn_predict_layers": 0,
           "attn_res_block_size": 4})
la = tc["linear_attn_config"]
la["kda_layers"] = [1, 2, 3]
la["full_attn_layers"] = [4]
la["num_heads"] = 8
tc.pop("quantization_config", None)
c["auto_map"] = {"AutoConfig": "configuration_kimi_k3.KimiLinearConfig",
                 "AutoModelForCausalLM": "modeling_kimi_linear.KimiLinearForCausalLM"}
json.dump(c, open(f"{OUT}/config.json", "w"), indent=1)
print("HF slice ready at", OUT)
