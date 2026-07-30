"""Synthetic K3 slice in real checkpoint layout (nested config, language_model.* names).
Shape formulas are verified against real shard headers before emitting."""
import json, torch
from safetensors.torch import save_file

R = json.load(open("/workspace/k3/real_shapes.json"))
P = "language_model."

def formulas(c):
    h, H, d = c["h"], c["H"], 128
    E, lat, mi, shi = c["E"], c["lat"], c["mi"], c["n_sh"]*c["mi"]
    ql, kvl, rope = c["ql"], c["kvl"], 64
    fr = 128  # forget-gate low rank
    F = {}
    F["model.embed_tokens.weight"] = ("BF16", [c["V"], h])
    F["lm_head.weight"] = ("BF16", [c["V"], h])
    F["model.norm.weight"] = ("BF16", [h])
    F["model.output_attn_res_norm.weight"] = ("BF16", [h])
    F["model.output_attn_res_proj.weight"] = ("BF16", [1, h])
    for L in range(c["n_layers"]):
        p = f"model.layers.{L}."
        F[p+"input_layernorm.weight"] = ("BF16", [h])
        F[p+"post_attention_layernorm.weight"] = ("BF16", [h])
        for r in ("self_attention_res", "mlp_res"):
            F[p+r+"_norm.weight"] = ("BF16", [h])
            F[p+r+"_proj.weight"] = ("BF16", [1, h])
        if L in c["kda"]:
            a = p+"self_attn."
            F[a+"A_log"] = ("F32", [d]); F[a+"o_norm.weight"] = ("F32", [d])
            F[a+"dt_bias"] = ("F32", [H*d])
            for t in ("q","k","v"):
                F[a+t+"_proj.weight"] = ("BF16", [H*d, h])
                F[a+t+"_conv1d.weight"] = ("F32", [H*d, 1, 4])
            F[a+"b_proj.weight"] = ("BF16", [H, h])
            F[a+"f_a_proj.weight"] = ("BF16", [fr, h])
            F[a+"f_b_proj.weight"] = ("BF16", [H*d, fr])
            F[a+"g_proj.weight"] = ("BF16", [H*d, h])
            F[a+"o_proj.weight"] = ("BF16", [h, H*d])
        else:
            a = p+"self_attn."
            F[a+"q_a_proj.weight"] = ("BF16", [ql, h])
            F[a+"q_a_layernorm.weight"] = ("BF16", [ql])
            F[a+"q_b_proj.weight"] = ("BF16", [H*(d+rope), ql])
            F[a+"kv_a_proj_with_mqa.weight"] = ("BF16", [kvl+rope, h])
            F[a+"kv_a_layernorm.weight"] = ("BF16", [kvl])
            F[a+"kv_b_proj.weight"] = ("BF16", [H*(d+d), kvl])
            F[a+"g_proj.weight"] = ("BF16", [H*d, h])
            F[a+"o_proj.weight"] = ("BF16", [h, H*d])
        if L == 0 and c["dense0"]:
            m = p+"mlp."
            F[m+"gate_proj.weight"] = ("BF16", [c["di"], h])
            F[m+"up_proj.weight"] = ("BF16", [c["di"], h])
            F[m+"down_proj.weight"] = ("BF16", [h, c["di"]])
        else:
            m = p+"block_sparse_moe."
            F[m+"gate.weight"] = ("BF16", [E, h])
            F[m+"gate.e_score_correction_bias"] = ("F32", [E])
            F[m+"routed_expert_up_proj.weight"] = ("BF16", [h, lat])
            F[m+"routed_expert_down_proj.weight"] = ("BF16", [lat, h])
            F[m+"routed_expert_norm.weight"] = ("BF16", [lat])
            F[m+"shared_experts.gate_proj.weight"] = ("BF16", [shi, h])
            F[m+"shared_experts.up_proj.weight"] = ("BF16", [shi, h])
            F[m+"shared_experts.down_proj.weight"] = ("BF16", [h, shi])
            for e in range(E):
                F[m+f"experts.{e}.w1.weight"] = ("BF16", [mi, lat])
                F[m+f"experts.{e}.w3.weight"] = ("BF16", [mi, lat])
                F[m+f"experts.{e}.w2.weight"] = ("BF16", [lat, mi])
    return F

REAL = dict(h=7168, H=96, V=163840, E=896, lat=3584, mi=3072, n_sh=2,
            ql=1536, kvl=512, di=33792, n_layers=93, dense0=True,
            kda=set(i for i in range(93) if (i+1) % 4 != 0 and i+1 != 93))
FR = formulas(REAL)
bad = 0
for name, (dt, shape) in R.items():
    key = name[len(P):] if name.startswith(P) else name
    if key not in FR: continue
    fdt, fsh = FR[key]
    if fdt != dt or fsh != shape:
        print("MISMATCH", key, "real", (dt, shape), "formula", (fdt, fsh)); bad += 1
print("verified against real:", len([k for k in R if (k[len(P):] if k.startswith(P) else k) in FR]), "tensors; mismatches:", bad)
assert bad == 0

SLICE = dict(h=1024, H=8, V=163840, E=8, lat=512, mi=256, n_sh=2,
             ql=1536, kvl=512, di=2048, n_layers=4, dense0=True, kda={0,1,2})
FS = formulas(SLICE)
g = torch.Generator().manual_seed(42)
tensors = {}
for key, (dt, shape) in FS.items():
    dtype = torch.bfloat16 if dt == "BF16" else torch.float32
    t = torch.randn(shape, generator=g, dtype=torch.float32) * 0.02
    if "A_log" in key: t = t.abs() + 0.5
    if "conv1d" in key: t = t * 0.5
    if "layernorm" in key or "norm.weight" in key: t = torch.ones(shape)
    tensors[P+key] = t.to(dtype).contiguous()
save_file(tensors, "/workspace/k3/gguf-slice/model.safetensors",
          metadata={"format": "pt"})
print("emitted", len(tensors), "tensors")
