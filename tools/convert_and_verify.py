"""CT pf10-packed int3 -> Humming bit-continuous repack, with NUMERIC proof.
Uses compressed-tensors own unpack (source of truth for CT layout) and
humming.ops.pack_weight (source of truth for Humming layout)."""
import torch, json, shutil
from safetensors.torch import load_file, save_file
from humming.ops import pack_weight

BITS, GROUP = 3, 64
SRC, DST = "/workspace/k3/k3-slice-w3", "/workspace/k3/k3-slice-w3h"

def ct_unpack(packed, K):
    # compressed-tensors pack: pack_factor = 32 // bits values per int32, LSB-first
    pf = 32 // BITS
    idx = torch.arange(K, device=packed.device)
    word, pos = idx // pf, (idx % pf) * BITS
    vals = (packed[..., word] >> pos[None, :]) & ((1 << BITS) - 1)
    # symmetric signed: stored offset by 2^(bits-1)
    return vals - (1 << (BITS - 1))

T = load_file(f"{SRC}/model.safetensors")
# --- numeric self-test on one expert tensor ---
name = "model.layers.1.block_sparse_moe.experts.0.w1"
pk, sc = T[name + ".weight_packed"], T[name + ".weight_scale"]
N, K = T[name + ".weight_shape"].tolist()
vals = ct_unpack(pk.cuda(), K)                       # [N, K] signed ints
ref = (vals.to(torch.bfloat16).reshape(N, K // GROUP, GROUP)
       * sc.cuda().reshape(N, K // GROUP, 1).to(torch.bfloat16)).reshape(N, K)
x = torch.randn(8, K, dtype=torch.bfloat16, device="cuda") * 0.1
y_ref = x @ ref.T

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
init_distributed_environment(world_size=1, rank=0, local_rank=0,
                             distributed_init_method="tcp://127.0.0.1:29512", backend="nccl")
ctx = set_current_vllm_config(VllmConfig()); ctx.__enter__()
initialize_model_parallel(1, 1)
from vllm.model_executor.layers.quantization.utils.humming_utils import prepare_humming_layer
from humming.layer import HummingMethod

best = None
for offset in (0, 1 << (BITS - 1)):
    codes = (vals + offset).to(torch.int32).contiguous()
    hp = pack_weight(codes, BITS)                    # [N, K*3/32]
    layer = ReplicatedLinear(K, N, bias=False, params_dtype=torch.bfloat16).cuda()
    layer.register_parameter("weight_packed", torch.nn.Parameter(hp, requires_grad=False))
    layer.register_parameter("weight_scale", torch.nn.Parameter(sc.cuda().to(torch.bfloat16), requires_grad=False))
    qc = {"num_bits": BITS, "group_size": GROUP, "strategy": "group", "type": "int",
          "symmetric": True, "format": "pack-quantized", "quant_method": "compressed-tensors"}
    try:
        prepare_humming_layer(layer, qc)
        y = HummingMethod.forward_layer(layer, x)
        cos = torch.nn.functional.cosine_similarity(
            y.float().flatten(), y_ref.float().flatten(), dim=0).item()
        print(f"offset={offset}: cosine={cos:.6f}")
        if best is None or cos > best[1]: best = (offset, cos)
    except Exception as e:
        print(f"offset={offset}: FAIL {type(e).__name__} {str(e)[:120]}")

assert best and best[1] > 0.99, f"no offset passes: {best}"
OFFSET = best[0]
print(f"VERIFIED: offset={OFFSET}, cosine={best[1]:.6f}")

# --- convert every expert tensor ---
shutil.copytree(SRC, DST, dirs_exist_ok=True)
out = dict(T)
n = 0
for k in list(T):
    if k.endswith(".weight_packed") and ".experts." in k:
        base = k[: -len(".weight_packed")]
        N, K = T[base + ".weight_shape"].tolist()
        v = ct_unpack(T[k].cuda(), K) + OFFSET
        out[k] = pack_weight(v.to(torch.int32).contiguous(), BITS).cpu()
        n += 1
save_file(out, f"{DST}/model.safetensors", metadata={"format": "pt"})
print(f"repacked {n} expert tensors -> {DST}")
