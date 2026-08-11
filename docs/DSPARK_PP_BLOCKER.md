# K3's only speculative-decoding lever refuses to run under pipeline parallelism

Found 2026-08-11 by static analysis of vLLM main (`a311916a2`) plus the draft
checkpoint published the same day. **Not yet tested on hardware.**

## Why this matters more than it looks

On the GLM-5.2 campaign, MTP was *the* throughput lever — the measured cost
curve (k=0..3) is what made that serving plan work. **That lever does not
transfer to K3.** The real `moonshotai/Kimi-K3` config says:

```
num_nextn_predict_layers = 0     # no MTP head ships with the model
num_hidden_layers        = 93    # 69 KDA + 24 MLA
num_experts              = 896   # top-16
hidden_size              = 7168
max_position_embeddings  = 1048576
```

So K3's speculative decoding is **DSpark**, and nothing else.

## The draft model now exists

`lightseekorg/kimi-k3-dspark` appeared **2026-08-11** (0 downloads when found):
one 7.12 GB safetensors + config. Per its card and config: a five-layer MLA
backbone, five target hidden-state taps (`target_layer_ids: [7, 31, 47, 63,
87]`), a low-rank Markov head (`markov_rank: 256`) and a confidence head;
proposes **up to 7 draft tokens per step**. Trained with TorchSpec, following
*DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive
Generation* (arXiv 2607.05147).

vLLM main already registers it:

```
"K3DSparkModel": ("vllm.models.kimi_k3.nvidia.dspark_mla", "K3DSparkForCausalLM")
```

## sm_86 outlook: promising, unverified

`vllm/models/kimi_k3/nvidia/dspark_mla.py` (490 lines) imports
`MultiHeadLatentAttention` from `vllm.models.kimi_k3.nvidia.mla` — **the same
MLA class the main model uses**, which we have already run on 3090 via
TRITON_MLA + FA2 prefill. Grepping that file finds **no compute-capability
gates** (no sm90/is_hopper/device_capability checks), unlike FlashKDA. So there
is no *known* new kernel gap. This is a code reading, not a hardware result.

## The blocker

`vllm/v1/worker/gpu/spec_decode/dspark/utils.py:49-50`:

```python
if get_pp_group().world_size != 1:
    raise NotImplementedError("DSpark does not support pipeline parallelism.")
```

A hard raise. Our rig is **PP-only by design** (TP over PCIe is a measured trap
from the GLM-5.2 campaign). So as things stand: **K3 on 50×3090 gets no
speculative decoding at all.**

## Root cause — and why it is fixable

It is not a kernel or hardware limitation. Immediately after that check, the
same function wires the draft to the target by **direct Python object
reference**:

```python
draft_inner.embed_tokens = target_embed
draft_model.lm_head      = target_lm_head
```

(the draft declares `has_own_embed_tokens = False`, `has_own_lm_head = False`,
and its checkpoint deliberately omits both — `checkpoint_skip_substrs =
("confidence_head", "embed_tokens", "lm_head")`.)

Under pipeline parallelism **rank 0 owns `embed_tokens` and the last rank owns
`lm_head`; no single rank holds both**, so the assignment cannot be satisfied
and the guard bails out.

### Fix sketch (ours to build if upstream doesn't)

Host the draft on the **last** PP rank — which already owns `lm_head` — and give
it a private copy of `embed_tokens`:

| Item | Size |
|---|---|
| draft weights | 7.12 GB |
| private `embed_tokens` copy (163840 × 7168, bf16) | 2.35 GB |
| **total on one card** | **≈ 9.5 GB** |

Plus a small KV cache for the draft's 5 MLA layers
(`get_draft_kv_cache_layer_names` returns one name per layer). Roughly one
card's headroom out of fifty, to buy a lever that proposes 7 tokens per step.

Note `precompute_and_store_context_kv(context_states, …)` — the draft *does*
consume target hidden states, just not through `forward` (which takes only
`input_ids`/`positions`). Where those context states come from under PP is the
part of the design that still needs reading before committing to the patch.

## Upstream status

[vllm#50098 — "[Feature]: Kimi K3 DSpark Pipeline Parallelism"](https://github.com/vllm-project/vllm/issues/50098)
is **open**. One comment, from YZYY95K on 2026-07-28: *"I'm trying to solve
this issue, but since I don't have B-series GPUs, I'm submitting the simulated
PP unit test."*

That is a hardware-shaped gap, and hardware-shaped gaps are exactly what this
project has: multi-GPU PP boxes are rentable for cents, and we have already
shipped MTP-under-PP work for GLM-5.2 (overlay of #47629 + #46994, spec-decode
hook ported into `triton_mla_sparse`). Worth offering to test, or to implement.

## Consequence for the rig decision

Until DSpark runs under PP, decode throughput on the rig is **unaccelerated** —
and K3 decodes 104B active parameters with 896-expert top-16 routing that
amortizes poorly across a batch. Any throughput projection for the 100-agent
swarm that assumes speculative decoding is currently unfounded.

**This does not block the KV-offload result** (proven separately, bit-exact),
and it does not block correctness — only speed.
