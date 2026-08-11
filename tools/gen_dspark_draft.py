"""Generate a miniature K3 DSpark draft config matched to our 4-layer slice.

The published draft (`lightseekorg/kimi-k3-dspark`) expects `hidden_size 7168`
and taps at target layers 7/31/47/63/87 — none of which exist in the slice, so
it cannot be used to test the PP patch. This emits the same `K3DSparkModel`
architecture scaled to the slice, so `--load-format dummy` can fill the weights
and the whole spec-decode path still gets exercised.

MLA dimensions are deliberately left at production values (q_lora 1536,
kv_lora 512, nope 128, rope 64) — the same choice `make_slice_config.py` makes,
because those are the layout-critical ones for the kernels under test.

Usage:
    python tools/gen_dspark_draft.py [--out DIR] [--slice-config PATH]
"""

import argparse
import json
import os

# Mirrors the real draft's schema; only the sizes change.
DRAFT_LAYERS = 5


def build(slice_cfg: dict, draft_layers: int = DRAFT_LAYERS) -> dict:
    h = slice_cfg["hidden_size"]
    n_target = slice_cfg["num_hidden_layers"]

    # Tap every target layer we have. The real checkpoint spreads 5 taps over 93
    # layers; with 4 layers the honest analogue is "all of them", which also
    # maximises the number of PP boundaries a tap has to survive — exactly what
    # the patch under test is for.
    # vLLM adds 1 when converting these to aux-layer ids (DFlash semantics),
    # so 0-based here lands on 1..n and stays inside the model. Using 1..n
    # produced (2..n+1) and the out-of-range last id silently yielded one tap
    # fewer than context_proj was sized for:
    #   RuntimeError: mat1 and mat2 shapes cannot be multiplied (2048x3072 and 4096x1024)
    target_layer_ids = list(range(0, n_target))

    return {
        "architectures": ["K3DSparkModel"],
        "model_type": "k3_dspark",
        "hidden_size": h,
        "intermediate_size": slice_cfg.get("intermediate_size", 2 * h),
        "num_hidden_layers": draft_layers,
        "num_attention_heads": slice_cfg["num_attention_heads"],
        "num_key_value_heads": slice_cfg["num_attention_heads"],
        # MLA geometry. The per-head dims are layout-critical and stay at
        # production values; the LoRA ranks are NOT — they must stay below the
        # model width or the kernels get a shape combination that does not
        # occur in any real checkpoint. Copying q_lora_rank=1536 verbatim onto
        # a 1024-wide slice (the real draft is 1536 against 7168) produced
        # `Triton Error [CUDA]: an illegal memory access was encountered`.
        "q_lora_rank": max(256, min(1536, h // 2)),
        "kv_lora_rank": max(128, min(512, h // 4)),
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_nope": False,
        "mla_use_output_gate": False,
        "vocab_size": slice_cfg["vocab_size"],
        "draft_vocab_size": slice_cfg["vocab_size"],
        "rms_norm_eps": slice_cfg.get("rms_norm_eps", 1e-5),
        "max_position_embeddings": slice_cfg["max_position_embeddings"],
        "rope_theta": slice_cfg.get("rope_theta", 50000.0),
        # the taps: this is what the PP patch has to deliver
        "num_target_layers": len(target_layer_ids),
        "target_hidden_size": h,
        "target_num_hidden_layers": n_target,
        "target_layer_ids": target_layer_ids,
        "fc_norm": True,
        # Required: the speculator refuses to start without one of
        # mask_token_id / dspark_noise_token_id / pard_token / ptd_token_id.
        # The real draft uses 163837 against the same 163840-token vocabulary.
        "mask_token_id": slice_cfg["vocab_size"] - 3,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "sample_from_anchor": True,
        "bos_token_id": slice_cfg.get("bos_token_id"),
        "eos_token_id": slice_cfg.get("eos_token_id"),
        "pad_token_id": slice_cfg.get("pad_token_id"),
        "dtype": slice_cfg.get("dtype", "bfloat16"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-config", default="/workspace/k3/k3-slice-hf/config.json")
    ap.add_argument("--out", default="/workspace/k3/k3-dspark-draft")
    ap.add_argument("--draft-layers", type=int, default=DRAFT_LAYERS)
    args = ap.parse_args()

    slice_cfg = json.load(open(args.slice_config))
    # gen_slice_hf.py emits a flat text-only config; tolerate a nested one too.
    slice_cfg = slice_cfg.get("text_config", slice_cfg)

    cfg = build(slice_cfg, args.draft_layers)
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/config.json", "w") as f:
        json.dump(cfg, f, indent=1)

    print(f"draft config -> {args.out}/config.json")
    print(
        f"  hidden {cfg['hidden_size']} | draft layers {cfg['num_hidden_layers']} | "
        f"taps {cfg['target_layer_ids']} over {cfg['target_num_hidden_layers']} "
        f"target layers"
    )


if __name__ == "__main__":
    main()
