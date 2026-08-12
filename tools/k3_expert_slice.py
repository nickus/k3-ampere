#!/usr/bin/env python3
"""Build a runnable Kimi-K3 out of a subset of its experts, by ranged download.

WHY. Acceptance rate is meaningless against a random draft, and the real K3 is
837 GB (REAP-448) - it fits no 3090 box. But the *non-expert* weights are only
113.8 GB of that, and one expert costs 1.61 GB across all 92 MoE layers
(17.55 MB per layer). So a K3 with a handful of real experts, real attention,
real embeddings and the real tokenizer fits 8x3090:

    weights = 113.8 GB + N * 1.61 GB      (N = experts kept)
    N=24  ->  152 GB, leaves ~20 GB for KV at util 0.90

This is NOT a quality artifact and its acceptance rate is NOT K3's: the DSpark
draft was trained against the full 448-expert target, so a 24-expert target
shifts the distribution the draft is predicting. It is a *mechanism* stand -
real architecture, real draft, non-zero acceptance - which is strictly better
than the 4-layer dummy-weight slice for catching plumbing regressions.

Keep N > num_experts_per_token (16), or routing degenerates: top-16 of 16 selects
everything and the MoE becomes a dense sum.

Downloads only the tensors it keeps, via HTTP range requests against the
safetensors shards, so it moves ~152 GB rather than 837.
"""

import argparse
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

REPO = "runrunway/Kimi-K3-REAP-448experts"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"

# safetensors dtype -> (numpy dtype string, bytes per element)
DT = {
    "F64": ("float64", 8), "F32": ("float32", 4), "F16": ("float16", 2),
    "BF16": ("bfloat16", 2), "I64": ("int64", 8), "I32": ("int32", 4),
    "I16": ("int16", 2), "I8": ("int8", 1), "U8": ("uint8", 1),
    "BOOL": ("bool", 1), "F8_E4M3": ("float8_e4m3fn", 1),
    "F8_E5M2": ("float8_e5m2", 1),
}


def get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:  # transient 5xx / reset on a 96-shard sweep
            if attempt == 4:
                raise
            print(f"  retry {attempt + 1} after {type(e).__name__}", flush=True)
    raise RuntimeError("unreachable")


def read_header(shard: str) -> tuple[dict, int]:
    n = struct.unpack("<Q", get(BASE + shard, 0, 7))[0]
    hdr = json.loads(get(BASE + shard, 8, 8 + n - 1))
    return hdr, 8 + n


def keep_tensor(name: str, n_keep: int) -> bool:
    """Text-only, and only the first n_keep experts of each MoE layer."""
    if not name.startswith("language_model."):
        return False  # vision_tower / mm_projector: we serve text
    if ".experts." in name:
        try:
            eid = int(name.split(".experts.")[1].split(".")[0])
        except (IndexError, ValueError):
            return False
        return eid < n_keep
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=24)
    ap.add_argument("--out", default="/workspace/models/k3-slice")
    ap.add_argument("--shards", default="", help="e.g. 1-96, for resuming")
    args = ap.parse_args()

    import numpy as np
    import torch
    from safetensors.torch import save_file

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"index for {REPO}", flush=True)
    idx = json.loads(get(BASE + "model.safetensors.index.json"))
    shards = sorted({v for v in idx["weight_map"].values()})
    if args.shards:
        lo, hi = (int(x) for x in args.shards.split("-"))
        shards = [s for s in shards if lo <= int(s.split("-")[1]) <= hi]
    print(f"{len(shards)} shards, keeping experts 0..{args.experts - 1}", flush=True)

    new_map: dict[str, str] = {}
    total_bytes = 0
    for si, shard in enumerate(shards, 1):
        dst = out / shard
        if dst.exists():
            hdr, _ = read_header(shard)
            for name in hdr:
                if name != "__metadata__" and keep_tensor(name, args.experts):
                    new_map[name] = shard
            print(f"[{si}/{len(shards)}] {shard}: present, skipped", flush=True)
            continue

        hdr, data_start = read_header(shard)
        wanted = [
            (n, i) for n, i in hdr.items()
            if n != "__metadata__" and keep_tensor(n, args.experts)
        ]
        if not wanted:
            print(f"[{si}/{len(shards)}] {shard}: nothing to keep", flush=True)
            continue

        wanted.sort(key=lambda kv: kv[1]["data_offsets"][0])
        tensors: dict[str, torch.Tensor] = {}
        # Merge neighbouring ranges: kept experts are contiguous, so this turns
        # thousands of small requests into a handful of large ones.
        spans: list[list[int]] = []
        for _, info in wanted:
            s, e = info["data_offsets"]
            if spans and s - spans[-1][1] <= 4 << 20:
                spans[-1][1] = e
            else:
                spans.append([s, e])
        blob: dict[tuple[int, int], bytes] = {}
        for s, e in spans:
            blob[(s, e)] = get(BASE + shard, data_start + s, data_start + e - 1)
            total_bytes += e - s

        for name, info in wanted:
            s, e = info["data_offsets"]
            for (bs, be), buf in blob.items():
                if bs <= s and e <= be:
                    raw = buf[s - bs:e - bs]
                    break
            else:
                raise RuntimeError(f"range miss for {name}")
            npdt, _ = DT[info["dtype"]]
            if npdt == "bfloat16":
                t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
            elif npdt.startswith("float8"):
                t = torch.frombuffer(bytearray(raw), dtype=getattr(torch, npdt))
            else:
                t = torch.from_numpy(
                    np.frombuffer(bytearray(raw), dtype=np.dtype(npdt)).copy()
                )
            t = t.reshape(info["shape"])
            # The router indexes experts by row; leaving 448 rows would point at
            # experts that no longer exist.
            if name.endswith("block_sparse_moe.gate.weight"):
                t = t[: args.experts].clone()
            elif name.endswith("gate.e_score_correction_bias"):
                t = t[: args.experts].clone()
            tensors[name] = t
            new_map[name] = shard

        save_file(tensors, str(dst), metadata={"format": "pt"})
        del tensors, blob
        print(
            f"[{si}/{len(shards)}] {shard}: {len(wanted)} tensors, "
            f"{total_bytes / 1e9:.1f} GB so far",
            flush=True,
        )

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": new_map})
    )

    cfg = json.loads(get(BASE + "config.json"))
    text = dict(cfg["text_config"])
    text["num_experts"] = args.experts
    # Flat, text-only: the upstream config is multimodal and vLLM will otherwise
    # try to build a vision tower whose weights we deliberately did not fetch.
    text["architectures"] = ["KimiLinearForCausalLM"]
    text["model_type"] = cfg.get("model_type", "kimi_k3")
    for k in ("bos_token_id", "eos_token_id", "pad_token_id", "dtype",
              "tie_word_embeddings"):
        if k in cfg:
            text.setdefault(k, cfg[k])
    (out / "config.json").write_text(json.dumps(text, indent=1))

    # K3 ships no tokenizer.json: the tokenizer is tiktoken, loaded by
    # tokenization_kimi.py under trust_remote_code. Asking for tokenizer.json
    # gets a 404, and a slice without tiktoken.model cannot be served at all.
    for extra in ("configuration_kimi_k3.py", "encoding_k3.py",
                  "generation_config.json", "tokenizer_config.json",
                  "tokenization_kimi.py", "tiktoken.model",
                  "modeling_kimi_k3.py", "modeling_kimi_linear.py"):
        try:
            (out / extra).write_bytes(get(BASE + extra))
        except Exception as e:
            print(f"  optional {extra}: {type(e).__name__}", flush=True)

    print(f"DONE: {total_bytes / 1e9:.1f} GB into {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
