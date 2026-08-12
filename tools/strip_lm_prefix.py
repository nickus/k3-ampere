#!/usr/bin/env python3
"""Rename `language_model.*` -> `*` in place, by rewriting headers only.

Measured: serving the slice as KimiLinearForCausalLM fails with

    ValueError: There is no module or parameter named 'language_model' in
    KimiLinearForCausalLM

The prefix belongs to the multimodal wrapper, which owns the text model as a
`language_model` submodule. The text-only class is the top level itself, so its
parameters are `model.layers.N...` and `lm_head.weight` with no prefix. Since we
deliberately dropped the vision tower, the prefix has to go too.

Rewriting 141 GB to rename keys would be absurd, and it is unnecessary. A
safetensors file is:

    [8-byte little-endian header length N][N bytes of JSON header][tensor data]

Every tensor's `data_offsets` are relative to the end of the header, so as long
as N does not change, the data does not move. The header is JSON and the format
pads it with spaces for alignment, so a SHORTER header can be padded back to
exactly N. Removing a 15-character prefix only ever shrinks it. So this rewrites
the first 8+N bytes of each shard and touches nothing else - seconds, not hours.

Refuses to write if the new header does not fit, rather than corrupting a shard.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

PREFIX = "language_model."


def fix_shard(path: Path) -> tuple[int, int]:
    with open(path, "r+b") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
        hdr = json.loads(raw)

        renamed = {}
        n_renamed = 0
        for k, v in hdr.items():
            if k.startswith(PREFIX):
                renamed[k[len(PREFIX):]] = v
                n_renamed += 1
            else:
                renamed[k] = v
        if n_renamed == 0:
            return 0, len(hdr)

        new = json.dumps(renamed, separators=(",", ":")).encode()
        if len(new) > n:
            raise RuntimeError(
                f"{path.name}: new header {len(new)} > original {n}; refusing"
            )
        # Pad with spaces: json.loads and safetensors' own parser both tolerate
        # trailing whitespace, and the format already uses space padding.
        f.seek(8)
        f.write(new + b" " * (n - len(new)))
    return n_renamed, len(hdr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/models/k3-slice")
    args = ap.parse_args()

    d = Path(args.dir)
    shards = sorted(d.glob("*.safetensors"))
    if not shards:
        print(f"no shards in {d}", file=sys.stderr)
        return 1

    total = 0
    for s in shards:
        r, _ = fix_shard(s)
        total += r
    print(f"renamed {total} tensors across {len(shards)} shards")

    idx_path = d / "model.safetensors.index.json"
    idx = json.loads(idx_path.read_text())
    idx["weight_map"] = {
        (k[len(PREFIX):] if k.startswith(PREFIX) else k): v
        for k, v in idx["weight_map"].items()
    }
    idx_path.write_text(json.dumps(idx))
    print(f"index rewritten: {len(idx['weight_map'])} entries")

    # Prove the files still parse, and show what the names became.
    from safetensors import safe_open

    with safe_open(str(shards[0]), "pt") as g:
        keys = list(g.keys())
        print("sample names:", keys[:3])
        assert not any(k.startswith(PREFIX) for k in keys), "prefix survived"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
