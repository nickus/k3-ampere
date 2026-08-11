# SPDX-License-Identifier: Apache-2.0
"""TEST-ONLY: skip kernel warmup.

Warmup pre-JITs kernels so the first real request is not slow. It is an
optimisation, not a correctness requirement. Under PP + speculative decoding the
last rank hangs inside warmup's sampling step (before the drafter is even
invoked), which blocks the engine from ever serving.

Skipping it answers the question that actually matters — does the end-to-end
speculative path work under pipeline parallelism — while leaving the warmup bug
to be fixed separately. Expect a slower first request.

DO NOT propose upstream.

    python tools/skip_warmup_hack.py [site-packages-path]
"""

import sys


def main(sp: str) -> None:
    p = f"{sp}/vllm/v1/worker/gpu/warmup.py"
    s = open(p).read()
    anchor = """    if model_runner.is_encoder_only:
        return"""
    new = """    if model_runner.is_encoder_only:
        return

    import os as _os

    if _os.environ.get("SKIP_KERNEL_WARMUP") == "1":
        print("[WM] kernel warmup SKIPPED by SKIP_KERNEL_WARMUP=1", flush=True)
        return"""
    if "SKIP_KERNEL_WARMUP" in s:
        print("skip-warmup hack: already applied")
        return
    assert anchor in s, "anchor missing"
    open(p, "w").write(s.replace(anchor, new, 1))
    import ast

    ast.parse(open(p).read())
    print("skip-warmup hack: applied (set SKIP_KERNEL_WARMUP=1)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/venv/nm/lib/python3.12/site-packages")
