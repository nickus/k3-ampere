#!/usr/bin/env python3
"""Show what the last rank actually selects for the speculative logits.

Evidence so far: under PP the target's argmax is correct at position 0 and the
same wrong token at positions 1..k. That is what several rows reading one
identical bad hidden-state row looks like. This prints, on the rank that
computes logits:

  hs      - shape of the hidden_states tensor it holds (received over the
            pipeline on a non-first rank)
  idx     - input_batch.logits_indices
  rows    - a per-row fingerprint of sample_hidden_states

If rows 1..k share a fingerprint at PP>=2 and differ at PP=1, the selection is
reading the same row repeatedly and the bug is in what arrives, or in the
indices, not in the sampler.
"""

import sys
from pathlib import Path


def main() -> int:
    import vllm

    target = Path(vllm.__file__).parent / "v1" / "worker" / "gpu" / "model_runner.py"
    src = target.read_text()

    if "[HIDDEN]" in src:
        print("hidden-rows probe already installed")
        return 0

    # Anchor on the PAIR of lines: the single line also appears earlier, in the
    # profiling path, and `replace(..., 1)` silently patched that one instead -
    # the probe then fired exactly once, during warmup, and never on a request.
    anchor = (
        "        sample_hidden_states = hidden_states[input_batch.logits_indices]\n"
        "        logits = self.model.compute_logits(sample_hidden_states)"
    )
    if anchor not in src:
        print("ANCHOR MISSING - upstream moved; refusing to guess", file=sys.stderr)
        return 1

    probe = anchor + '''
        try:
            import os as _os

            _r = [
                round(float(sample_hidden_states[_i].float().sum()), 6)
                for _i in range(min(6, sample_hidden_states.shape[0]))
            ]
            print(
                f"[HIDDEN] pid={_os.getpid()} hs={tuple(hidden_states.shape)} "
                f"idx={input_batch.logits_indices.flatten().tolist()[:8]} "
                f"sel={tuple(sample_hidden_states.shape)} rows={_r}",
                flush=True,
            )
        except Exception as _e:
            print("[HIDDEN] probe error:", _e, flush=True)'''

    target.write_text(src.replace(anchor, probe, 1))
    print(f"hidden-rows probe installed in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
