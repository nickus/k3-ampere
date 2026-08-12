#!/usr/bin/env python3
"""Compare speculative *verification* across PP degrees, step by step.

Greedy output is identical at PP=1 with and without speculation, and identical
across PP degrees without speculation — but differs at every PP>=2 *with*
speculation, first at generated token #22 of 24. Acceptance counters say PP=1
accepts nothing (1.00) while PP>=2 accepts something (1.05), so the PP path
takes a draft the reference rejects.

This prints, per `sample()` call on the last rank:
  ndraft   - input_batch.num_draft_tokens
  target   - argmax of the target logits, one per verified position
  draft    - argmax of the draft logits the rejection sampler is given
  sampled  - sampler_output.num_sampled
  rejected - sampler_output.num_rejected

Diff the traces from a PP=1 run and a PP=2 run: the first line that differs says
whether the draft proposed something different (a drafting problem) or the same
draft was judged differently (a verification problem).
"""

import sys
from pathlib import Path

TAG = "[VERIFY]"


def main() -> int:
    import vllm

    target = Path(vllm.__file__).parent / "v1" / "worker" / "gpu" / "model_runner.py"
    src = target.read_text()

    if TAG in src:
        print("verify probe already installed")
        return 0

    anchor = (
        "        return sampler_output, sampler_output.num_sampled, "
        "sampler_output.num_rejected"
    )
    if anchor not in src:
        print("ANCHOR MISSING - upstream moved; refusing to guess", file=sys.stderr)
        return 1

    probe = '''        try:
            import os as _os

            _dl = getattr(self.speculator, "draft_logits", None) if self.speculator else None
            _d = (
                _dl.argmax(dim=-1).flatten().tolist()[:8]
                if _dl is not None and _dl.numel()
                else []
            )
            print(
                f"[VERIFY] pid={_os.getpid()} ndraft={input_batch.num_draft_tokens} "
                f"target={logits.argmax(dim=-1).flatten().tolist()[:8]} draft={_d} "
                f"sampled={sampler_output.num_sampled.flatten().tolist()[:8]} "
                f"rejected={sampler_output.num_rejected.flatten().tolist()[:8]}",
                flush=True,
            )
        except Exception as _e:  # instrumentation must never break the server
            print("[VERIFY] probe error:", _e, flush=True)
'''

    target.write_text(src.replace(anchor, probe + anchor, 1))
    print(f"verify probe installed in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
