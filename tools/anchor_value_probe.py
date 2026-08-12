#!/usr/bin/env python3
"""Print the anchor token VALUE at the moment the step's inputs are built.

Fix #1 gave every spec step an anchor SLOT. Greedy parity still fails and the
output still contains `!`, which is token id 0, while the traces show the drafts
being rejected - so the zero is not an accepted draft, it is the anchor itself.

`combine_sampled_and_draft_tokens` builds the step's input ids from
`req_states.last_sampled_tokens` plus the drafts. Under PP that sampled token
arrives on the deferred relay, so this prints what the value actually is when the
kernel reads it. A zero here is the whole story.

Idempotent; asserts its anchor; supports --remove (it synchronises, so it must
come out before anything is timed).
"""

import sys
from pathlib import Path

TAG = "[ANCHOR]"


def main() -> int:
    import vllm

    target = Path(vllm.__file__).parent / "v1" / "worker" / "gpu" / "model_runner.py"
    src = target.read_text()

    if "--remove" in sys.argv:
        if TAG not in src:
            print("anchor probe not installed")
            return 0
        start = src.index("        try:\n            import os as _os_anchor")
        end = src.index('print("[ANCHOR] probe error:", _e, flush=True)')
        end = src.index("\n", end) + 1
        target.write_text(src[:start] + src[end:])
        print("anchor probe removed")
        return 0

    if TAG in src:
        print("anchor probe already installed")
        return 0

    anchor = "        logits_indices = combine_sampled_and_draft_tokens(\n"
    if anchor not in src:
        print("ANCHOR MISSING - upstream moved; refusing to guess", file=sys.stderr)
        return 1

    probe = '''        try:
            import os as _os_anchor

            _ls = self.req_states.last_sampled_tokens[idx_mapping]
            _dt = self.req_states.draft_tokens[idx_mapping]
            print(
                f"[ANCHOR] pid={_os_anchor.getpid()} "
                f"last_sampled={_ls.flatten().tolist()[:6]} "
                f"drafts={_dt.flatten().tolist()[:6]} "
                f"cu_num_logits={cu_num_logits.tolist()[:6]}",
                flush=True,
            )
        except Exception as _e:
            print("[ANCHOR] probe error:", _e, flush=True)
'''

    target.write_text(src.replace(anchor, probe + anchor, 1))
    print(f"anchor probe installed in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
