#!/usr/bin/env python3
"""Test the one asymmetry that can produce the PP>=3 decode deadlock.

`PPHandler.receive` (non-last ranks) and `PPHandler.broadcast` (last rank) are
each guarded by `compute_need_sampled_mask(input_batch)`:

    receive:   if need_sampled_mask is None: return False    # skips the recv
    broadcast: if compute_need_sampled_mask(...) is None: return  # skips the send

Both sides must decide identically or the collective has receivers and no
sender. That is exactly the observed state: three ranks blocked waiting for the
sampled-token broadcast while the last rank has moved on to the next
microbatch's execute_model.

This prints, per rank and per call, what the predicate decided and the inputs it
decided from, so the two sides can be compared directly instead of assumed
equal.
"""

import sys
from pathlib import Path

TAG = "[NEEDMASK]"


def main() -> int:
    import vllm

    target = Path(vllm.__file__).parent / "v1" / "worker" / "gpu" / "pp_utils.py"
    src = target.read_text()

    if TAG in src:
        print("need-mask probe already installed")
        return 0

    anchor = "    need_sampled_mask = produces_sample & not_finishing\n    return need_sampled_mask if need_sampled_mask.any() else None"
    if anchor not in src:
        print("ANCHOR MISSING - upstream moved; refusing to guess", file=sys.stderr)
        return 1

    replacement = '''    need_sampled_mask = produces_sample & not_finishing
    import os as _os

    _decided_none = not need_sampled_mask.any()
    print(
        f"{TAG} pid={_os.getpid()} num_reqs={input_batch.num_reqs} "
        f"decided={'NONE_skip_collective' if _decided_none else 'MASK_do_collective'} "
        f"computed={old_computed.tolist()[:4]} sched={input_batch.num_scheduled_tokens.tolist()[:4]} "
        f"prefill_len={prefill_len.tolist()[:4]} max_seq={max_seq_len.tolist()[:4]}",
        flush=True,
    )
    return need_sampled_mask if need_sampled_mask.any() else None'''.replace(
        "{TAG}", TAG
    )

    target.write_text(src.replace(anchor, replacement, 1))
    print(f"need-mask probe installed in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
