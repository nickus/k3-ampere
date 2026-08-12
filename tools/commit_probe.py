#!/usr/bin/env python3
"""Log every commit of sampled tokens, per rank, to catch the duplicated tail.

The tail defect is size k-1 and anchored to the end of generation, and the
visible symptom is a *repeat*: the reference alternates two tokens, PP=2 emits
the previous one again. A repeat means a step's tokens reach `all_token_ids`
twice, or a stale FIFO slot is applied a second time.

`postprocess_sampled` is the single funnel for that: the last rank calls it
directly, non-last ranks call it from `PPHandler.get_prev_sampled_outputs`
pp_size steps later. Printing (rank, num_sampled, token ids, whether
query_start_loc was supplied) on every call shows a double-commit immediately.
"""

import sys
from pathlib import Path

TAG = "[COMMIT]"


def main() -> int:
    import vllm

    target = Path(vllm.__file__).parent / "v1" / "worker" / "gpu" / "model_runner.py"
    src = target.read_text()

    if TAG in src:
        print("commit probe already installed")
        return 0

    anchor = """        # Update the number of computed tokens.
        if self.is_last_pp_rank:"""
    if anchor not in src:
        print("ANCHOR MISSING - upstream moved; refusing to guess", file=sys.stderr)
        return 1

    probe = '''        try:
            import os as _os

            _n = num_sampled.flatten().tolist()[:4]
            _t = sampled_tokens.flatten().tolist()[:8]
            print(
                f"[COMMIT] pid={_os.getpid()} last={self.is_last_pp_rank} "
                f"qsl={query_start_loc is not None} num_sampled={_n} "
                f"num_rejected={num_rejected.flatten().tolist()[:4]} tokens={_t}",
                flush=True,
            )
        except Exception as _e:
            print("[COMMIT] probe error:", _e, flush=True)
'''

    target.write_text(src.replace(anchor, probe + anchor, 1))
    print(f"commit probe installed in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
