#!/usr/bin/env python3
"""Print, per rank and per call, which branch of `sample_tokens` was taken.

The PP>=3 decode deadlock has receivers and no sender for the sampled-token
broadcast. There are exactly two places the last rank can leave `sample_tokens`
without broadcasting:

  1. `if self.execute_model_state is None: return None`  — before the rank
     branch, so it is invisible to the other ranks;
  2. `PPHandler.broadcast`'s own `compute_need_sampled_mask(...) is None` guard.

Both are printed here, with the rank, so the two sides of the collective can be
compared instead of assumed to agree.
"""

import sys
from pathlib import Path

TAG = "[STPROBE]"


def patch(path: Path, old: str, new: str, tag: str) -> bool:
    src = path.read_text()
    if TAG in src and tag in src:
        print(f"  {tag}: already present")
        return True
    if old not in src:
        print(f"  {tag}: ANCHOR MISSING", file=sys.stderr)
        return False
    path.write_text(src.replace(old, new, 1))
    print(f"  {tag}: applied")
    return True


def main() -> int:
    import vllm

    sp = Path(vllm.__file__).parent
    ok = True

    # 1. the pre-branch early return
    ok &= patch(
        sp / "v1" / "worker" / "gpu" / "model_runner.py",
        """        if self.execute_model_state is None:""",
        f'''        import os as _os
        from vllm.distributed.parallel_state import get_pp_group as _gpp

        print(
            f"{TAG} pid={{_os.getpid()}} rank={{_gpp().rank_in_group}} "
            f"last={{_gpp().is_last_rank}} enter_sample_tokens "
            f"state_is_none={{self.execute_model_state is None}}",
            flush=True,
        )
        if self.execute_model_state is None:''',
        "S1 sample_tokens entry",
    )

    # 2. the broadcast-side guard
    ok &= patch(
        sp / "v1" / "worker" / "gpu" / "pp_utils.py",
        """        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:""",
        f'''        assert self.is_last_rank
        import os as _os

        _m = compute_need_sampled_mask(input_batch)
        print(
            f"{TAG} pid={{_os.getpid()}} LAST rank broadcast guard: "
            f"{{'SKIP (mask None)' if _m is None else 'SEND'}}",
            flush=True,
        )
        if _m is None:''',
        "S2 broadcast guard",
    )

    # 3. the receive-side guard
    ok &= patch(
        sp / "v1" / "worker" / "gpu" / "pp_utils.py",
        """        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:""",
        f'''        need_sampled_mask = compute_need_sampled_mask(input_batch)
        import os as _os

        print(
            f"{TAG} pid={{_os.getpid()}} non-last receive guard: "
            f"{{'SKIP (mask None)' if need_sampled_mask is None else 'RECV'}}",
            flush=True,
        )
        if need_sampled_mask is None:''',
        "S3 receive guard",
    )

    print("SAMPLE_TOKENS_PROBE_DONE" if ok else "SAMPLE_TOKENS_PROBE_INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
