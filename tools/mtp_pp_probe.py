#!/usr/bin/env python3
"""Print what `propose()` is handed on the last PP rank, just before it faults.

`_prepare_prefill_inputs_kernel` does raw pointer arithmetic with no bounds
masking on two quantities that PP can make unusual:

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    ... tl.load(last_sampled_ptr + req_state_idx)          # -1 reads BEFORE the buffer
    query_len -= num_rejected
    last_token_index = query_start + query_len - 1          # can go negative
    tl.store(draft_input_ids_ptr + last_token_index, ...)   # ...and write out of bounds

Triton does not wrap negative indices the way torch does; it just adds them to
the base pointer. So either a -1 in idx_mapping or a num_rejected larger than
query_len is enough for the illegal access we see. This prints both, plus the
shapes, so we stop guessing which.

Idempotent; asserts its anchor.
"""

import sys
from pathlib import Path


def main() -> int:
    import vllm

    target = Path(vllm.__file__).parent / "v1" / "worker" / "gpu" / "model_runner.py"
    src = target.read_text()

    # `--remove` matters before benchmarking: the probe calls .tolist() on GPU
    # tensors, which synchronises every step and would be measured as latency.
    if "--remove" in sys.argv:
        if "[MTPPROBE]" not in src:
            print("mtp probe not installed")
            return 0
        start = src.index("            try:\n                import os as _os")
        end = src.index('print("[MTPPROBE] probe error:", repr(_e), flush=True)\n')
        end = src.index("\n", end) + 1
        target.write_text(src[:start] + src[end:])
        print("mtp probe removed")
        return 0

    if "[MTPPROBE]" in src:
        print("mtp probe already installed")
        return 0

    anchor = "            draft_tokens = self.speculator.propose(\n"
    if anchor not in src:
        print("ANCHOR MISSING - upstream moved; refusing to guess", file=sys.stderr)
        return 1

    probe = '''            try:
                import os as _os

                _im = input_batch.idx_mapping
                _qsl = input_batch.query_start_loc
                _nr = num_rejected
                _ns = num_sampled
                _ls = self.req_states.last_sampled_tokens
                _qlen = (_qsl[1:] - _qsl[:-1])[: input_batch.num_reqs]
                _eff = _qlen - _nr[: input_batch.num_reqs]
                print(
                    f"[MTPPROBE] pid={_os.getpid()} nreqs={input_batch.num_reqs} "
                    f"idx_mapping={_im.tolist()[:8]} min={int(_im.min())} "
                    f"num_sampled={_ns.tolist()[:8]} "
                    f"num_rejected={_nr.tolist()[:8]} "
                    f"qlen={_qlen.tolist()[:8]} eff_qlen={_eff.tolist()[:8]} "
                    f"eff_min={int(_eff.min())} "
                    f"last_sampled.shape={tuple(_ls.shape)} "
                    f"seq_lens={input_batch.seq_lens.tolist()[:8]}",
                    flush=True,
                )
            except Exception as _e:
                print("[MTPPROBE] probe error:", repr(_e), flush=True)
'''

    target.write_text(src.replace(anchor, probe + anchor, 1))
    print(f"mtp probe installed in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
