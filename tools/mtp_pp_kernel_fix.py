#!/usr/bin/env python3
"""Stop the MTP draft-prefill kernel writing before the start of its buffer.

MEASURED, GLM-4.5-Air + method=mtp + PP=4 on 8x3090, first request, with
CUDA_LAUNCH_BLOCKING=1 so the report lands on the kernel that actually faulted:

    File ".../v1/worker/gpu/spec_decode/autoregressive/speculator.py", line 752,
      in prepare_prefill_inputs
        _prepare_prefill_inputs_kernel[(num_reqs,)](
    RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered

A probe placed immediately before `propose()` shows the state that reaches it:

    nreqs=1 idx_mapping=[3] num_sampled=[1] num_rejected=[3]
    qlen=[3] eff_qlen=[0] seq_lens=[12]

Three tokens scheduled, three drafts rejected, so

    query_len -= num_rejected            ->  0
    last_token_index = query_start + query_len - 1     ->  query_start - 1
    tl.store(draft_input_ids_ptr + last_token_index, next_token)

writes one element BEFORE the buffer. Triton does not wrap negative indices the
way torch does - it adds them to the base pointer - and the store carries no
mask, so this is an out-of-bounds write, not a wrong value.

WHY PIPELINE PARALLELISM IS REQUIRED TO HIT IT. Without PP the scheduler knows
the sampled token before it commits the next step, so a spec step always carries
k drafts PLUS the accepted token: query_len = k+1, at most k can be rejected,
and the effective length never falls below 1. Under PP the next step is
committed before the sampled token comes back over the deferred relay, so a step
can be scheduled containing ONLY draft tokens - qlen 3 for k=3 above. Reject all
of them and the assumed floor of 1 is gone.

Verified this is upstream and not ours: the same fault, at the same kernel, with
the same probe values, reproduces with our draft-token relay disabled
(`relay_drafts=False`).

THE FIX. When every draft is rejected, rejection sampling still emits the
corrected token, and that token is exactly what the next draft round must read.
So the floor is one token, not zero. Clamping restores the invariant the kernel
already assumes everywhere below this line.

This is a guard, not a redesign: it does not change any case that was already
correct, since query_len < 1 could only ever have produced an out-of-bounds
write.
"""

import sys
from pathlib import Path


def main() -> int:
    import vllm

    target = (
        Path(vllm.__file__).parent
        / "v1" / "worker" / "gpu" / "spec_decode" / "autoregressive" / "speculator.py"
    )
    src = target.read_text()

    old = """    # Get the true query length and next token after accounting for rejected tokens.
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    query_len -= num_rejected
"""
    new = """    # Get the true query length and next token after accounting for rejected tokens.
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    query_len -= num_rejected
    # Under pipeline parallelism the scheduler commits the next step before the
    # sampled token returns over the deferred relay, so a step can be scheduled
    # containing ONLY draft tokens. Rejecting all of them leaves query_len == 0,
    # and the unmasked store below then writes to draft_input_ids[query_start-1].
    # Rejection sampling still emits the corrected token, and that token is what
    # the next draft round reads, so the floor is one - never zero.
    if query_len < 1:
        query_len = 1
"""

    if new in src:
        print("mtp pp kernel guard: already applied")
        return 0
    if old not in src:
        print("ANCHOR MISSING - upstream moved; refusing to guess", file=sys.stderr)
        return 1

    target.write_text(src.replace(old, new, 1))
    print(f"mtp pp kernel guard: applied in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
