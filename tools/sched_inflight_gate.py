#!/usr/bin/env python3
"""Attach drafts only to a request whose previous step has been reported.

MEASURED (16-layer stand, PP=2, DSpark k=3, one request):

    [ANCHOR] last_sampled=[16925]  drafts=[16925, 16925, 16925]   healthy
    [ANCHOR] last_sampled=[0]      drafts=[0, 0, 0]               6 of 26 steps

The zeros land at num_computed_tokens = 13, 17, 21, 25, 29 - every 4 tokens,
which is one spec step at k=3. They are upstream, not ours: with our draft relay
disabled they get MORE frequent, not fewer. And they are not a
pipeline-starvation artefact: running four concurrent requests leaves the rate
unchanged at ~25%.

WHY. The scheduler advances optimistically when it schedules:

    request.num_computed_tokens  += num_scheduled_token     # anchor + k drafts
    request.num_in_flight_tokens += num_scheduled_token

and only corrects when the step is reported:

    request.num_computed_tokens  -= num_rejected
    request.num_in_flight_tokens -= num_tokens_scheduled

Under pipeline parallelism the report is pp_size steps late, so in between the
request is booked as up to k tokens further along than it really is. Planning the
next step against that inflated count is what drops the anchor's SLOT (fixed
separately) and what makes the runner read an anchor VALUE the relay has not
delivered yet - token id 0, which is `!`, which is what broke greedy parity.

THE GATE. `num_in_flight_tokens` is exactly "scheduled but not yet reported", so
requiring it to be zero before attaching drafts means the previous step's sampled
token has landed and the anchor is real. vLLM already uses the same quantity to
recover a confirmed position elsewhere:

    max(0, request.num_computed_tokens - request.num_in_flight_tokens)

Cost: with one request at PP=2 speculation engages every other step instead of
every step. It does not disable speculation - a request simply waits for its own
previous step, which is the pipelining constraint that was being violated. With
several requests in flight the stages stay busy regardless.

Idempotent; asserts its anchor.
"""

import sys
from pathlib import Path


def main() -> int:
    import vllm

    target = Path(vllm.__file__).parent / "v1" / "core" / "sched" / "scheduler.py"
    src = target.read_text()

    old = """            # Speculative decode related.
            if request.spec_token_ids:"""
    new = """            # Speculative decode related.
            # Only speculate for a request whose previous step has been
            # reported. Under PP the report is pp_size steps late, and until it
            # lands num_computed_tokens is booked ahead of reality, so the step
            # would be planned against a position the runner cannot supply an
            # anchor token for - it reads last_sampled_tokens before the relay
            # writes it and embeds token 0. num_in_flight_tokens is exactly
            # "scheduled but not yet reported".
            if request.spec_token_ids and request.num_in_flight_tokens:
                # Drop this round's drafts rather than speculate blind; they are
                # regenerated for the next step by update_draft_token_ids.
                request.spec_token_ids = []
            if request.spec_token_ids:"""

    if new in src:
        print("scheduler in-flight gate: already applied")
        return 0
    if old not in src:
        print("ANCHOR MISSING - upstream moved; refusing to guess", file=sys.stderr)
        return 1

    target.write_text(src.replace(old, new, 1))
    print(f"scheduler in-flight gate: applied in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
