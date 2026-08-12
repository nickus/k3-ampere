#!/usr/bin/env python3
"""Never schedule a speculative step without its anchor token.

MEASURED (16-layer stand, DSpark draft, k=3):

    PP=1:  24 spec steps, every one   qlen=4  num_logits=4  ndraft=3
    PP=2:  a step arrives with        qlen=3  num_logits=4  ndraft=3
           -> logits_start = query_end - num_logits = 3 - 4 = -1
           -> hidden_states[[-1, 0, 1, 2]] : four indices, three rows

A spec step must contain the previously accepted token followed by the drafts:
logits at a token's position predict the NEXT token, so draft 1 is verifiable
only against the logits of the token before it. The failing step contains the
three drafts and nothing else, so it is unverifiable by construction.

WHY PP CAUSES IT. The scheduler advances optimistically at schedule time:

    request.num_computed_tokens += num_scheduled_token      # anchor + drafts

and rolls back when the output comes home:

    request.num_computed_tokens -= num_rejected

Under pipeline parallelism that output is `pp_size` steps late, so for those
steps the scheduler plans against a `num_computed_tokens` that counts tokens
which were never accepted. Then

    num_new_tokens = num_tokens_with_spec + num_output_placeholders
                     - num_computed_tokens

comes out one short, and the token that gets dropped is the anchor. At PP=1 the
output always returns before the next scheduling pass, which is exactly why the
control is clean.

THE FIX. Keep at least one non-draft token in any step that carries drafts, by
clipping the draft count to `num_new_tokens - 1`. In the healthy case
num_new_tokens is k+1 and the clip is inert (min(k, k) == k). In the broken case
it schedules one fewer draft instead of an unverifiable step - a lost
speculation opportunity, never a wrong or crashing one.

This does not touch the optimistic accounting itself, which is load-bearing for
prefill chunking; it only refuses to emit the malformed step that the accounting
can produce under PP.

Idempotent; asserts its anchor.
"""

import sys
from pathlib import Path


def main() -> int:
    import vllm

    target = Path(vllm.__file__).parent / "v1" / "core" / "sched" / "scheduler.py"
    src = target.read_text()

    old = """                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids"""
    new = """                # Under pipeline parallelism the rejection rollback of
                # num_computed_tokens is pp_size steps late, so num_new_tokens
                # can come out one short and drop the accepted token this step's
                # drafts hang off. Logits predict the NEXT token, so a step made
                # only of drafts cannot verify its first draft at all; it shows
                # up downstream as logits_start = query_end - num_logits < 0.
                # Keep one non-draft token: schedule fewer drafts, never an
                # unverifiable step. Inert when the step is well formed.
                num_scheduled_spec_tokens = min(
                    num_scheduled_spec_tokens, num_new_tokens - 1
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids"""

    if new in src:
        print("scheduler anchor fix: already applied")
        return 0
    if old not in src:
        print("ANCHOR MISSING - upstream moved; refusing to guess", file=sys.stderr)
        return 1

    target.write_text(src.replace(old, new, 1))
    print(f"scheduler anchor fix: applied in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
