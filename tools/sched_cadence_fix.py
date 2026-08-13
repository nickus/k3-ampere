#!/usr/bin/env python3
"""Set the PP decode cadence in the base scheduler too — the fix for vllm#52071.

`next_decode_eligible_step` is read at scheduler.py:509 to keep a request's
decodes `pp_size` steps apart, matching the worker-side sampled-token broadcast
slot ring. It is assigned only in AsyncScheduler (async_scheduler.py:49), so the
base Scheduler — what `--no-async-scheduling` selects — never arms the guard and
reads the ring out of phase: the anchor token arrives as 0 and greedy output is
silently wrong.

The fix mirrors the async assignment into the base scheduler's post-schedule
update, gated exactly like the guard itself: PP > 1 and speculation enabled.
Without speculation the base scheduler is already correct (measured: byte-equal
output PP=1 vs PP=2), so the gate keeps the change from touching working paths.

Applies to the INSTALLED tree for testing; the PR carries the same hunk.
Idempotent; asserts its anchor.
"""

import sys
from pathlib import Path


def main() -> int:
    import vllm

    target = Path(vllm.__file__).parent / "v1" / "core" / "sched" / "scheduler.py"
    src = target.read_text()

    old = """            request.num_computed_tokens += num_scheduled_token
            request.num_in_flight_tokens += num_scheduled_token"""
    new = """            request.num_computed_tokens += num_scheduled_token
            request.num_in_flight_tokens += num_scheduled_token
            if (
                self.vllm_config.speculative_config is not None
                and self.parallel_config.pipeline_parallel_size > 1
            ):
                # Keep this request's decodes `pp_size` steps apart so the
                # worker-side sampled-token broadcast slot ring is read in
                # phase. AsyncScheduler sets this in its own update path;
                # without it the guard at schedule time never fires and a spec
                # step can execute before its anchor token has arrived over the
                # deferred PP relay (embedding token id 0). See #52071.
                request.next_decode_eligible_step = (
                    self.current_step
                    + self.parallel_config.pipeline_parallel_size
                )"""

    if new in src:
        print("cadence fix: already applied")
        return 0
    if old not in src:
        print("ANCHOR MISSING - upstream moved; refusing to guess", file=sys.stderr)
        return 1

    target.write_text(src.replace(old, new, 1))
    print(f"cadence fix: applied in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
