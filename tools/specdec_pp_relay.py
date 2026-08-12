#!/usr/bin/env python3
"""Carry draft tokens to every PP rank on the relay that already exists.

WHY THIS SHAPE
--------------
Measured here: the first PP rank embeds token id 0 for every speculative
position, because draft VALUES live only in `req_states.draft_tokens`, written
right after `speculator.propose()` — which runs on the last rank only. Row 0
survives because it comes from `last_sampled_tokens`, which IS relayed.

Two earlier attempts failed and are recorded in
`results/specdec_pp4_FIXED_2026-08-12.md`:

  A8  rehydrate from `scheduler_output.scheduled_spec_decode_tokens` — wrong at
      the root: that dict carries counts only, by design (`DraftTokensHandler`
      returns `[-1] * k` unless structured outputs need grammar validation).
  A9  a fresh broadcast at the top of the step — deadlocks, because it puts a
      collective in the flow while the ranks are on different microbatches.

The design here follows what SGLang's PP spec-decode PRs converged on
(sgl-project/sglang#30775): the drafter stays on the LAST stage (it needs the
final hidden states and lm_head), it drafts for round r+1 at the END of round r
("tail drafting"), and the draft tokens ride the SAME last->first relay that
already carries the sampled token every round. Nothing stalls and the draft model
is not replicated.

In vLLM that relay is `PPHandler`: it runs on a side stream with an event and a
FIFO, so a receiver does not block its main stream and consumes the payload
`pp_size` steps later — exactly the lag the pipeline needs. So:

  * receivers post a third broadcast inside `receive()` and stash it in the slot;
  * the last rank issues the matching send AFTER `propose()`, when the drafts for
    the next round actually exist;
  * `get_prev_sampled_outputs()` hands the drafts back with the sampled tokens,
    and `postprocess_sampled()` writes them into `req_states.draft_tokens`.

NCCL matches collectives by their order on the group, not by wall clock, so the
sender issuing its third broadcast later in the step is fine.

Idempotent; every block asserts its anchor.
"""

import sys
from pathlib import Path


def patch(path: Path, old: str, new: str, tag: str) -> bool:
    src = path.read_text()
    if new in src:
        print(f"  {tag}: already applied")
        return True
    if old not in src:
        print(f"  {tag}: ANCHOR MISSING in {path.name}", file=sys.stderr)
        return False
    path.write_text(src.replace(old, new, 1))
    print(f"  {tag}: applied")
    return True


def main() -> int:
    import vllm

    sp = Path(vllm.__file__).parent
    pp = sp / "v1" / "worker" / "gpu" / "pp_utils.py"
    runner = sp / "v1" / "worker" / "gpu" / "model_runner.py"
    ok = True

    # R1: carry a draft-token buffer on the pending-recv slot.
    ok &= patch(
        pp,
        "    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]",
        "    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]\n"
        "    draft_tokens: torch.Tensor | None = None  # [num_reqs, num_spec_steps]",
        "R1 slot carries drafts",
    )

    # R2: receivers post the third broadcast and keep the buffer.
    ok &= patch(
        pp,
        """            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )""",
        """            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            # Third payload: the draft tokens for these requests' NEXT step. The
            # last rank issues the matching send after propose(); NCCL matches by
            # order on the group, so issuing it later there is fine.
            self._pending_drafts = torch.empty(
                num_reqs,
                max(1, self.num_speculative_steps),
                dtype=torch.int64,
                device=self.device,
            )
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )""",
        "R2 receiver allocates the draft buffer",
    )

    # R3: remember how many speculative steps there are (used by R2).
    ok &= patch(
        pp,
        "        self.max_sample_len = num_speculative_steps + 1",
        "        self.max_sample_len = num_speculative_steps + 1\n"
        "        self.num_speculative_steps = num_speculative_steps\n"
        "        self._pending_drafts: torch.Tensor | None = None",
        "R3 remember the spec depth",
    )

    # R4: the sender's half, issued after propose().
    ok &= patch(
        pp,
        "    def get_prev_sampled_outputs(self)",
        '''    def broadcast_drafts(self, draft_tokens: torch.Tensor) -> None:
        """Send the freshly proposed drafts on the same relay, after propose()."""
        assert self.is_last_rank
        with torch.cuda.stream(self.broadcast_stream):
            torch.distributed.broadcast(
                draft_tokens.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            draft_tokens.record_stream(self.broadcast_stream)

    def get_prev_sampled_outputs(self)''',
        "R4 sender half after propose",
    )

    # NOT FINISHED. R1-R4 are the transport half. Still to write, each needing an
    # exact anchor from the installed tree:
    #   R5  stash `self._pending_drafts` into the PendingRecv slot in `receive()`
    #   R6  return it from `get_prev_sampled_outputs()` alongside sampled_tokens
    #   R7  accept it in `postprocess_sampled()` and write
    #       `self.req_states.draft_tokens[idx_mapping] = draft_tokens`
    #   R8  call `pp_handler.broadcast_drafts(draft_tokens)` on the last rank
    #       immediately after `req_states.draft_tokens[...] = draft_tokens`
    # Applying R1-R4 alone changes nothing and is safe, but it is not the fix.
    print("SPECDEC_PP_RELAY_TRANSPORT_ONLY" if ok else "SPECDEC_PP_RELAY_INCOMPLETE")
    print("  R5-R8 (wiring) are NOT implemented yet - see the comment above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
