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

    # R3: remember the spec depth and the pending buffer.
    ok &= patch(
        pp,
        "        self.max_sample_len = num_speculative_steps + 1",
        "        self.max_sample_len = num_speculative_steps + 1\n"
        "        self.num_speculative_steps = num_speculative_steps\n"
        "        self._pending_drafts: torch.Tensor | None = None",
        "R3 remember spec depth",
    )

    # R1: carry the drafts on the slot. MUST go last in the dataclass - a field
    # with a default before the non-default ones does not even construct.
    ok &= patch(
        pp,
        "    gen_at_receive_np: np.ndarray  # [num_reqs]",
        "    gen_at_receive_np: np.ndarray  # [num_reqs]\n"
        "    # Drafts for these requests' NEXT step, relayed from the last rank.\n"
        "    draft_tokens: torch.Tensor | None = None",
        "R1 slot carries drafts",
    )

    # R2: receiver posts the third broadcast, matching the sender's later send.
    ok &= patch(
        pp,
        """            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)""",
        """            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            self._pending_drafts = torch.empty(
                num_reqs,
                max(1, self.num_speculative_steps),
                dtype=torch.int64,
                device=self.device,
            )""",
        "R2 receiver allocates the draft buffer",
    )

    # R2b: the receive itself, after the two existing ones so the order on the
    # group matches the sender.
    ok &= patch(
        pp,
        """            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)""",
        """            if self.num_speculative_steps > 0:
                torch.distributed.broadcast(
                    self._pending_drafts,
                    src=self.last_rank,
                    group=self.broadcast_group,
                )
                self._pending_drafts.record_stream(self.main_stream)
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)""",
        "R2b receiver third broadcast",
    )

    # R5: stash it on the slot (positional - it is the last field).
    ok &= patch(
        pp,
        """            need_sampled_mask,
            gen_at_receive_np,
        )
        return bool(need_sampled_mask.all())""",
        """            need_sampled_mask,
            gen_at_receive_np,
            self._pending_drafts,
        )
        return bool(need_sampled_mask.all())""",
        "R5 stash drafts on the slot",
    )

    # R6: hand them back with the sampled tokens, on the same lag.
    ok &= patch(
        pp,
        """            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
        )""",
        """            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
            draft_tokens=slot.draft_tokens,
        )""",
        "R6 return drafts from the FIFO",
    )

    # R4: the sender half, issued after propose().
    ok &= patch(
        pp,
        "    def get_prev_sampled_outputs(self)",
        '''    def broadcast_drafts(self, draft_tokens: torch.Tensor) -> None:
        """Send freshly proposed drafts on the same relay, after propose()."""
        assert self.is_last_rank
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            _d = draft_tokens.contiguous()
            torch.distributed.broadcast(
                _d, src=self.last_rank, group=self.broadcast_group
            )
            _d.record_stream(self.broadcast_stream)

    def get_prev_sampled_outputs(self)''',
        "R4 sender half",
    )

    # R7: accept and apply them on the receiving ranks.
    ok &= patch(
        runner,
        """        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
    ) -> None:""",
        """        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
        draft_tokens: torch.Tensor | None = None,
    ) -> None:
        if draft_tokens is not None and not self.is_last_pp_rank:
            # idx_mapping carries -1 for rows filtered since receive; those must
            # not be written, or they land on the last row of the buffer.
            _valid = idx_mapping >= 0
            if bool(_valid.any()):
                self.req_states.draft_tokens[idx_mapping[_valid]] = draft_tokens[
                    _valid
                ]""",
        "R7 apply drafts on receiving ranks",
    )

    # R8: send them the moment they exist.
    ok &= patch(
        runner,
        """            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens""",
        """            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
            if self.pp_handler is not None and self.num_speculative_steps > 0:
                self.pp_handler.broadcast_drafts(draft_tokens)""",
        "R8 send drafts after propose",
    )

    print("SPECDEC_PP_RELAY_DONE" if ok else "SPECDEC_PP_RELAY_INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
