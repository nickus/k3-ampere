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

    # ---- V1: one rank-invariant switch, decided from config, not from local
    # objects. `self.speculator` is None on every non-last rank by construction,
    # and `num_speculative_steps > 0` also catches diffusion models that have no
    # speculator at all - either predicate makes the sender and the receiver
    # disagree, which is a cluster hang, not a wrong answer.
    ok &= patch(
        pp,
        """    def __init__(
        self, max_num_reqs: int, num_speculative_steps: int, device: torch.device
    ):""",
        """    def __init__(
        self,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
        relay_drafts: bool = False,
    ):""",
        "V1 handler takes the switch",
    )
    ok &= patch(
        pp,
        "        self.max_sample_len = num_speculative_steps + 1",
        "        self.max_sample_len = num_speculative_steps + 1\n"
        "        self.num_speculative_steps = num_speculative_steps\n"
        "        self.relay_drafts = relay_drafts and num_speculative_steps > 0",
        "V1b store it",
    )
    ok &= patch(
        runner,
        """                num_speculative_steps=self.num_speculative_steps,
                device=self.device,
            )""",
        """                num_speculative_steps=self.num_speculative_steps,
                device=self.device,
                relay_drafts=self.speculative_config is not None,
            )""",
        "V1c runner passes it",
    )

    # ---- V2: the drafts ride the EXISTING broadcast. Two lines above the
    # insertion point the author stacks num_sampled and num_rejected precisely to
    # avoid a second collective; a third one would contradict that in the same
    # function. Same function on both sides, same predicate, so the ordering is
    # symmetric by construction.
    ok &= patch(
        pp,
        """    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:""",
        """    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor | None = None,
    ) -> None:""",
        "V2 sender signature",
    )
    ok &= patch(
        pp,
        """            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)""",
        """            if self.relay_drafts:
                assert draft_tokens is not None
                # A fresh gather, never propose()'s view of the speculator's
                # persistent buffer: record_stream defers allocator reuse, it does
                # not stop the next step overwriting that memory from the main
                # stream while this read is still in flight.
                _d = draft_tokens.contiguous()
                torch.distributed.broadcast(
                    _d, src=self.last_rank, group=self.broadcast_group
                )
                _d.record_stream(self.broadcast_stream)
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)""",
        "V2b sender payload",
    )
    ok &= patch(
        pp,
        """            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)""",
        """            drafts = None
            if self.relay_drafts:
                drafts = torch.empty(
                    num_reqs,
                    self.num_speculative_steps,
                    dtype=torch.int64,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    drafts, src=self.last_rank, group=self.broadcast_group
                )
                drafts.record_stream(self.main_stream)
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)""",
        "V2c receiver payload",
    )

    # ---- V3: carry it on the deferred slot, exactly like the sampled tokens.
    ok &= patch(
        pp,
        "    gen_at_receive_np: np.ndarray  # [num_reqs]",
        "    gen_at_receive_np: np.ndarray  # [num_reqs]\n"
        "    # Drafts for these requests' NEXT step; None when not relaying.\n"
        "    draft_tokens: torch.Tensor | None = None",
        "V3 slot field",
    )
    ok &= patch(
        pp,
        """            need_sampled_mask,
            gen_at_receive_np,
        )
        return bool(need_sampled_mask.all())""",
        """            need_sampled_mask,
            gen_at_receive_np,
            drafts,
        )
        return bool(need_sampled_mask.all())""",
        "V3b stash on slot",
    )
    ok &= patch(
        pp,
        """            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
        )""",
        """            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
            draft_tokens=slot.draft_tokens,
        )""",
        "V3c return from FIFO",
    )

    # ---- V4: apply on the receiving ranks. idx_mapping carries -1 for rows
    # filtered since receive - advanced indexing would wrap that to the LAST row
    # and silently clobber an unrelated request, so mask first.
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
            _valid = idx_mapping >= 0
            if bool(_valid.any()):
                self.req_states.draft_tokens[idx_mapping[_valid]] = draft_tokens[
                    _valid
                ]""",
        "V4 apply on receivers",
    )

    # ---- V5: move the relay below propose() so the payload is this step's
    # drafts, and send the fresh gather the runner already builds.
    ok &= patch(
        runner,
        """        if self.pp_handler is not None:
            # Broadcast to non-last PP ranks (handles spec decode multi-token).
            self.pp_handler.broadcast(
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                input_batch,
            )""",
        """        _pp_relay = None
        if self.pp_handler is not None:
            # Deferred to below propose(): the drafts for the NEXT step do not
            # exist yet here, and the payloads are all materialised by sample()
            # and not mutated afterwards, so moving the send is safe.
            _pp_relay = (sampler_output.sampled_token_ids, num_sampled, num_rejected)""",
        "V5 defer the relay",
    )
    ok &= patch(
        runner,
        """        if self.num_speculative_steps > 0:
            # Spec-decode and diffusion LLMs both use draft tokens but the latter does
            # not have a speculator (i.e. self.speculator is None)
            self.draft_tokens_handler.set_draft_tokens(""",
        """        if _pp_relay is not None:
            assert self.pp_handler is not None
            self.pp_handler.broadcast(
                _pp_relay[0],
                _pp_relay[1],
                _pp_relay[2],
                input_batch,
                self.req_states.draft_tokens[input_batch.idx_mapping]
                if self.pp_handler.relay_drafts
                else None,
            )

        if self.num_speculative_steps > 0:
            # Spec-decode and diffusion LLMs both use draft tokens but the latter does
            # not have a speculator (i.e. self.speculator is None)
            self.draft_tokens_handler.set_draft_tokens(""",
        "V5b send it below propose",
    )

    # ---- V6: the finish test reads an inflated num_computed_tokens under PP and
    # ends the relay a few steps early; widening it by the speculative depth can
    # only delay calling a request finished, never advance it.
    ok &= patch(
        pp,
        """def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:""",
        """def compute_need_sampled_mask(
    input_batch: InputBatch, spec_slack: int = 0
) -> np.ndarray | None:""",
        "V6 finish-test signature",
    )
    ok &= patch(
        pp,
        """    not_finishing = np.maximum(old_computed, prefill_len) + 1 < max_seq_len""",
        """    not_finishing = (
        np.maximum(old_computed, prefill_len) + 1 < max_seq_len + spec_slack
    )""",
        "V6b apply slack",
    )
    ok &= patch(
        pp,
        """        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:""",
        """        need_sampled_mask = compute_need_sampled_mask(
            input_batch, self.num_speculative_steps
        )
        if need_sampled_mask is None:""",
        "V6c receiver slack",
    )
    ok &= patch(
        pp,
        """        if compute_need_sampled_mask(input_batch) is None:""",
        """        if compute_need_sampled_mask(input_batch, self.num_speculative_steps) is None:""",
        "V6d sender slack",
    )

    print("SPECDEC_PP_RELAY_DONE" if ok else "SPECDEC_PP_RELAY_INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
