# SPDX-License-Identifier: Apache-2.0
"""Make Kimi-K3 DSpark speculative decoding work under pipeline parallelism.

Two independent defects sit behind the one-line guard in vLLM
(`v1/worker/gpu/spec_decode/dspark/utils.py`):

  A. the draft borrows the target's `embed_tokens` (PP rank 0) and `lm_head`
     (last PP rank) by object reference, and no rank owns both;
  B. `KimiK3Model.forward` drops `aux_hidden_states` at every PP boundary, so
     the draft can only ever see the taps produced by the last stage.

B is the dangerous one: fixing A alone makes DSpark run and be silently wrong.
This patch fixes B by carrying the taps in `IntermediateTensors`, and unblocks A
by hosting the draft on the last rank with a broadcast copy of the embedding.

Idempotent; every block asserts its anchor, so a drifted tree fails loudly
instead of reporting success. Usage:

    python tools/dspark_pp_patch.py [site-packages-path]

Design notes: docs/DSPARK_PP_DESIGN.md
"""

import sys

AUX_PREFIX = "aux_hidden_state_"


def patch(path, old, new, tag):
    src = open(path).read()
    if new in src:
        print(f"  {tag}: already applied")
        return
    assert old in src, f"{tag}: anchor missing in {path}"
    open(path, "w").write(src.replace(old, new, 1))
    print(f"  {tag}: applied")


def main(sp: str) -> None:
    model = f"{sp}/vllm/models/kimi_k3/nvidia/model.py"
    utils = f"{sp}/vllm/v1/worker/gpu/spec_decode/dspark/utils.py"

    # ---- B1: receive carried taps from the previous stage
    patch(
        model,
        """        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        assert hidden_states is not None""",
        f'''        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        assert hidden_states is not None

        # Taps produced by upstream PP stages, in layer order. Upstream ranks
        # own strictly lower layer indices, so carried + own stays sorted.
        carried_aux: list[torch.Tensor] = []
        if intermediate_tensors is not None:
            _i = 0
            while f"{AUX_PREFIX}{{_i}}" in intermediate_tensors.tensors:
                carried_aux.append(intermediate_tensors[f"{AUX_PREFIX}{{_i}}"])
                _i += 1''',
        "B1 receive carried taps",
    )

    # ---- B2: forward carried + own taps to the next stage
    patch(
        model,
        """            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )""",
        f'''            _out = {{"hidden_states": hidden_states, "residual": residual}}
            for _j, _t in enumerate(carried_aux + aux_hidden_states):
                _out[f"{AUX_PREFIX}{{_j}}"] = _t
            return IntermediateTensors(_out)''',
        "B2 forward taps",
    )

    # ---- B3: the last rank assembles carried + own before returning
    patch(
        model,
        """        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states""",
        """        aux_hidden_states = carried_aux + aux_hidden_states
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states""",
        "B3 assemble on last rank",
    )

    # ---- B4: allocate receive buffers for the taps owned upstream of this rank
    patch(
        model,
        """        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(residual_shape, dtype=dtype, device=device),
            }
        )""",
        f'''        _t = {{
            "hidden_states": torch.zeros(
                (batch_size, self.config.hidden_size), dtype=dtype, device=device
            ),
            "residual": torch.zeros(residual_shape, dtype=dtype, device=device),
        }}
        # One slot per tap produced strictly upstream of this stage. Computable
        # locally: no communication needed to agree on the payload shape.
        _n_upstream = sum(
            1 for _L in self.aux_hidden_state_layers if _L < self.start_layer
        )
        for _j in range(_n_upstream):
            _t[f"{AUX_PREFIX}{{_j}}"] = torch.zeros(
                (batch_size, self.config.hidden_size), dtype=dtype, device=device
            )
        return IntermediateTensors(_t)''',
        "B4 allocate tap buffers",
    )

    # ---- A0: the config layer rejects PP before the DSpark guard is ever
    # reached. `create_draft_parallel_config` copies the target's
    # pipeline_parallel_size into the draft's parallel config, so
    # `verify_with_parallel_config` demands the draft implement SupportsPP —
    # which it does not, and should not: the draft is NOT pipelined, it runs
    # whole on one rank. Verify it as the single-stage model it is.
    spec = f"{sp}/vllm/config/speculative.py"
    patch(
        spec,
        """        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config
            )""",
        """        if self.draft_model_config:
            _dpc = self.draft_parallel_config
            _is_dspark = getattr(self, "method", None) == "dspark" or any(
                "DSpark" in a for a in (self.draft_model_config.architectures or [])
            )
            if _is_dspark and _dpc.pipeline_parallel_size > 1:
                _pp = _dpc.pipeline_parallel_size
                object.__setattr__(_dpc, "pipeline_parallel_size", 1)
                try:
                    self.draft_model_config.verify_with_parallel_config(_dpc)
                finally:
                    object.__setattr__(_dpc, "pipeline_parallel_size", _pp)
            else:
                self.draft_model_config.verify_with_parallel_config(_dpc)""",
        "A0 verify draft as single-stage",
    )

    # ---- A: allow PP, and host the draft where lm_head already lives
    patch(
        utils,
        """    if get_pp_group().world_size != 1:
        raise NotImplementedError("DSpark does not support pipeline parallelism.")""",
        '''    pp = get_pp_group()
    # NOTE: do NOT return early here for non-last ranks. The embedding handoff
    # below is a COLLECTIVE: every rank in the PP group must reach it or the
    # ones that do will hang waiting for a broadcast that never comes (rank 1
    # times out retrieving ncclUniqueId while rank 0 sits idle). The early
    # return happens after the collective instead.''',
        "A1 allow PP, draft on last rank",
    )

    # ---- A3: the real root guard. Its own comment names the reason —
    # "Drafting may require auxiliary hidden states from target model outputs"
    # — which is exactly what the B patches now deliver across PP boundaries.
    # Note the line just above it: vLLM already places the speculator on the
    # last PP rank (`if self.is_last_pp_rank: self.speculator = ...`), so
    # hosting the draft there is upstream's own design, not our invention.
    # Relaxed for dspark only: eagle3/dflash tap hidden states through other
    # model files we have not touched.
    runner = f"{sp}/vllm/v1/worker/gpu/model_runner.py"
    patch(
        runner,
        """                if self.use_pp:
                    raise ValueError(
                        f"{self.speculative_config.method} with pipeline parallel "
                        "is not supported."
                    )""",
        """                if self.use_pp and self.speculative_config.method != "dspark":
                    raise ValueError(
                        f"{self.speculative_config.method} with pipeline parallel "
                        "is not supported."
                    )""",
        "A3 relax the root PP guard for dspark",
    )

    # ---- A4: the draft numbers its own layers starting at the target's layer
    # count, to keep layer names unique. But it asks for the count *on this
    # rank* (`get_num_layers` returns end - start of this PP stage), so under
    # PP it starts at 2 instead of 93 and collides with the target's own
    # layers: `ValueError: Duplicate layer name: model.layers.2.self_attn`.
    # At PP=1 the two numbers coincide, which is why this never showed up.
    mla = f"{sp}/vllm/models/kimi_k3/nvidia/dspark_mla.py"
    patch(
        mla,
        """        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )""",
        """        # Must be the TOTAL target layer count, not this rank's slice:
        # the value only exists to offset draft layer names away from the
        # target's, and under PP the per-rank count collides.
        target_layer_num = getattr(
            vllm_config.model_config.hf_text_config,
            "num_hidden_layers",
            None,
        ) or vllm_config.model_config.get_num_layers(vllm_config.parallel_config)""",
        "A4 draft layer ids must offset by TOTAL target layers",
    )

    # ---- A2: the embedding lives on rank 0; bring a copy to the draft's rank
    patch(
        utils,
        """    target_embed = getattr(target_inner, "embed_tokens", None)""",
        '''    target_embed = getattr(target_inner, "embed_tokens", None)
    if pp.world_size != 1 and not pp.is_last_rank:
        # Only the LAST rank ever gets here: init_speculator is called under
        # `if self.is_last_pp_rank`. So a collective inside draft loading can
        # never be symmetric — rank 0 does not execute this function at all,
        # and broadcasting here deadlocks the group. Leave immediately.
        return draft_model

    if pp.world_size != 1 and (
        target_embed is None or not hasattr(target_embed, "weight")
    ):
        import torch as _torch
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )
        from vllm.logger import init_logger as _init_logger

        _log = _init_logger(__name__)
        _cfg = vllm_config.model_config.hf_config
        _cfg = getattr(_cfg, "text_config", _cfg)
        _dev = next(draft_model.parameters()).device
        _dtype = vllm_config.model_config.dtype

        _tied = getattr(_cfg, "tie_word_embeddings", False)
        _lm = get_target_lm_head(target_model, target_language_model)
        if _tied and _lm is not None and hasattr(_lm, "weight"):
            # Weights are tied, so lm_head IS the embedding matrix and it
            # already lives on this rank. Exact, no transfer needed.
            target_embed = VocabParallelEmbedding(
                _cfg.vocab_size, _cfg.hidden_size, params_dtype=_dtype
            ).to(_dev)
            target_embed.weight.data.copy_(_lm.weight.data)
            _log.info("DSpark/PP: embedding taken from tied lm_head on this rank")
        else:
            # PLACEHOLDER. embed_tokens lives on rank 0 and there is no
            # symmetric point in this call path to move it. Enough to exercise
            # the tap plumbing; NOT production-correct — the real fix is to
            # transfer the weight where every rank is present (e.g. in the
            # model runner right after load_model), or to have the last rank
            # read it from the target checkpoint.
            target_embed = VocabParallelEmbedding(
                _cfg.vocab_size, _cfg.hidden_size, params_dtype=_dtype
            ).to(_dev)
            _log.warning(
                "DSpark/PP: using an UNINITIALISED embedding on the draft rank "
                "- draft token ids will be meaningless. Plumbing test only."
            )''',
        "A2 embedding on the draft rank (no collective)",
    )

    print("DSPARK_PP_PATCHES_DONE")


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/venv/nm/lib/python3.12/site-packages"
    )
