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
    if pp.world_size != 1 and not pp.is_last_rank:
        # The draft lives on the last stage: that rank already owns lm_head, and
        # vLLM's layer split deliberately gives the last stage fewer layers, so
        # it has the most room. Other ranks only carry the taps (see the
        # KimiK3Model patch) and hold no draft weights.
        return draft_model''',
        "A1 allow PP, draft on last rank",
    )

    # ---- A2: the embedding lives on rank 0; bring a copy to the draft's rank
    patch(
        utils,
        """    target_embed = getattr(target_inner, "embed_tokens", None)""",
        '''    target_embed = getattr(target_inner, "embed_tokens", None)
    if pp.world_size != 1 and (
        target_embed is None or not hasattr(target_embed, "weight")
    ):
        # Last rank under PP: embed_tokens is a PPMissingLayer here and the real
        # weight sits on rank 0. Broadcast it once at load time (163840 x 7168
        # bf16 = 2.35 GB) rather than re-reading the checkpoint.
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        cfg = vllm_config.model_config.hf_config
        cfg = getattr(cfg, "text_config", cfg)
        target_embed = VocabParallelEmbedding(
            cfg.vocab_size,
            cfg.hidden_size,
            params_dtype=vllm_config.model_config.dtype,
        ).to(draft_model.model.layers[0].self_attn.o_proj.weight.device)
        pp.broadcast(target_embed.weight.data, src=0)''',
        "A2 broadcast embedding to the draft rank",
    )

    print("DSPARK_PP_PATCHES_DONE")


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/venv/nm/lib/python3.12/site-packages"
    )
