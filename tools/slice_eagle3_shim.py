# SPDX-License-Identifier: Apache-2.0
"""TEST-HARNESS ONLY — not part of the upstream DSpark/PP patch.

Our synthetic slice runs the text-only `KimiLinearForCausalLM` architecture.
The real model runs `KimiK3ForConditionalGeneration`, which declares
`SupportsEagle3`; the text-only wrapper does not, even though the inner
`KimiLinearModel` already carries `EagleModelMixin` and implements the tap
machinery. So the slice trips `RuntimeError: Model does not support EAGLE3
interface` before it can exercise the tap plumbing we actually want to test.

This shim forwards the two protocol methods from the wrapper to the inner model
so the slice can stand in for the production architecture. It changes no
behaviour that the test measures — the taps, the PP transport and the draft are
all untouched.

Keep this OUT of anything proposed upstream: there the right fix is either to
declare the interface on `KimiLinearForCausalLM` properly, or to test against
`KimiK3ForConditionalGeneration`.

    python tools/slice_eagle3_shim.py [site-packages-path]
"""

import sys


def main(sp: str) -> None:
    path = f"{sp}/vllm/models/kimi_k3/nvidia/model.py"
    src = open(path).read()

    marker = "    # --- test shim: expose EAGLE3 taps on the text-only wrapper"
    if marker in src:
        print("  eagle3 shim: already applied")
        return

    anchor = "class KimiLinearForCausalLM("
    assert anchor in src, "eagle3 shim: KimiLinearForCausalLM not found"

    # Find the end of the class signature (the line with the closing paren + colon)
    i = src.index(anchor)
    j = src.index("):", i) + len("):\n")

    shim = f'''
{marker}
    # (see tools/slice_eagle3_shim.py — the inner model already implements the
    # tap machinery via EagleModelMixin; only the wrapper's protocol surface is
    # missing, which blocks the slice from standing in for the real class.)
    supports_eagle3 = True
    # The protocol is runtime-checkable, so isinstance() checks member
    # PRESENCE: without these two the wrapper fails the check even with the
    # methods defined. Asked the protocol directly rather than guessing —
    # SupportsEagle3.__protocol_attrs__ lists exactly five members, and these
    # were the two the text-only wrapper lacked.
    has_own_embed_tokens = True
    has_own_lm_head = True

    def set_aux_hidden_state_layers(self, layers) -> None:
        self.model._set_aux_hidden_state_layers(tuple(layers))

    def get_eagle3_default_aux_hidden_state_layers(self):
        n = self.config.num_hidden_layers
        # Mirrors the usual early/middle/late choice; for the slice the draft
        # config supplies explicit ids anyway, so this is only a fallback.
        return tuple(sorted({{max(1, n // 4), max(1, n // 2), max(1, n - 1)}}))
'''
    open(path, "w").write(src[:j] + shim + src[j:])
    print("  eagle3 shim: applied")
    print("EAGLE3_SHIM_DONE")


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/venv/nm/lib/python3.12/site-packages"
    )
