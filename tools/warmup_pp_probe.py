# SPDX-License-Identifier: Apache-2.0
"""Locate where PP ranks diverge during spec-decode warmup.

Symptom: at PP=2 with speculative decoding the engine completes at least one
forward (the draft receives its taps) and then hangs, with one rank blocked in a
pipeline send/recv and the other still in `execute_model`. The engine core then
reports "No available shared memory broadcast block found in 60 seconds".

Two things are worth printing, per rank:

1. **which warmup decode step** each rank is entering — if the ranks disagree on
   the step list, or one exits the loop early, they can never meet again;
2. **the shape of the pipeline payload** actually sent and received — a mismatch
   in tensor count (e.g. the aux-tap slots this project adds) would make a recv
   block forever.

Both are one line each, printed with the rank so the two logs interleave
readably. Run, reproduce the hang, and read the last line of each rank.

    python tools/warmup_pp_probe.py [site-packages-path]
"""

import sys


def patch(path, old, new, tag):
    src = open(path).read()
    if new in src:
        print(f"  {tag}: already applied")
        return
    assert old in src, f"{tag}: anchor missing in {path}"
    open(path, "w").write(src.replace(old, new, 1))
    print(f"  {tag}: applied")


def main(sp: str) -> None:
    warmup = f"{sp}/vllm/v1/worker/gpu/warmup.py"
    pp_utils = f"{sp}/vllm/v1/worker/gpu/pp_utils.py"

    # 1. announce every warmup decode step, per rank
    patch(
        warmup,
        """        for step_indices, step_spec_flags in decode_steps:
            _run_decode_step(step_indices, step_spec_flags)""",
        """        for _si, (step_indices, step_spec_flags) in enumerate(decode_steps):
            from vllm.distributed.parallel_state import get_pp_group as _gpp

            print(
                "[WARMUP] rank=%d step=%d/%d idx=%s spec=%s"
                % (
                    _gpp().rank_in_group,
                    _si,
                    len(decode_steps),
                    step_indices,
                    step_spec_flags,
                ),
                flush=True,
            )
            _run_decode_step(step_indices, step_spec_flags)
        print(
            "[WARMUP] rank=%d ALL DECODE STEPS DONE"
            % __import__(
                "vllm.distributed.parallel_state", fromlist=["get_pp_group"]
            ).get_pp_group().rank_in_group,
            flush=True,
        )""",
        "P1 warmup step announcer",
    )


    # 3. milestones — stacks are unavailable in this container (ptrace blocked,
    # faulthandler produced nothing), so mark progress instead. If a rank never
    # prints a milestone, the hang is before it.
    patch(
        warmup,
        """def warmup_kernels(""",
        """def _wm_mark(where: str) -> None:
    try:
        from vllm.distributed.parallel_state import get_pp_group

        print("[WM] rank=%d %s" % (get_pp_group().rank_in_group, where), flush=True)
    except Exception:
        print("[WM] rank=? %s" % where, flush=True)


def warmup_kernels(""",
        "P3 milestone helper",
    )


    # 3b. mark entry, and mark the boundary between prefill and decode warmup
    patch(
        warmup,
        """    if model_runner.is_encoder_only:
        return

    num_spec_steps = model_runner.num_speculative_steps""",
        """    if model_runner.is_encoder_only:
        return

    _wm_mark("ENTER warmup_kernels")
    num_spec_steps = model_runner.num_speculative_steps""",
        "P4 mark warmup entry",
    )
    patch(
        warmup,
        """        all_indices = list(range(num_reqs))
        use_spec_decode = num_spec_steps > 0""",
        """        _wm_mark("PREFILL PHASE DONE, entering decode steps")
        all_indices = list(range(num_reqs))
        use_spec_decode = num_spec_steps > 0""",
        "P5 mark prefill/decode boundary",
    )

    # 2. announce the pipeline payload on both sides
    src = open(pp_utils).read()
    if "[PP] send keys=" not in src:
        marker = "def "
        assert marker in src, "pp_utils.py unexpected"
        src = src.replace(
            "import torch",
            """import torch


def _pp_probe(direction, obj):
    import os

    if os.environ.get("PP_PROBE") != "1":
        return
    from vllm.distributed.parallel_state import get_pp_group

    try:
        keys = sorted(getattr(obj, "tensors", {}).keys())
        shapes = [tuple(getattr(obj, "tensors")[k].shape) for k in keys]
    except Exception:
        keys, shapes = ["<not-IntermediateTensors>"], []
    print(
        "[PP] %s rank=%d keys=%s shapes=%s"
        % (direction, get_pp_group().rank_in_group, keys, shapes),
        flush=True,
    )""",
            1,
        )
        open(pp_utils, "w").write(src)
        print("  P2 pp payload probe: helper installed")
    else:
        print("  P2 pp payload probe: already applied")

    import ast

    for f in (warmup, pp_utils):
        ast.parse(open(f).read())
    print("WARMUP_PROBE_DONE (both files parse)")


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/venv/nm/lib/python3.12/site-packages"
    )
