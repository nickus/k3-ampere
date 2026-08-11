"""Instrumentation for the DSpark-under-PP gates.

Gate 2 is the one that matters: the taps the draft receives at PP=2 must equal
the taps it receives at PP=1. That failure mode is silent — the model still
generates, it just conditions the draft on a quarter of its inputs — so booting
proves nothing and only a value comparison does.

This wraps `K3DSparkForCausalLM.combine_hidden_states`, which is the single
funnel every tap passes through on its way into the draft, and prints a
fingerprint of its input. Run the same prompt at PP=1 and PP=2 and diff.

Import via PYTHONSTARTUP-style injection:
    PYTHONPATH=/workspace/k3 VLLM_DSPARK_PROBE=1 vllm serve ...
and add `import dspark_pp_probe` to a sitecustomize.py on the path.
"""

import os

_ENABLED = os.environ.get("VLLM_DSPARK_PROBE") == "1"


def install() -> None:
    if not _ENABLED:
        return
    import torch
    from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM

    original = K3DSparkForCausalLM.combine_hidden_states

    def probed(self, hidden_states: torch.Tensor):
        t = hidden_states.detach()
        # float64 accumulation so the fingerprint is not itself lossy, and a
        # per-column checksum so a *reordered* set of taps is distinguishable
        # from a correct one — a plain sum would hide ordering bugs.
        flat = t.reshape(-1, t.shape[-1]).to(torch.float64)
        cols = flat.sum(dim=0)
        idx = torch.arange(cols.numel(), device=cols.device, dtype=torch.float64)
        print(
            f"[DSPARK_PROBE] shape={tuple(t.shape)} "
            f"sum={flat.sum().item():.10e} "
            f"weighted={(cols * (idx + 1)).sum().item():.10e} "
            f"absmax={flat.abs().max().item():.10e}",
            flush=True,
        )
        return original(self, hidden_states)

    K3DSparkForCausalLM.combine_hidden_states = probed
    print("[DSPARK_PROBE] installed", flush=True)


install()
