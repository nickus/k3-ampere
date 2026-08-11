#!/usr/bin/env python3
"""Measure the claim in vllm#51752 that block sizes diverge across PP ranks
even with no KV connector attached.

The issue says the OffloadingConnector's divisibility assert is only what makes
the divergence *visible*, and that the divergence itself is connector
independent. That is an assertion, not a measurement — this makes it one.

Instruments `Platform.update_block_size_for_backend` to print, per rank:
  - the block sizes on entry,
  - whether it took the `backend_cls is None` early return,
  - the block sizes on exit.

Run the server afterwards with NO --kv-transfer-config. If the printed exit
values differ between ranks, the divergence is real without any connector.
"""

import sys
from pathlib import Path

TAG = "[BLOCKSIZE_PROBE]"


def main() -> int:
    import vllm

    sp = Path(vllm.__file__).parent
    target = sp / "platforms" / "interface.py"
    src = target.read_text()

    if TAG in src:
        print("probe already installed")
        return 0

    anchor = """        backend_cls = cls._find_non_ssm_backend(vllm_config)
        if backend_cls is None:
            return
"""
    if anchor not in src:
        print("ANCHOR MISSING — upstream moved; refusing to guess", file=sys.stderr)
        return 1

    replacement = '''        import os as _os

        def _bs_report(_stage, _cc):
            print(
                f"{TAG} pid={_os.getpid()} stage={_stage} "
                f"block_size={getattr(_cc, 'block_size', None)} "
                f"mamba_block_size={getattr(_cc, 'mamba_block_size', None)}",
                flush=True,
            )

        _bs_report("entry", cache_config)
        backend_cls = cls._find_non_ssm_backend(vllm_config)
        if backend_cls is None:
            _bs_report("EARLY_RETURN_no_attention_layer", cache_config)
            return
'''.replace("{TAG}", TAG)

    src = src.replace(anchor, replacement, 1)

    # Also report on the way out, wherever the function ends.
    tail_anchor = "    @classmethod\n    def _align_hybrid_block_size("
    if tail_anchor in src:
        src = src.replace(
            tail_anchor,
            '        _bs_report("exit_aligned", cache_config)\n\n' + tail_anchor,
            1,
        )

    target.write_text(src)
    print(f"blocksize probe installed in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
