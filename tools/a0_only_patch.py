#!/usr/bin/env python3
"""Apply ONLY the draft-model PP verification relaxation, nothing else.

For the upstream reproduction we need a tree that is otherwise stock. The full
patcher also carries our Kimi-K3 and DSpark work, which would let a maintainer
say the finding is ours rather than vLLM's. This applies the single block from
issue 1 - the one without which MTP under PP cannot start at all - and stops.

Usage: python a0_only_patch.py <site-packages>
"""

import sys
from pathlib import Path

OLD = """        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config
            )"""

NEW = """        if self.draft_model_config:
            _dpc = self.draft_parallel_config
            if _dpc.pipeline_parallel_size > 1:
                _pp = _dpc.pipeline_parallel_size
                object.__setattr__(_dpc, "pipeline_parallel_size", 1)
                try:
                    self.draft_model_config.verify_with_parallel_config(_dpc)
                finally:
                    object.__setattr__(_dpc, "pipeline_parallel_size", _pp)
            else:
                self.draft_model_config.verify_with_parallel_config(_dpc)"""


def main() -> int:
    sp = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if sp is None:
        print("usage: a0_only_patch.py <site-packages>", file=sys.stderr)
        return 1
    target = sp / "vllm" / "config" / "speculative.py"
    src = target.read_text()
    if NEW in src:
        print("A0 only: already applied")
        return 0
    if OLD not in src:
        print("ANCHOR MISSING - upstream moved; refusing to guess", file=sys.stderr)
        return 1
    target.write_text(src.replace(OLD, NEW, 1))
    print(f"A0 only: applied in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
