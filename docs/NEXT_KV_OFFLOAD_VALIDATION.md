# NEXT (do not drop): KV-offload validation — task #27

It is the load-bearing
check before betting a multi-GPU serving plan on RAM+NVMe KV offload.

## Why it matters
A many-agents × 1M-context serving plan at this scale assumes sleeping sessions' KV can
be evicted to NVMe and restored per-worker. If the connector doesn't play with
PP or K3's hybrid layers, the plan breaks — quietly.

## Checklist (validate on a 2×3090 box, our slice, ~1 day)
1. vLLM + a KV connector (LMCache OR native v1 offloading) + **PP=2**.
2. Per-stage evict/restore: KV is spread across PP ranks — offload must work on
   EACH rank, not just the driver.
3. Hybrid K3: connector must not choke on **KDA state** (mamba-like, tiny —
   fine to keep in VRAM forever). **THE SUBTLE BUG:** if the connector saves
   MLA-KV but NOT the KDA state, a resumed session has correct MLA memory and
   LOST KDA memory → generation "continues" on silently-corrupted context.
   Catch it with greedy determinism before/after a resume.
4. Measure: session lift time (target ~sec/GB), greedy parity pre/post resume.
5. **fp8 pages (656B):** the connector must NOT hardcode a standard page size,
   or it fights our just-landed fp8-KV port. Newly urgent since fp8_ds_mla works.

## How
Same 2-card phase as the fp8-KV port PP=2 leg (task #26 tail) — do both together.
First read the `offload-compat` research topic result (LMCache vs native, PP +
custom-page-size support) before touching hardware.
