# What K3 should do on the rig — measured inputs, explicit arithmetic

Written 2026-08-12 from the 8x3090 campaign. Every input below is measured; the
projection is arithmetic on top, and the assumptions are named so they can be
attacked.

## What was measured, and what carries over unchanged

Kimi-K3 slice — 24 of 448 experts, but **all 93 layers**, real KDA + MLA, real
MXFP4, real tokenizer, real DSpark draft — on 8x3090, PP=8, concurrency 1:

```
spec off : 172.18 ms/token   5.81 tok/s
spec on  : 109.22 ms/token   9.16 tok/s      1.577x
aggregate, spec off, conc 1 / 2 / 4 : 5.81 / 9.66 / 14.70 tok/s
```

Per-stage cost, measured directly by sweeping PP on a small stand (twice, in both
orders, to rule out a first-run penalty):

```
PP=2  18.1 ms      PP=4  26.6 ms      PP=8  37.6 ms
slope over 2->4->8 : ~3.2 ms per token per added stage
```

(PP=1 is reproducibly *slower* than PP=2 — 29 ms — so the single-stage path is a
different regime and is excluded from the fit. Nobody runs K3 on one card.)

**Two things carry over to the full 448-expert model exactly**, and they are the
ones that matter:

1. **Per-token compute is identical.** Routing is top-16 either way, plus 2
   shared experts. Expert *count* changes how much is resident, not how much is
   read or multiplied per token.
2. **KV per sequence is identical.** Same 93 layers, same hidden 7168, same KDA.
   Measured: ~0.48 GiB of fixed state per sequence (KDA, length-independent)
   plus ~0.4 GiB per 1k tokens of context.

## Memory, on 50x24 GB

```
REAP-448 MXFP4 weights        837 GB   ->  16.7 GB per card
raw capacity                 1200 GB
at utilization 0.90          1080 GB   ->  ~240 GB left for KV
```

Concurrent sequences that fits:

```
 2k context  (~1.3 GiB/seq)  ->  ~180
 8k context  (~3.7 GiB/seq)  ->  ~65
16k context  (~6.9 GiB/seq)  ->  ~35
```

This part is solid — it is measured KV arithmetic, not extrapolation.

## Decode latency, single request

Split the measured 172.18 ms at PP=8 into its two parts:

```
stage overhead   8 x 3.2  =  26 ms
model work       remainder = 146 ms      (93 layers, unchanged on the rig)
```

At 50 stages the model work is the same chain and the overhead scales:

```
146 + 50 x 3.2  =  ~306 ms/token   ->  ~3.3 tok/s per agent, batch 1
with spec decode (1.577x)          ->  ~5.2 tok/s per agent
```

**More stages make a single request slower, not faster.** Pipeline parallelism
buys capacity, never latency. 837 GB on 24 GB cards forces ~35-50 stages, and
each one costs ~3.2 ms per token.

## Aggregate throughput, which is what a 100-agent swarm actually consumes

Per-stage time is ~306/50 ≈ 6.1 ms. A perfectly filled pipeline emits one token
per stage-time, so the ceiling is ~160 tok/s across all agents. Measured filling
efficiency on this campaign was well under ceiling — 8 stages at concurrency 4
delivered 14.7 tok/s against a ~46 tok/s ceiling, about 32%.

```
realistic aggregate, spec off :  ~60-120 tok/s
realistic aggregate, spec on  : ~100-190 tok/s
```

Spread over 100 agents that is ~1-2 tok/s each **if all 100 generate at once**.
Coding agents are bursty, so with ~20 generating simultaneously each sees
~5-9 tok/s. Plan capacity against the burst count, not the agent count.

## The biggest lever, and it is untested

**Every number above was measured with `--enforce-eager`.** At 50 stages the
stage overhead is 165 ms of the projected 306 — more than half the per-token
budget — and that overhead is kernel-launch and Python dispatch, which is exactly
what CUDA graphs remove. Cut it 3x and per-token drops to ~200 ms with aggregate
roughly doubling.

So the first experiment on the rig is not a new model or a new kernel: it is
turning eager mode off and re-measuring the PP slope. Second lever is fewer
stages — a smaller REAP variant trades quality for both latency and capacity, and
the trade is linear in stage count.

## Confidence

- Memory fit and concurrency: **measured**, carries over directly.
- 1.577x from speculation: **measured** on real weights, at concurrency 1, on a
  24-expert slice. The draft was trained against the full 448-expert target, so
  on the real model acceptance should be *better*, not worse — but that is an
  expectation, not a measurement.
- Per-token latency at 50 stages: **projection**, +/-30%. The stage cost came
  from a small stand; K3's hidden state is 7x wider per hop, still small in
  absolute terms and latency-dominated, but it is an extrapolation.
- Aggregate throughput: **weakest number here.** Pipeline filling was measured
  only to concurrency 4 on 8 stages, because the slice plus draft left under a
  gigabyte of KV. On the rig this is the first thing to measure properly.
