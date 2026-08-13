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

---

## MEASURED 2026-08-13: eager vs CUDA graphs, and a correction

The section above said the stage overhead is "kernel-launch and Python dispatch,
which is exactly what CUDA graphs remove — cut it 3x and per-token drops to
~200 ms with aggregate roughly doubling." That was an estimate. Here is the
measurement, same 16-layer stand, same box, both orders:

```
              PP=2      PP=4      PP=8
eager        19.12     32.41     38.74 ms
graphs        2.48      7.46     13.65 ms

least squares:
  eager  : TPOT = 15.95 + 3.03 · pp
  graphs : TPOT = -0.61 + 1.82 · pp
```

**Correction: the per-stage cost falls 1.67×, not 3×** — 3.03 → 1.82 ms. What
does collapse almost entirely is the *fixed* term: 15.95 ms → ~0. On this stand
that term is nearly all launch overhead, because its per-layer arithmetic is
negligible by construction.

On the stand alone, projecting the fit:

```
PP=30 : 107 ms eager  ->  54 ms graphs   1.98x
PP=50 : 167 ms eager  ->  90 ms graphs   1.86x
```

### What this does and does not settle for the rig

The rig's per-token budget is stage overhead **plus** ~148 ms of real K3 model
work (the PP=8 measurement of 172.18 ms minus 8 × 3.03). The sweep above prices
the first term and says nothing about the second, because the stand has almost no
arithmetic in it. So:

```
eager , 50 stages : 148 + 167 = ~315 ms/token   ~3.2 tok/s
graphs, 50 stages : 148 + 90  = ~238 ms/token   ~4.2 tok/s   (1.32x)
```

That 1.32× is a **lower bound**, and it assumes graphs do nothing for the 148 ms.
The stand shows graphs erase essentially all launch overhead inside a stage, and
93 layers of KDA + MLA + MoE launch a great many kernels, so the true figure is
somewhere between 1.32× and roughly 2×. Pinning it down needs an eager-vs-graphs
comparison on a real model, not a stand — which is the measurement running next.

Not "roughly doubling", then. Between a third better and twice as fast, with the
lower end already banked.

## MEASURED on a real model: graphs are worth 3.41×, and my "lower bound" was wrong

GLM-4.5-Air AWQ, PP=4, concurrency 1, 256-token generations, 2 repeats, nothing
varied but `--enforce-eager`:

```
eager  : 50.27, 50.29 ms/token
graphs : 14.72, 14.76 ms/token
         3.41x, saving 35.5 ms per token
```

Decomposed with the per-stage slopes measured above:

```
eager  : 50.28 = model_work + 4 × 3.03  ->  model_work = 38.2 ms
graphs : 14.74 = model_work + 4 × 1.82  ->  model_work =  7.5 ms
                                            5.1x on the model work itself
```

**So the "1.32× lower bound" was built on a false assumption.** I assumed CUDA
graphs do nothing for model work. On a real model at batch 1 they cut it 5.1×,
because batch-1 decode is launch-bound, not arithmetic-bound: the GPU spends its
time being told what to do, not doing it.

### Rig projection, redone from measurements

K3 at PP=8 eager measured 172.18 ms → model work = 172.18 − 8×3.03 = 148 ms.
Applying the measured 5.1× to that term and the measured graph slope to the
stages:

```
                        model work   +  stages        =  per token    per agent
eager , 50 stages :        148       +  50 × 3.03     =  ~300 ms       3.3 tok/s
graphs, 50 stages :         29       +  50 × 1.82     =  ~120 ms       8.3 tok/s
                                                          2.5x
with spec decode (1.577×) on top                      :   ~76 ms      13 tok/s
```

Aggregate ceiling rises with it: per-stage time falls to ~2.4 ms, so a filled
pipeline tops out near 400 tok/s, and at the ~32% filling efficiency measured
here that is ~130 tok/s across the swarm without speculation.

### What is still assumed

The 5.1× model-work reduction is measured on GLM-4.5-Air and applied to K3. Both
are MoE, but K3 adds KDA linear attention, whose many small Triton kernels are if
anything *more* launch-bound — so this is more likely conservative than
optimistic. It is still a transfer between architectures, and the honest way to
close it is to run the K3 slice with graphs. That needs a box with ≥160 GB of
disk; this one has 80 GB, which is my sizing error from when I thought the slice
would not be needed.

## Graphs and speculation together: they coexist, and on GLM speculation then hurts

Same box, same model, `--speculative-config '{"method":"mtp",...}'` added:

```
                       eager        graphs      graphs vs eager
no speculation      50.28 ms      14.74 ms          3.41x
with speculation    47.90 ms      20.40 ms          2.35x
```

Two things worth separating:

1. **Graph capture is not disabled by speculative decoding.** That was the risk
   worth checking — a captured graph is a fixed shape and the verify step's shape
   depends on how many drafts were accepted — and it did not materialise. Both
   configurations served and completed 8/8 requests.
2. **On GLM, once graphs are on, speculation costs 1.38×** (20.40 vs 14.74). That
   is consistent with the acceptance of 1.00 measured earlier on this model: the
   engine pays for a draft pass and banks nothing, and with graphs the baseline
   got so much cheaper that the drafting overhead now dominates.

Do not read that as a verdict on K3. There, DSpark accepts (1.577× measured on
real weights), so the arithmetic is different — the draft has to be worth its
overhead, and on GLM's MTP head it is not. What transfers is the mechanism:
graphs work, speculation does not switch them off, and the two levers must be
measured together rather than assumed additive.

## Where the K3 rig number stands

```
                        per token     per agent
eager , 50 stages        ~300 ms       3.3 tok/s     measured decomposition
graphs, 50 stages        ~120 ms       8.3 tok/s     using GLM's 5.1x on model work
graphs + spec decode      ~76 ms      13.0 tok/s     if K3 keeps its 1.577x
```

The last line is the one to distrust most: on GLM speculation stopped paying once
graphs were on. K3's draft actually accepts, so it should still pay there — but
that is exactly the pair of levers this campaign could not measure together on
K3, because the slice needs ≥160 GB of disk and this box had 80.
