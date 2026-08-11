# Kimi-K3 on 50×RTX 3090 — research journal

The consolidated record: what is proven, what is disproven, what was retracted,
what each experiment actually measured, and what is still unknown. Written to be
readable months later by someone with no memory of the sessions.

**Standing constraints** (these shape every decision below):
- Target fleet **50 × RTX 3090** = 1200 GB raw, **~1160 GB usable**, sm_86 —
  no native fp8, no FlashAttention-3, no Hopper/Blackwell kernels.
- **Pipeline parallelism only.** Tensor parallelism over PCIe was measured to be
  a trap during the GLM-5.2 campaign.
- **The target rig is not built yet.** **Every measurement in this repo was
  taken on rented boxes**, mostly a 4-layer synthetic slice on 2×3090. Nothing
  here has run at full scale.
- Serving target is **Kimi-K3 only**. Workload is a swarm of ~100 coding agents.

---

## 1. Model facts (fetched from the live config, not remembered)

| Field | Value |
|---|---|
| Total / active params | 2.8T / 104B |
| `num_hidden_layers` | 93 |
| Attention layers (`full_attn_layers`) | 24 — every 4th (4, 8, … 92), gated MLA |
| KDA layers (linear attention) | 69 |
| `num_experts` / per token | 896 / 16 ("Stable LatentMoE") |
| `hidden_size` | 7168 |
| KDA `num_heads` / `head_dim` | 96 / 128 |
| `max_position_embeddings` | 1,048,576 |
| Native weight format | MXFP4 (`compressed-tensors`, `mxfp4-pack-quantized`), experts only |
| `num_nextn_predict_layers` | **0 — no MTP head ships with the model** |

Derived, exact:
- **KDA recurrent state = 96 × 128 × 128 × 4 B = 6.29 MB per layer**, × 69 =
  **434 MB per session, constant in context length.** That is the property that
  makes long-context offload cheap.
- **MLA KV at 1M tokens**: 15.74 GB with `fp8_ds_mla` (656 B/token/layer × 24),
  27.65 GB at bf16 (1152 B). → a sleeping 1M-token session costs **≈16.2 GB**,
  100 of them ≈ **1.62 TB**.

---

## 2. What is PROVEN on hardware

Everything here was executed on rented GPUs, not reasoned about.

### 2.1 The K3 stack runs on sm_86 at all — with zero kernel patches
Phase A (bf16) and Phase B (MXFP4→Marlin) both GREEN at PP=2 on 2×3090, with
PIECEWISE CUDA graphs 51/51 and exactly three Python patches.
The sm_86 gaps resolve by fallback, not by porting:
KDA → vendored Triton, MoE MXFP4 → Marlin W4A16, MLA decode → TRITON_MLA,
MLA prefill → FA2. `FlashKDA`/`fused_kda_decode` are **boolean selectors, not
asserts** (`kimi_k3/nvidia/kda.py`), so on 8.6 they simply return False.
→ `results/2x3090_validation_2026-07-30.md`

**This matters more than it looks**: vLLM's own launch post scopes K3 support to
Hopper/Blackwell + MI355X. Ampere is not on that list. We are the counter-example.

### 2.2 Capacity is hardware-anchored, not estimated
One MoE layer at 896 experts = **15.72 GB packed** — fits a 3090, two OOM.
So full MXFP4 K3 = whole-layer PP = **~93 cards**. Full checkpoint ≈1601 GB
against 1160 GB usable: **does not fit 50 cards.** This is the constraint that
drove everything in §4.

### 2.3 fp8-KV port for K3 — written by us, correct
`fp8_ds_mla` packs 656 B/token/layer (512 fp8 nope + 4 fp32 tile scales + 64
bf16 rope) vs 1152 B at bf16 → **1.75× KV capacity**. Upstream refused it on
sm_86 because it gates fp8 KV behind SM89 native fp8; ours is *storage-only* fp8
with software dequant, so the gate is wrong for this dtype.
Results: kernel 12/12 on GPU, **cosine 0.9999967**, e2e boot, greedy output
**token-identical** to bf16, and identical again under PP=2.
→ `fp8kv_k3_port/RESULTS.md`

### 2.4 KV offload to NVMe works and is bit-exact — the "must-have" is closed
Hybrid (KDA + MLA) + PP=2 + CPU tier + filesystem/NVMe tier + `fp8_ds_mla`, all
working, with **max logprob delta 0.00000000** between a fresh run and one
restored after forced GPU-cache eviction. Verified at 4,063 / 8,107 / 16,247 /
28,417-token prompts, including rows that provably read off the disk
(79 MB at 482 MB/s measured from `/proc/<pid>/io`).

The fear that a resumed session would keep MLA-KV but lose KDA recurrent state —
silently corrupting context — is **disproven**. The state is the complete
Markovian summary of the sequence, stored as raw bytes: copy out, copy in.

Required launch flags: `--enable-prefix-caching`, an **explicit `--block-size`**,
and `cpu_bytes_to_use` (not `num_cpu_blocks`).
→ `results/kv_offload_PROVEN_2026-08-11.md`

**Do not quote the speedups.** The measured 1.2–1.9× is an artifact of the test
slice: 4 layers of dummy weights prefill at ~130k tok/s, so there is nothing to
save. On real K3 prefill costs orders of magnitude more per token while restore
scales only with KV bytes. This slice cannot measure the real win.

### 2.5 The stack is current, not pinned to something stale
Re-run on **vLLM main (`a311916a2`)**: works with **zero patches**, bit-exact.
On **release 0.27.0**: works with **one line** (`idx_mapping.long()`).
Our four fp8 patches applied cleanly to *both* trees — and they assert on anchor
drift, so that is a real check, not a silent no-op.
→ `results/revalidation_vllm_0270_2026-08-11.md`

### 2.6 W2 kernels are ready even though W2 weights are not
M1 gate: Humming W2A16 is **0.87–1.01× the time of W4** on a 3090 — 2 bits costs
nothing in time, so 2 layers/card would have been free.
W3 (vellum's int3/g64 geometry) is **servable on 3090 after repack**, cosine
**1.000000**; CT packs 3-bit word-wise, Humming wants bit-continuous.
→ `results/M1_humming_w2_3090_2026-07-31.md`, `results/w3_validation_2026-08-10.md`

### 2.7 llama.cpp route works, as a quality-gate vehicle only
K3 arch builds and runs on sm_86; slice-scale tg 962 tok/s; full-K3
extrapolation ~20–30 tok/s single-stream. Role: **M0 quality gate**, not serving.
→ `results/llamacpp_q2_validation_2026-07-30.md`

---

## 3. What was DISPROVEN — including two of my own published verdicts

Kept deliberately. Both were confidently wrong and both were retracted after
better evidence.

| Claim once published here | Reality |
|---|---|
| "K3-Q2 lost an A/B to GLM-5.2 Q5, so 2-bit K3 is doubtful" | One tester, one task, a homemade quant. Not evidence. |
| "KV offload is NOT ready for K3-hybrid — architectural gap" (2026-08-10) | **Wrong.** Three config requirements plus one upstream bug, not an architectural gap. Everything works. Superseded the next day by §2.4. |
| "The `index_fill_` crash is already fixed upstream, nothing to report" | Fixed **on main only**. Release 0.27.0, published 2026-08-10, still ships the broken call. Release-pinned deployments still need the patch. |
| "LMCache's 'not bit-exact after restore' warning means hybrid restore is lossy" | That caveat is about batch-composition nondeterminism in their connector, not state fidelity. Our path is bit-exact. |

Also retracted along the way: an assertion that a verification had been "verified
by execution" when it had not, and one counter-verification that was itself
wrong (`grep -c SITU` counted dispatcher mentions rather than whitelist entries;
the correct check anchors on `return activation in [`).

---

## 4. The capacity problem, and the route change of 2026-08-11

**The problem**: full K3 MXFP4 ≈1601 GB ≫ 1160 GB usable. K3 does not fit 50
cards. Everything else is downstream of this.

**Old plan**: produce a ~883 GB **W2** checkpoint ourselves via GSQ (2 layers per
card, ~47 cards). Blockers: the checkpoint exists nowhere, GSQ needs a kimi_k3
wrapper + mxfp4 reader + a `wbits=4` label bug fixed, **and serving 2-bit MoE
depends on upstream PR #48918**, which has been open since 2026-07-22 with merge
conflicts and zero approvals.

**New plan (found 2026-08-11)**: the checkpoint already exists.

| Repo | Size | Format | Experts |
|---|---|---|---|
| `runrunway/Kimi-K3-REAP-448experts` | **837.1 GB** | compressed-tensors / mxfp4-pack-quantized | 448, top-16, 93 layers |
| `runrunway/Kimi-K3-REAP-384experts` | 733.7 GB | same | 384 |

Same architecture, same format, same top-k as stock K3 — **the exact format we
already serve on sm_86 via Marlin**. Fits 50 cards at 16.7 GB/card, leaving
~320 GB for KV. Reaches the target size **at 4 bits, by pruning experts rather
than dropping precision**, and therefore has **no dependency on PR #48918**.

REAP = Router-weighted Expert Activation Pruning (Cerebras, arXiv 2510.13999).
Published retention at 50% pruning, on other models: 95.9% coding average
(Qwen3-30B-A3B), 96.7% SWE-bench (Qwen3-Coder-480B), "near-lossless" on code for
Kimi-K2 + W4A16. Our workload is coding agents, which is exactly where these
numbers hold.

**Not established**: the REAP paper predates K3 and never tested it; every
K3-REAP artifact is a third party applying the formula; and `runrunway`'s
artifact has no published evaluation. **Calibration corpus is a selection
criterion** — a Japanese-calibrated cut shares only ~473 of 640 experts with an
English/code one — but since the workload is English + mainstream programming
languages, picking an English/code-calibrated cut makes the published retention
numbers directly applicable rather than something to hedge against.

**Naming trap**: `REAP640`/`REAP576`/`REAP-320` = experts **kept**;
`REAP50` = percent **pruned** (so REAP50 = 448 kept). Do not compare vendors by
the number in the name.

→ `docs/REAP_ROUTE.md`

### Alternatives examined and set aside
- `QuantTrio/Kimi-K3-Cubic-2.5Bit`, 965 GB safetensors, 2.4986 effective bits.
  Genuinely interesting: `get_min_capability()` returns **80**, so Ampere is in
  scope, and dequant is pure Triton (a cubic polynomial per group — which is
  also why it can never reuse Marlin/WNA16, those are hard-wired affine).
  **But it requires QuantTrio's hard fork of vLLM 0.26.1**, cut 2026-08-05,
  before mainline's mature K3 support landed. Deprioritized on integration risk.
- `vessl/Kimi-K3-W4AFP8` 1484 GB, `RedHatAI/Kimi-K3` 1561 GB,
  `vellum-ai/Kimi-K3-W3A16-g64` 1291 GB — all too big for 1160 GB.
- All GGUF variants (REAP-512GB, Neuron IQ1/IQ2, mmnga REAP50-UD) — **not vLLM
  candidates at all**, regardless of size. Useful only as M0 proxies.

---

## 5. Throughput: K3 currently has no speculative-decoding lever

This is the least comfortable finding in the repo.

MTP was *the* lever in the GLM-5.2 campaign (a full k=0..3 cost curve was
measured). **It does not transfer**: K3 ships `num_nextn_predict_layers = 0` —
there is no MTP head.

The only alternative is **DSpark**, and its draft appeared on HF on 2026-08-11:
`lightseekorg/kimi-k3-dspark`, 7.12 GB, five-layer MLA backbone, low-rank Markov
head, confidence head, **up to 7 draft tokens per step** (paper: arXiv
2607.05147). vLLM main registers it (`K3DSparkModel` →
`vllm.models.kimi_k3.nvidia.dspark_mla`) and it reuses the **same
`MultiHeadLatentAttention` we already run on sm_86**, with **no
compute-capability gates anywhere in the file**.

**But**: `vllm/v1/worker/gpu/spec_decode/dspark/utils.py:49` —

```python
if get_pp_group().world_size != 1:
    raise NotImplementedError("DSpark does not support pipeline parallelism.")
```

Root cause, read from the code immediately below that guard: the draft is wired
to the target by **direct Python object reference** —
`draft_inner.embed_tokens = target_embed`, `draft_model.lm_head =
target_lm_head` — and the draft checkpoint deliberately omits both. Under PP,
rank 0 owns the embedding and the last rank owns the LM head; **no rank owns
both**, so the assignment cannot be satisfied.

Fix sketch: host the draft on the **last** PP rank (which already has the LM
head) and give it a private embedding copy — 163840 × 7168 bf16 = **2.35 GB** —
for ≈**9.5 GB on one card out of fifty**. Upstream [#50098] is open; its sole
commenter is blocked for lack of multi-GPU hardware, which we rent for cents.

**Consequence**: until this is solved, every throughput projection for the
100-agent swarm that assumes speculation is unfounded. It does not affect
correctness, and it does not affect the KV-offload result.

### Status after running it on 2×3090: the core fix works, two pieces remain

**It is not one guard — it is 13 obstacles in series**, six upstream and seven
ours, each only visible once the previous is removed. Full map with file:line in
[`docs/DSPARK_PP_LADDER.md`](docs/DSPARK_PP_LADDER.md). That ratio is itself the
lesson: most of the cost was not the upstream fix but building an environment
faithful enough to exercise it.

**Proven on hardware: the taps now cross stage boundaries.** The evidence came
from a failure message, not a green check:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (2048x3072 and 4096x1024)
```

`context_proj` expected 4 taps (4 × 1024), got 3. With taps at layers 2, 3, 4
under PP=2, layer 2 lives on **rank 0** — so one of those three arrived from the
other GPU. Before the patch it was discarded at the boundary. A tensor dimension
cannot be faked.

**Two real upstream defects found by execution, not reading**: the tap dropping
(above) and `dspark_mla.py:419`, where the draft offsets its layer names by the
layer count *on this rank*, so under PP it collides with the target's own layers
(`Duplicate layer name: model.layers.2.self_attn`) — invisible at PP=1 where the
two counts coincide. Plus a robustness bug: optional usage telemetry can kill
the engine ([vllm#51825](https://github.com/vllm-project/vllm/issues/51825)).

**Not finished, and not to be overstated.** Gate 2 — tap fingerprints matching
between PP=1 and PP=2 — has never run, so "they cross" is established but
"they arrive intact and ordered" is not. And the embedding transfer is only
worked around: `load_dspark_model` executes **only on the last rank**
(`init_speculator` runs under `if self.is_last_pp_rank`), so no collective
inside it can ever be symmetric — attempting one hangs the group. The draft rank
currently builds an uninitialised placeholder with a loud warning, which
exercises the plumbing and would produce meaningless draft tokens in
production. The real fix must move the weight where every rank is present.

Posted to [vllm#50098](https://github.com/vllm-project/vllm/issues/50098#issuecomment-5254101547)
with the ladder and a direct question to maintainers about where that transfer
belongs.

### The patch itself, and what was verifiable without a GPU

Reading the code turned up the **second** defect that the guard hides, and it is
the more dangerous one. K3's DSpark draft is not standalone — it consumes
auxiliary hidden states tapped from five target layers. In
`KimiK3Model.forward`, a non-last rank returns
`IntermediateTensors({"hidden_states": …, "residual": …})` and **nothing else**,
so every intermediate stage silently discards the taps it just computed. Fixing
only the embedding/lm_head sharing would make DSpark initialise and generate
while conditioning on one tap out of five — strictly worse than today's honest
refusal.

`tools/dspark_pp_patch.py` fixes both: it carries the taps in
`IntermediateTensors` (receive → forward → reassemble, with receive buffers
sized locally as `sum(1 for L in aux_hidden_state_layers if L < start_layer)`),
and hosts the draft on the last rank with a one-time broadcast of rank 0's
embedding. Verified without renting anything: all six anchors match real main
(`a311916a2`), both patched files parse, a second run is a no-op, and the
mechanism checks out — `IntermediateTensors.tensors` is a plain dict, the V2
runner pre-allocates receive buffers through exactly the function being patched
(`model_runner.py:417`), and the transport iterates the dict generically, so
extra keys ride along and get sliced per token like the rest.

`tools/dspark_pp_test.sh` runs three gates. Gate 2 — tap fingerprints identical
at PP=1 and PP=2, using a position-weighted checksum so a *reordered* tap set is
distinguishable from a correct one — is the only one that proves anything, since
the defect is silent. `tools/dspark_pp_probe.py` instruments
`combine_hidden_states`, the single funnel all taps pass through.
`tools/gen_dspark_draft.py` builds a `K3DSparkModel` config scaled to the slice,
because the published draft wants hidden 7168 and taps at layers 7–87.

→ `docs/DSPARK_PP_BLOCKER.md`, `docs/DSPARK_PP_DESIGN.md`, task #32

---

## 6. Upstream: what we filed, what we found, what is blocking

| Item | State (2026-08-11) |
|---|---|
| **[#51752](https://github.com/vllm-project/vllm/issues/51752)** — hybrid block-size alignment skipped on PP ranks with no attention layer. **Ours.** | Open, no maintainer response. Reproduced on 0.27.0 *and* newest main. |
| **[#50947](https://github.com/vllm-project/vllm/issues/50947)** — `index_fill_` int64 crash | Open since 08-04, filed by someone else. Fixed on main by **PR #50327** (merged 08-03), which was never linked to the issue. **0.27.0 still ships the bug.** We commented with the release repro and a backport request. |
| **PR #48918** — Humming for WNA16 MoE (gates 2-/3-bit MoE serving) | Open, last touched 07-22, `mergeable_state: dirty`, 7 reviewers, **0 approvals**. The REAP route removes our dependence on it. |
| **PR #50653 / #50821** — `num_cpu_blocks` diverging across PP ranks | Both open; PR dirty since 08-04. A different bug from #51752 (one level later); relevant at high PP. |
| **[#50098](https://github.com/vllm-project/vllm/issues/50098)** — DSpark PP support | Open, contributor blocked on hardware. |

### The #51752 blast radius scales with PP — this is the important part
Counting stages that own no attention layer, using vLLM's own `get_pp_indices`
split of K3's 93 layers (24 attention, every 4th):

| PP size | stages with no attention layer |
|---|---|
| 25 | 2 of 25 (8%) |
| **47** (2 layers/card) | **23 of 47 (49%)** |
| **50** | **26 of 50 (52%)** |
| 93 (1 layer/card) | 69 of 93 (74%) |

At PP=2 it reads like a corner case; at rig scale it hits **the majority of
ranks**. Practical consequence: `--block-size N` is mandatory in every launch,
not an optional workaround. `blocks_per_chunk` — which the offloading docs
describe as the knob "for models whose KV cache groups have different block
sizes", i.e. apparently aimed at hybrids — **does not work around it**, verified
by experiment; the divergence happens before the connector is consulted.

---

## 7. Serving shape for 100 agents (model, not measurement)

- Per sleeping 1M-token session: **16.2 GB** (15.74 MLA-KV fp8 + 0.43 KDA).
  100 sessions ≈ **1.62 TB** → fits a 4 TB NVMe tier, or per-node NVMe.
- With REAP-448 on 50 cards: 16.7 GB/card of weights leaves ~6.4 GB/card, so
  **~320 GB of resident KV** ≈ 20M tokens live on the GPUs.
- Wake latency for a 1M session: 15.7 GB of KV. At the **482 MB/s** actually
  measured on a shared vast.ai disk that is ~33 s; on a dedicated Gen4 ×4 drive
  (~7 GB/s) ~2.3 s; striped per node and pulled in parallel by the 24
  MLA-owning ranks, well under a second. Versus 15+ minutes to re-prefill.
  **Per-node NVMe is therefore an architecture requirement, not a detail.**
- Unmeasured: per-agent throughput at 100-way concurrency. K3's 896-expert
  top-16 routing amortizes poorly with batch, so expect expert-read-bound
  decode. With no working speculative decoding (§5), there is currently no
  lever against this.

---

## 8. Operational lessons (each one cost real time or money)

**Rented boxes**
- vast.ai **stopped instances are not durable** — box 46282900 was evicted with
  its 1 TB volume after 10 idle days, taking a wheel and all on-box scripts.
  Pull artifacts the same day; destroy rather than stop.
- Discard bad boxes fast. In one session: box 1 accepted no SSH key, box 2 could
  not pull a Docker image. Both killed within minutes, cents lost. Waiting would
  have cost more than re-renting.
- **Pull the whole evidence bundle before destroying**, not just the last file.
  Learned by losing the raw logs behind a published results table; the numbers
  were reported accurately but could no longer be re-derived from an artifact.

**Remote shell traps**
- `pkill -f "foo.sh"` / `pgrep -f "api_server"` executed over SSH **matches the
  SSH command's own command line and kills the launcher**. Cost two failed
  launches that looked like silent no-ops. Fix: bracket the first character
  (`[f]oo.sh`), or put the launch in a script file on the box so the pattern
  never appears in the invoking command.
- Never start a second test battery while the first is running — they kill each
  other's servers through the same cleanup logic.
- A `scp` can silently truncate (96 MB arrived for a 476 MB wheel). Verify size
  and `zipfile.testzip()` after every transfer.

**Patch hygiene**
- Patch scripts that print DONE while applying zero patches are worse than
  failures. `apply_vllm_patches.py` asserts on anchor drift — keep it that way.
  Verify with a regex anchored to real syntax, not a keyword count.

**Environment drift**
- Working 11 days behind main cost two of three "upstream findings" — both were
  already fixed. **Re-verify every finding against a fresh checkout immediately
  before filing, and cite the SHA.**
- `pip install vllm --pre` resolves to the **release**, not to main. To get main,
  install from `https://wheels.vllm.ai/nightly/cu130` and then **check
  `vllm.__version__`** — a hand-built wheel URL with a literal `+` 404s, and the
  failure is silent.
- Upstream model configs drift. K3's config went multimodal, so vLLM ≥0.27.0
  resolves `KimiK3ForConditionalGeneration`, demands an image processor plus
  five remote-code files, and then OOM-kills building a dummy vision tower.
  Our slice generator now emits a **flat, text-only** config with
  `architectures: ["KimiLinearForCausalLM"]` — and flat matters: nested under
  `text_config` it fails with `'KimiK3Config' object has no attribute
  'linear_attn_config'`.

**Research discipline**
- Independent agents given no project context are worth their cost precisely
  because they contradict you. One correctly flagged that vLLM does not
  officially support K3 on Ampere — true, and we are the counter-evidence,
  which is only a strong statement because the check was blind.
- Cheap models for subagents; the expensive one stays in the main loop.

---

## 9. Open questions, ranked by what they decide

1. **M0 — is K3 actually better for our work than GLM-5.2?** Nothing else
   matters if the answer is no. Contender is now **REAP-448 MXFP4**, not a
   homemade 2-bit quant. Eval set = real agentic coding sessions, not synthetic
   benchmarks. Scope is **English + mainstream programming languages only** —
   which is exactly the domain REAP's published retention numbers cover, so pick
   an English/code-calibrated cut and they apply directly. Cost researched:
   ~$28–90 for a CPU/GGUF proxy, **~$200 to test the real artifact** on rented
   GPUs. It gates a large hardware commitment, so test the real one. (Task #30)
2. **DSpark under PP** — the only throughput lever K3 has. Fix is understood and
   cheap in VRAM; upstream wants it and is blocked on hardware. (Task #32)
3. **Does REAP-448 boot and serve on our stack with real weights?** Structurally
   nothing new — our slice has always run a reduced expert count through the
   stock `kimi_k3` code — but unverified for this artifact.
4. **Per-agent throughput at 100-way concurrency.** Entirely unmeasured.
5. **Multi-node**: NVMe locality, PP across hosts, and #50821's per-rank block
   accounting at PP≫2.
6. W2 via GSQ — now a fallback if REAP-448 fails M0. (Task #31)

---

## 10. Index

| Document | Contents |
|---|---|
| `docs/REAP_ROUTE.md` | The 837 GB checkpoint that changes the plan |
| `docs/DSPARK_PP_BLOCKER.md` | Why K3 has no speed lever, and how to build one |
| `docs/RIG_NUMBERS_VERIFIED.md` | Exact per-session memory, PP blast radius |
| `docs/LESSONS.md` | Twenty things worth knowing before starting this again |
| `docs/DSPARK_PP_LADDER.md` | The 13-obstacle ladder behind one NotImplementedError |
| `docs/UPSTREAM_FILING_RECORD.md` | What was filed, what evaporated on re-check |
| `docs/CAPACITY_50x3090.md` | Original capacity arithmetic |
| `docs/GAP_ANALYSIS.md` | The sm_86 gap analysis (15-agent, adversarial) |
| `docs/W23_KERNEL_VERDICT.md` | Why we do not write our own W2/W3 kernel |
| `docs/GSQ_W2_PLAN.md` | The W2 production plan (now a fallback) |
| `docs/MTP_PP2_DEADLOCK.md` | The GLM-5.2-era MTP+PP init deadlock |
| `docs/M0_RUNBOOK.md` | How to run the quality gate |
| `results/kv_offload_PROVEN_2026-08-11.md` | Bit-exact NVMe offload, long-context sweep |
| `results/revalidation_vllm_0270_2026-08-11.md` | 0.27.0 vs main, patch counts |
| `results/2x3090_validation_2026-07-30.md` | The original sm_86 bring-up |
| `results/w3_validation_2026-08-10.md` | W3-g64 servable after repack |
| `results/M1_humming_w2_3090_2026-07-31.md` | W2 == W4 in time |
| `results/llamacpp_q2_validation_2026-07-30.md` | The GGUF route |
| `results/negative_matrix.md` | Configurations that do **not** work |
| `fp8kv_k3_port/RESULTS.md` | The fp8-KV port, kernel through e2e |
| `tools/offload_repro/` | Scripts behind the offload results |
