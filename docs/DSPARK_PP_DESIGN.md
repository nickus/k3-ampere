# Making DSpark work under pipeline parallelism — design

Target: remove `NotImplementedError("DSpark does not support pipeline
parallelism.")` (`vllm/v1/worker/gpu/spec_decode/dspark/utils.py:49`) for
Kimi-K3, on a PP-only rig. Upstream request: [vllm#50098](https://github.com/vllm-project/vllm/issues/50098).

Status: **design + patch written, not yet executed on GPUs.**

## There are two independent problems, not one

The guard is one line, but it is standing in front of two different things.

### Problem A — the draft borrows tensors that live on different ranks

`load_dspark_model` wires the draft to the target by direct object reference:

```python
draft_inner.embed_tokens = target_embed      # lives on PP rank 0
draft_model.lm_head      = target_lm_head    # lives on the LAST PP rank
```

The draft checkpoint deliberately ships neither
(`checkpoint_skip_substrs = ("confidence_head", "embed_tokens", "lm_head")`).
Under PP no single rank owns both, so the wiring cannot be satisfied.

### Problem B — the taps the draft eats are thrown away at every PP boundary

This is the one that actually matters, and it is invisible from the guard.

The K3 DSpark draft is **not** a standalone model. It consumes auxiliary hidden
states tapped from five target layers — `target_layer_ids: [7, 31, 47, 63, 87]`
in the published checkpoint — which the target concatenates onto its output and
the draft projects down via `context_proj` / `context_norm`
(`combine_hidden_states`), then turns into its per-layer latent cache
(`precompute_and_store_context_kv`).

In `KimiK3Model.forward` (`vllm/models/kimi_k3/nvidia/model.py`), each rank
happily computes the taps that fall inside its own layer range — and then:

```python
if not get_pp_group().is_last_rank:
    return IntermediateTensors(
        {"hidden_states": hidden_states, "residual": residual}
    )
```

**`aux_hidden_states` is not in that payload.** Every intermediate rank silently
drops the taps it just computed. Only the last rank's taps survive, so at PP>1
the draft can never see four of its five inputs. No amount of fixing Problem A
would make the draft correct — it would make it silently wrong, which is worse.

At the rig's PP degree this is total: layers 7, 31, 47 and 63 all sit on
different cards than layer 87.

## The fix

### B: carry the taps down the pipeline

Add the taps to the inter-stage payload, so they ride to the last rank.

- **Receive side**: a non-first rank pulls `aux_0 … aux_{k-1}` out of
  `intermediate_tensors` alongside `hidden_states`/`residual`.
- **Send side**: a non-last rank forwards `carried + its own` taps.
- **Last rank**: concatenates `carried + own` in layer order — which is exactly
  what the existing single-rank code already does with `aux_hidden_states`.
- **`make_empty_intermediate_tensors`**: must allocate the receive buffers, so
  it needs the count of taps produced *upstream* of this rank —
  `sum(1 for L in aux_hidden_state_layers if L < self.start_layer)`.
  Deterministic per rank, no communication needed to compute it.

Ordering is preserved for free: upstream ranks own strictly lower layer indices,
so `carried + own` is already sorted by layer.

**Cost**: each carried tap is `hidden_size × 2 B` = **14 KB per token**
(hidden 7168, bf16). The tail segment of the pipeline carries all five, i.e.
70 KB/token on top of the existing 28 KB (`hidden_states` + `residual`). At
decode rates of a few hundred tokens/s that is tens of MB/s — comfortably inside
1GbE. It is the prefill burst, not decode, that would want 10GbE.

A cheaper variant exists and is worth noting for later: `context_proj` is linear
over the concatenation, so `W · concat(x₀…x₅) = Σ Wᵢ · xᵢ`. Each rank could
apply its own slice of `context_proj` and accumulate a **single** partial sum,
making the carried payload constant (14 KB/token) regardless of tap count. That
requires slicing draft weights across ranks, so it is the second iteration, not
the first.

### A: give the draft what it needs on one rank

Host the draft on the **last** PP rank — it already owns `lm_head`, and vLLM's
own layer split deliberately gives the last stage fewer layers, so that card has
the most room.

For the embedding, broadcast rank 0's `embed_tokens` to the last rank once at
load time (2.35 GB for 163840 × 7168 in bf16 — a one-off startup transfer, not a
per-step cost). The alternative — re-reading the tensor from the target
checkpoint on the last rank — avoids the transfer but duplicates loader
plumbing; the broadcast is simpler and happens once.

VRAM on the hosting card: 7.12 GB draft + 2.35 GB embedding ≈ **9.5 GB**, plus a
small KV cache for the draft's five MLA layers.

## Test plan (2×3090, PP=2)

The published draft expects `hidden_size 7168` and taps at layers 7…87, which do
not exist in our 4-layer slice, so the test needs a **matched miniature draft**:
same `K3DSparkModel` architecture, `hidden_size` equal to the slice's, five draft
layers, and `target_layer_ids` inside the slice's layer range — generated with
dummy weights exactly like the slice itself.

Gates, in order:
1. **Boots at PP=2** with the guard removed — proves Problem A is handled.
2. **Taps arrive intact**: with PP=2, the tensor the last rank assembles must
   equal the tensor a PP=1 run assembles for the same input. This is the real
   test of Problem B, and it is checkable to bit-exactness.
3. **Drafts are actually accepted** — acceptance counters non-zero, output
   still correct against a no-spec run.

Gate 2 is the one that must not be skipped: Problem B fails *silently*, so a
"it boots and generates" result proves nothing.

## PP=2 is not sufficient to prove this patch

A two-stage pipeline has no **middle** rank: rank 0 receives nothing (it is
first) and rank 1 forwards nothing (it is last). The branch that actually
carries state across the pipeline — *receive carried taps, append your own,
forward the union* — is only exercised when a rank is neither first nor last,
i.e. at **PP ≥ 3**.

On the target rig that is the common case, not the corner: at PP=47, forty-five
of forty-seven ranks are middle ranks. A patch validated only at PP=2 would be
validated on the one topology where its main path never runs.

Therefore the proof obligation is:

| Topology | What it establishes |
|---|---|
| PP=2 | taps cross a boundary at all; ends of the pipeline behave |
| **PP≥3** | **carried + own forwarding, ordering across multiple hops, buffer sizing on a rank with upstream taps** |

`make_empty_intermediate_tensors` sizes receive buffers as
`sum(1 for L in aux_hidden_state_layers if L < self.start_layer)` — at PP=2 that
sum is 0 on rank 0 and the full set on rank 1, so the *arithmetic* is never
tested against an intermediate value either.
