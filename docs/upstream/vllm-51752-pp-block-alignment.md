### Your current environment

```
Hardware: 2 × NVIDIA RTX 3090 (sm_86), single node
vLLM: built from source @38a267cdd; bug re-verified by code inspection on main @1a1727330a
Distributed executor: mp, --pipeline-parallel-size 2, --tensor-parallel-size 1
Model: Kimi-K3 (hybrid: KDA linear attention + MLA), 4-layer slice, --load-format dummy
KV offloading: OffloadingConnector (TieringOffloadingSpec, CPU + fs tiers), prefix caching enabled
```

### 🐛 Describe the bug

For hybrid (mamba / linear-attention + attention) models, the mamba↔attention block-size
alignment is computed **independently in every worker process**, from the layers that worker
actually instantiated. Under pipeline parallelism a rank whose stage contains **no attention
layer** skips the alignment entirely and keeps the unaligned block size, while ranks that do own
an attention layer raise it. The result is KV-cache groups with mismatched block sizes and a
startup failure whose error message points at the wrong cause.

#### Mechanism

`Platform.update_block_size_for_backend` is invoked **per worker, after `load_model()`**
(`vllm/v1/executor/multiproc_executor.py:677`; likewise `ray_executor.py:366`,
`uniproc_executor.py:75`), and begins with:

```python
# vllm/platforms/interface.py:624-626
backend_cls = cls._find_non_ssm_backend(vllm_config)
if backend_cls is None:
    return
```

`_find_non_ssm_backend` (`interface.py:589-606`) iterates
`get_layers_from_vllm_config(vllm_config, AttentionLayerBase)` — i.e. this rank's
`static_forward_context` — and returns the first non-SSM backend. Under PP each rank builds only
its own stage; the remaining layers are `PPMissingLayer` and are absent from that context. So a
stage consisting purely of linear-attention/SSM layers returns `None` and returns early,
**skipping `_align_hybrid_block_size`** and leaving `cache_config.block_size` and
`cache_config.mamba_block_size` at their pre-alignment values. Ranks that do own an attention
layer mutate both to the aligned value. Nothing reconciles these fields across ranks afterwards.

Because KV-cache specs are merged by layer **name**, and PP stages have disjoint layer names, the
usual "specs differ across workers" check never fires — the mismatch survives into
`resolve_kv_cache_block_sizes`, which backs off to a common block size, and the offloading
connector's divisibility assert then fires:

```
AssertionError: tokens_per_block=16 not divisible by tokens_per_hash=512.
Hybrid models (e.g. Mamba+Attention) need --enable-prefix-caching to align block sizes.
```

The message is misleading in this configuration: prefix caching **was** enabled. The group stuck
at 16 is a mamba group belonging to the attention-less rank, which never ran alignment.

#### Reproduction

Any hybrid model whose PP split leaves at least one stage without an attention layer. In our case
a 4-layer Kimi-K3 slice (layers 1–3 = KDA linear attention, layer 4 = MLA) with
`--pipeline-parallel-size 2` puts pure-KDA layers on rank 0:

```bash
vllm serve <hybrid-model> --trust-remote-code \
  --pipeline-parallel-size 2 --tensor-parallel-size 1 \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"cpu_bytes_to_use":2147483648}}'
```

The same divergence exists without any connector — the connector's divisibility assert is simply
what makes it visible. Real deployments hit the attention-less-stage case easily: Kimi-K3 has 69
linear-attention layers to 24 attention layers (attention every 4th layer), so many PP splits
produce at least one attention-free stage.

#### Workaround

Pass `--block-size N` explicitly. This sets `user_specified_block_size` in the parent process
before workers are created, so every rank starts from the same value and the divergent mutation
never happens. With `--block-size 512` (bf16 KV) / `1024` (fp8) the full stack — hybrid + PP=2 +
CPU tier + `fs` NVMe tier — boots and serves, and restore is bit-exact (max logprob delta
`0.00000000` between a fresh run and one restored after forced GPU-cache eviction).

#### Suggested fix

Derive the alignment from the **model config** rather than from locally-instantiated layers, or
broadcast the aligned `(block_size, mamba_block_size)` from a rank that owns an attention layer
before any rank allocates. Failing that, at minimum detect the attention-less-stage case and
raise something better than the current divisibility assert.

#### Relationship to #50821 / #50653

Those cover a **different** divergence: `num_cpu_blocks` computed per rank from local KV tensors,
where the scheduler uses rank 0's count. This report is one level earlier — `block_size` /
`mamba_block_size` themselves diverge because the alignment step is skipped on attention-less
ranks, which happens for hybrid models with or without offloading. #50653 would not fix it.

I'm happy to open a PR if maintainers indicate which direction they prefer (config-derived
alignment vs. broadcast).

### Before submitting a new issue...

- [x] I have searched for similar issues and read the documentation.
