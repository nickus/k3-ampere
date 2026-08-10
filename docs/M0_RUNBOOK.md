# M0 gate runbook — K3 UD-Q2_K_XL vs GLM-5.2 INT4 (draft)

Goal: task-level A/B on OUR eval set. Speed is irrelevant; quality verdict is.
Node requirement: RAM+VRAM ≥ ~880 GB for UD-Q2_K_XL (mmap from NVMe works,
slower). Cheap recon variant: UD-IQ1_S (594 GB) — weaker (KLD 0.56), use only
to smoke the pipeline, not for the verdict.

## 1. Build (unsloth fork — has vision + K3 fixes)

```bash
git clone https://github.com/unslothai/llama.cpp
cd llama.cpp
git fetch origin pull/48/head:kimi-k3 && git checkout kimi-k3
cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release -j --target llama-cli llama-server llama-gguf-split
```

sm_86 gotchas (from our 2×3090 bring-up, apply if the image is trimmed):
pip `cmake>=3.26`; `ln -s libnvrtc.so.13 /usr/local/cuda/lib64/libnvrtc.so`;
`CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include` if
cusparse.h missing; transformers-new: `bytes_to_unicode` import fix in
conversion code (not needed for pre-made GGUFs).

## 2. Download (onto the big volume, ~861 GB)

```bash
hf download unsloth/Kimi-K3-GGUF --include "UD-Q2_K_XL/*" --local-dir /data/k3-gguf
```

## 3. PRE-FLIGHT (mandatory, before any eval token)

llama.cpp Issue #20052: dual RTX 3090 **without P2P** + layer-split →
**incoherent output past ~2048 ctx** (open, unresolved; 3090 P2P is
driver-blocked by default = exactly our topology). If this reproduces,
every M0 score is garbage. Test: same >4k-token prompt, greedy, on
(a) 1 GPU, (b) 2 GPUs layer-split same host, (c) across an RPC boundary —
diff outputs. Divergence in (b)/(c) = STOP, file/track upstream first.

## 4. Serve (flags corrected per source-verified batching deep-check)

```bash
./build/bin/llama-server \
  -m /data/k3-gguf/UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL-00001-of-*.gguf \
  --n-gpu-layers 999 --split-mode layer \
  --kv-unified --ctx-size <N_agents*100k + headroom> --parallel <N_agents> \
  --cache-ram -1 --cache-reuse 256 --slot-prompt-similarity 0.5 \
  -ub 512 -b 2048 \
  --temp 1.0 --top-p 0.95 --host 0.0.0.0 --port 18080
```

Why: explicit `--parallel N` **disables** kv-unified auto-enable
(server.cpp:151-155) → without `-kvu` the hybrid-arch ubatch builder
(`split_equal`, sequential mode) fragments non-contiguous slots into
1-token ubatches and MoE expert reads amortize across NOTHING. `-kvu`
makes --ctx-size a SHARED pool. `--cache-ram` default is 8 GiB ≈ 2 saved
conversations (one 100k K3 convo ≈ 3.2 GB serialized) — raise it, but
keep host-RAM headroom (limit enforcement is weak on Linux, #22629).
No `-ot`: 861 GB fits VRAM. Consider `--spec-type ngram-map-k4v` (no
draft model needed, suits code). Budget: cold 100k prefill ≈ 15–20 min
(sequential 93-layer chain; pipelining is OFF whenever any RPC device is
present — ggml-rpc.cpp:1874 reports async=false/events=false). M0 is a
multi-day run, not an afternoon.

## 4. A/B protocol

- Same task set through both endpoints (OpenAI-compat both sides).
- Tasks: our real coding-agent scenarios + a public anchor (e.g. a
  polyglot/aider subset) for calibration.
- Score: task success (did it work), not vibes; 2–3 attempts per task at
  the model's recommended sampling; record tok/s incidentally.
- Verdict rule (pre-registered): K3-Q2 must WIN on task success to justify
  any further K3 engineering (route b). Tie or loss → GLM-5.2 stays, K3
  topic closes until better quants/kernels/cards.
