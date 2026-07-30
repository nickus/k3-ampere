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

## 3. Serve

```bash
./build/bin/llama-server \
  -m /data/k3-gguf/UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL-00001-of-*.gguf \
  --n-gpu-layers 999 --split-mode layer \
  --ctx-size 131072 --parallel <N_agents> \
  --temp 1.0 --top-p 0.95 \
  --host 0.0.0.0 --port 18080
```

Flags to tune after the batching deep-check lands: `--cache-reuse`,
kv-unified, ubatch sizing. K3 sampling per unsloth: temp 1.0 / top-p 0.95
(NOT greedy). GLM-5.2 reference runs on the existing vLLM stack unchanged.

## 4. A/B protocol

- Same task set through both endpoints (OpenAI-compat both sides).
- Tasks: Nick's real coding-agent scenarios + a public anchor (e.g. a
  polyglot/aider subset) for calibration.
- Score: task success (did it work), not vibes; 2–3 attempts per task at
  the model's recommended sampling; record tok/s incidentally.
- Verdict rule (pre-registered): K3-Q2 must WIN on task success to justify
  any further K3 engineering (route b). Tie or loss → GLM-5.2 stays, K3
  topic closes until better quants/kernels/cards.
