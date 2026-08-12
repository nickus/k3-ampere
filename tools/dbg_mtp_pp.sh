#!/bin/bash
# Locate the true source of the CUDA fault under MTP + PP.
#
# The reported frame is our relay gather in model_runner.sample_tokens, but the
# identical gather appears three lines below it in stock vLLM, so the index
# expression cannot be inherently invalid. An illegal access is reported
# asynchronously at the next synchronising op, so the first suspect is a kernel
# that ran EARLIER - inside the MTP proposer or its attention - and our line is
# merely where it surfaces. CUDA_LAUNCH_BLOCKING makes the report land on the
# kernel that actually faulted.
set -u
cd /workspace/k3
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

PORT=18500
LOG=/workspace/k3/bench/dbg_mtp.log

/venv/nm/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/glm45air --served-model-name m --trust-remote-code \
  --pipeline-parallel-size 4 --tensor-parallel-size 1 --max-model-len 4096 \
  --no-async-scheduling --max-num-seqs 4 --gpu-memory-utilization 0.90 \
  --enforce-eager --port $PORT \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  > "$LOG" 2>&1 &
SRV=$!

for i in $(seq 1 120); do
  sleep 10
  curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"' && break
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED DURING STARTUP"; exit 1; }
done

echo "=== sending one short request"
curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","prompt":"Write a Python function that adds two numbers.","max_tokens":16,"temperature":0}' \
  | head -c 400
echo
echo "=== done; server log holds the blocking traceback"
